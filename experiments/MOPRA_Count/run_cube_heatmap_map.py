#!/usr/bin/env python
"""
Apply a trained center heatmap model to a real PPV cube -> 2D K_pred map.

K comes from prominence peak-picking on sigmoid(heatmap logits), not global regression.

Example:
  python experiments/MOPRA_Count/run_cube_heatmap_map.py \\
    --cube data/CMZ_3mm_HNCO_60.fits \\
    --run-dir experiments/MOPRA_Count/runs/mopra_heatmap_<ts>_heatmap_v1 \\
    --infer-on-scouse-labels \\
    --out data/mopra_cmz_k_pred_heatmap_v1.fits \\
    --compare-scouse
"""
from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from astropy.io import fits
from spectral_cube import SpectralCube

_SCRIPT = Path(__file__).resolve()
_MOPRA = _SCRIPT.parent
_REPO = _MOPRA.parents[1]
_DEFAULT_CUBE = _REPO / "data" / "CMZ_3mm_HNCO.fits"
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_MOPRA / "plots"))

from plot_k_map_image import plot_k_map_figure  ### noqa: E402
from plot_crop_utils import crop_bbox_from_mask, labeled_mask_from_dat  ### noqa: E402
from plot_k_residual_map import compare_k_pred_to_scouse  ### noqa: E402

from spectackle.data.generator import _make_v_axis  ### noqa: E402
from spectackle.data.mopra_preprocess import (  ### noqa: E402
    NORM_MODES,
    prepare_mopra_input,
    snr_peak_rms_mopra,
    snr_peak_scouse_mopra,
)
from spectackle.data.mopra_resample import resample_spec_batch, velocity_axis_kms  ### noqa: E402
from spectackle.models import CenterHeatmapNet1DDeep, HeatmapCountNet  ### noqa: E402
from spectackle.models.center_heatmap_decode import (  ### noqa: E402
    decode_centers_batch_from_heatmap,
    decode_centers_from_heatmap,
)
from spectackle.wcs_plot import wcs_celestial, wcs_header_for_array_cutout  ### noqa: E402


def _load_heatmap_model(run_dir: Path, manifest: dict, *, device: str) -> CenterHeatmapNet1DDeep:
    args = dict(manifest.get("args", {}))
    cfg = manifest["cfg"]
    coord_cfg = manifest.get("coord", {})
    coord = None
    if coord_cfg.get("enabled"):
        v_scale = float(coord_cfg.get("v_scale_kms", 100.0))
        v_axis = _make_v_axis(cfg).astype(np.float32)
        coord = v_axis / v_scale
    model = CenterHeatmapNet1DDeep(
        width=int(args.get("width", 96)),
        n_blocks=int(args.get("n_blocks", 6)),
        coord=coord,
    )
    state = torch.load(run_dir / "center_heatmap_net.pt", map_location="cpu")
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def _load_heatmap_count_model(run_dir: Path, manifest: dict, *, device: str) -> HeatmapCountNet:
    """Stage-2 HeatmapCountNet; Stage-1 weights come from the joint state dict."""
    stage1_dir = Path(manifest["stage1_run_dir"])
    stage1_manifest = json.loads((stage1_dir / "manifest.json").read_text(encoding="utf-8"))
    heatmap = _load_heatmap_model(stage1_dir, stage1_manifest, device="cpu")
    args = dict(manifest.get("args", {}))
    model = HeatmapCountNet(
        heatmap,
        width=int(args.get("width", 96)),
        n_blocks=int(args.get("n_blocks", 6)),
        k_input=str(manifest.get("k_input", args.get("k_input", "spec_p"))),
        freeze_heatmap=bool(manifest.get("freeze_heatmap", True)),
    )
    state = torch.load(run_dir / "heatmap_count_net.pt", map_location="cpu")
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def _decode_cfg(manifest: dict) -> dict:
    ### heatmap_count stores Stage-1 thresholds under decode_stage1.
    dec = dict(manifest.get("decode_stage1") or manifest.get("decode") or {})
    args = dict(manifest.get("args", {}))
    return {
        "height": float(dec.get("height", 0.35)),
        "prominence": float(dec.get("prominence", 0.15)),
        "min_sep_kms": float(dec.get("min_sep_kms", 4.0)),
        "Kmax": int(dec.get("Kmax", args.get("Kmax", 10))),
    }


def _prep_batch(
    spec_raw: np.ndarray,
    *,
    min_finite: int,
    min_peak: float,
    min_snr: float,
    blank_value: float,
    vel_kms: np.ndarray | None = None,
    snr_vel_range: tuple[float, float] | None = None,
    snr_method: str = "scouse",
    norm_mode: str = "zscore",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_norm, chan_mask = prepare_mopra_input(spec_raw, blank_value=blank_value, norm_mode=norm_mode)
    n_valid = chan_mask.sum(axis=1)
    valid_bool = chan_mask > 0.5
    spec_for_peak = np.where(valid_bool, spec_raw, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        peak = np.nanmax(np.abs(spec_for_peak), axis=1)
    peak = np.where(np.isfinite(peak), peak, 0.0)
    if snr_method == "scouse":
        snr = snr_peak_scouse_mopra(
            spec_raw,
            blank_value=blank_value,
            vel_kms=vel_kms,
            vel_range=snr_vel_range,
        )
    elif snr_method == "global":
        snr = snr_peak_rms_mopra(spec_raw, blank_value=blank_value)
    else:
        raise ValueError(f"Unknown snr_method {snr_method!r}")
    pixel_ok = (n_valid >= min_finite) & (peak > min_peak)
    if min_snr > 0.0:
        pixel_ok &= snr >= min_snr
    return x_norm, chan_mask, pixel_ok


@torch.no_grad()
def _kv_batch_heatmap(
    model,
    x_norm: np.ndarray,
    chan_mask: np.ndarray,
    *,
    device: str,
    v_axis: np.ndarray,
    decode: dict,
    k_from_head: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (K_pred, v_slots, p_slots) with slots NaN-padded to Kmax."""
    t = torch.from_numpy(x_norm).to(device)
    m = torch.from_numpy(chan_mask).to(device)
    Kmax = int(decode["Kmax"])
    if k_from_head:
        ### Learned K head; centers = top-K_pred heatmap peaks (by probability).
        k_hat = model(t, m)
        k = torch.clamp(torch.round(k_hat), 0, Kmax).cpu().numpy().astype(np.int64)
        logits = model.heatmap_logits(t, m)
        prob = torch.sigmoid(logits).cpu().numpy()
        B = prob.shape[0]
        v_slots = np.full((B, Kmax), np.nan, dtype=np.float32)
        p_slots = np.full((B, Kmax), np.nan, dtype=np.float32)
        vel = np.asarray(v_axis, dtype=np.float64).reshape(-1)
        for i in range(B):
            ki = int(k[i])
            if ki <= 0:
                continue
            _kd, peak_idx = decode_centers_from_heatmap(
                prob[i],
                vel,
                valid_mask=chan_mask[i],
                height=decode["height"],
                prominence=decode["prominence"],
                min_sep_kms=decode["min_sep_kms"],
                Kmax=ki,
            )
            for j, ix in enumerate(peak_idx.tolist()):
                if j >= Kmax:
                    break
                p_slots[i, j] = float(prob[i, int(ix)])
                v_slots[i, j] = float(vel[int(ix)])
        return k.astype(np.float32), v_slots, p_slots

    logits = model(t, m)
    prob = torch.sigmoid(logits).cpu().numpy()
    k, v_slots, p_slots = decode_centers_batch_from_heatmap(
        prob,
        v_axis,
        valid_mask=chan_mask,
        height=decode["height"],
        prominence=decode["prominence"],
        min_sep_kms=decode["min_sep_kms"],
        Kmax=Kmax,
    )
    return k.astype(np.float32), v_slots, p_slots


def _process_spatial_chunk(
    arr_vyx: np.ndarray,
    model,
    *,
    device: str,
    v_axis: np.ndarray,
    decode: dict,
    min_finite: int,
    min_peak: float,
    min_snr: float,
    blank_value: float,
    infer_batch: int,
    v_src: np.ndarray | None = None,
    v_tgt: np.ndarray | None = None,
    snr_vel_range: tuple[float, float] | None = None,
    snr_method: str = "scouse",
    norm_mode: str = "zscore",
    infer_mask_yx: np.ndarray | None = None,
    save_centers: bool = False,
    k_from_head: bool = False,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, int, int]:
    nv, ny, nx = arr_vyx.shape
    Kmax = int(decode["Kmax"])
    spec = np.transpose(arr_vyx, (1, 2, 0)).reshape(ny * nx, nv).astype(np.float64, copy=False)
    k_flat = np.full(ny * nx, np.nan, dtype=np.float32)
    v_flat = np.full((ny * nx, Kmax), np.nan, dtype=np.float32) if save_centers else None
    p_flat = np.full((ny * nx, Kmax), np.nan, dtype=np.float32) if save_centers else None
    n_infer = 0
    vel_for_snr = v_tgt if v_tgt is not None else v_src
    for i0 in range(0, spec.shape[0], infer_batch):
        i1 = min(i0 + infer_batch, spec.shape[0])
        block = spec[i0:i1]
        if v_tgt is not None:
            block = resample_spec_batch(block, v_src, v_tgt, blank_value=blank_value)
        x_norm, chan_mask, ok = _prep_batch(
            block,
            min_finite=min_finite,
            min_peak=min_peak,
            min_snr=min_snr,
            blank_value=blank_value,
            vel_kms=vel_for_snr,
            snr_vel_range=snr_vel_range,
            snr_method=snr_method,
            norm_mode=norm_mode,
        )
        if infer_mask_yx is not None:
            flat = np.arange(i0, i1)
            cy = flat // nx
            cx = flat % nx
            ok &= infer_mask_yx[cy, cx]
        if not np.any(ok):
            continue
        k, v_slots, p_slots = _kv_batch_heatmap(
            model,
            x_norm[ok],
            chan_mask[ok],
            device=device,
            v_axis=v_axis,
            decode=decode,
            k_from_head=k_from_head,
        )
        block_out = k_flat[i0:i1]
        block_out[ok] = k
        k_flat[i0:i1] = block_out
        if save_centers:
            assert v_flat is not None and p_flat is not None
            v_block = v_flat[i0:i1]
            p_block = p_flat[i0:i1]
            v_block[ok] = v_slots
            p_block[ok] = p_slots
            v_flat[i0:i1] = v_block
            p_flat[i0:i1] = p_block
        n_infer += int(ok.sum())
    n_skip = ny * nx - n_infer
    v_map = None if v_flat is None else v_flat.reshape(ny, nx, Kmax)
    p_map = None if p_flat is None else p_flat.reshape(ny, nx, Kmax)
    return k_flat.reshape(ny, nx), v_map, p_map, n_infer, n_skip


def main() -> None:
    parser = argparse.ArgumentParser(description="MOPRA center-heatmap K map from a real PPV cube.")
    parser.add_argument("--cube", type=Path, default=_DEFAULT_CUBE)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=_REPO / "data" / "mopra_cmz_k_pred_heatmap.fits")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--chunk-y", type=int, default=32)
    parser.add_argument("--chunk-x", type=int, default=64)
    parser.add_argument("--infer-batch", type=int, default=512)
    parser.add_argument("--min-finite-channels", type=int, default=100)
    parser.add_argument("--min-peak", type=float, default=0.02)
    parser.add_argument("--min-snr", type=float, default=3.0)
    parser.add_argument("--snr-method", choices=("scouse", "global"), default="scouse")
    parser.add_argument("--snr-vel-min", type=float, default=40.0)
    parser.add_argument("--snr-vel-max", type=float, default=140.0)
    parser.add_argument("--snr-full-band", action="store_true")
    parser.add_argument("--blank-value", type=float, default=-1.0)
    parser.add_argument("--ref-cube", type=Path, default=None)
    parser.add_argument("--dat", type=Path, default=_REPO / "data" / "final_fits_updated.dat")
    parser.add_argument("--crop-to-labels", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--crop-pad", type=int, default=0)
    parser.add_argument("--compare-scouse", action="store_true")
    parser.add_argument("--infer-on-scouse-labels", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--fig-out", type=Path, default=None)
    parser.add_argument("--decode-height", type=float, default=None)
    parser.add_argument("--decode-prominence", type=float, default=None)
    parser.add_argument(
        "--norm-mode",
        choices=NORM_MODES,
        default=None,
        help="Input norm. Default: read from run manifest (fallback zscore).",
    )
    args = parser.parse_args()

    snr_vel_range: tuple[float, float] | None = None
    if args.snr_method == "scouse" and not args.snr_full_band:
        snr_vel_range = (float(args.snr_vel_min), float(args.snr_vel_max))

    run_dir = args.run_dir.resolve()
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    variant = str(manifest.get("variant", ""))
    if variant not in ("center_heatmap", "heatmap_count"):
        raise ValueError(
            f"Expected center_heatmap or heatmap_count run, got variant={variant!r}"
        )
    k_from_head = variant == "heatmap_count"
    n_channels = int(manifest["cfg"]["n_channels"])
    decode = _decode_cfg(manifest)
    norm_mode = str(
        args.norm_mode
        or manifest.get("norm_mode")
        or manifest.get("cfg", {}).get("norm_mode")
        or manifest.get("args", {}).get("norm_mode")
        or "zscore"
    )
    if args.decode_height is not None:
        decode["height"] = float(args.decode_height)
    if args.decode_prominence is not None:
        decode["prominence"] = float(args.decode_prominence)
    v_axis = _make_v_axis(manifest["cfg"]).astype(np.float64)

    print(f"Loading cube (lazy): {args.cube}", flush=True)
    t0 = time.perf_counter()
    cube = SpectralCube.read(str(args.cube.resolve()), use_dask=True)
    nv, ny, nx = cube.shape
    v_src = cube.spectral_axis.to("km/s").value.astype(np.float64)
    v_tgt: np.ndarray | None = None
    ref_cube_path: Path | None = None
    if nv != n_channels:
        meta = manifest.get("cfg", {}).get("mopra_meta", {})
        ref_default = meta.get("axis_cube") or meta.get("cube_path") or str(args.cube)
        ref_cube_path = (args.ref_cube or Path(ref_default)).resolve()
        v_tgt = velocity_axis_kms(ref_cube_path)
        print(f"Resampling {nv} ch -> {n_channels} ch using {ref_cube_path.name}", flush=True)

    wcs_full = wcs_celestial(cube.header)
    scouse_mask_full: np.ndarray | None = None
    min_snr_infer = float(args.min_snr)
    if args.infer_on_scouse_labels:
        scouse_mask_full = labeled_mask_from_dat(args.dat, shape=(ny, nx), wcs=wcs_full)
        min_snr_infer = 0.0
        print(f"Infer-on-Scouse-labels: {int(scouse_mask_full.sum())} pixels", flush=True)

    print(f"Decode: height={decode['height']} prom={decode['prominence']} min_sep={decode['min_sep_kms']} km/s", flush=True)
    print(f"Norm mode: {norm_mode}", flush=True)
    print(f"Variant: {variant}  K from {'learned head' if k_from_head else 'peak decode'}", flush=True)
    if k_from_head:
        model = _load_heatmap_count_model(run_dir, manifest, device=args.device)
    else:
        model = _load_heatmap_model(run_dir, manifest, device=args.device)

    y0, y1, x0, x1 = 0, ny, 0, nx
    out_ny, out_nx = ny, nx
    Kmax = int(decode["Kmax"])
    k_map = np.full((out_ny, out_nx), np.nan, dtype=np.float32)
    ### Always keep decoded centers for QA overlays (same pixels as finite K_pred).
    v_map = np.full((out_ny, out_nx, Kmax), np.nan, dtype=np.float32)
    p_map = np.full((out_ny, out_nx, Kmax), np.nan, dtype=np.float32)
    n_infer = n_skip = 0
    t_infer = time.perf_counter()
    cy, cx = max(1, args.chunk_y), max(1, args.chunk_x)
    n_chunks = ((out_ny + cy - 1) // cy) * ((out_nx + cx - 1) // cx)
    done = 0
    for ys in range(y0, y1, cy):
        ye = min(ys + cy, y1)
        for xs in range(x0, x1, cx):
            xe = min(xs + cx, x1)
            sub = cube[:, ys:ye, xs:xe]
            arr = sub.filled(np.nan)
            if hasattr(arr, "compute"):
                arr = arr.compute()
            chunk_mask = scouse_mask_full[ys:ye, xs:xe] if scouse_mask_full is not None else None
            chunk_k, chunk_v, chunk_p, ni, ns = _process_spatial_chunk(
                np.asarray(arr, dtype=np.float32),
                model,
                device=args.device,
                v_axis=v_axis if v_tgt is None else v_tgt,
                decode=decode,
                min_finite=args.min_finite_channels,
                min_peak=args.min_peak,
                min_snr=min_snr_infer,
                blank_value=args.blank_value,
                infer_batch=args.infer_batch,
                v_src=v_src,
                v_tgt=v_tgt,
                snr_vel_range=snr_vel_range,
                snr_method=args.snr_method,
                norm_mode=norm_mode,
                infer_mask_yx=chunk_mask,
                save_centers=True,
                k_from_head=k_from_head,
            )
            k_map[ys:ye, xs:xe] = chunk_k
            if chunk_v is not None and chunk_p is not None:
                v_map[ys:ye, xs:xe, :] = chunk_v
                p_map[ys:ye, xs:xe, :] = chunk_p
            n_infer += ni
            n_skip += ns
            done += 1
            if done == 1 or done % 10 == 0 or done == n_chunks:
                print(f"  chunk {done}/{n_chunks}  inferred={ni} skipped={ns}", flush=True)

    wcs_header = wcs_header_for_array_cutout(wcs_full, y0=y0, y1=y1, x0=x0, x1=x1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    hdu = fits.PrimaryHDU(data=k_map, header=wcs_header)
    hdu.header["BUNIT"] = "1"
    hdu.header["COMMENT"] = (
        "HeatmapCountNet K_pred (learned head); NaN = not inferred"
        if k_from_head
        else "Center heatmap K_pred (peak decode); NaN = not inferred"
    )
    hdu.header["KMAX"] = int(decode["Kmax"])
    hdu.header["HMHEIGHT"] = float(decode["height"])
    hdu.header["HMPROM"] = float(decode["prominence"])
    hdu.header["NORMMODE"] = str(norm_mode)
    hdu.header["MOPRARUN"] = run_dir.name[:68]
    hdu.writeto(str(args.out.resolve()), overwrite=True)

    ys, xs = np.where(np.isfinite(k_map))
    centers_out = args.out.with_name(f"{args.out.stem}_centers.npz")
    np.savez_compressed(
        centers_out,
        yi=ys.astype(np.int64),
        xi=xs.astype(np.int64),
        k_pred=k_map[ys, xs].astype(np.float32),
        center_v_kms=v_map[ys, xs, :],
        center_prob=p_map[ys, xs, :],
        decode_height=np.float32(decode["height"]),
        decode_prominence=np.float32(decode["prominence"]),
        decode_min_sep_kms=np.float32(decode["min_sep_kms"]),
        Kmax=np.int32(Kmax),
        run_dir=np.asarray(str(run_dir)),
    )
    print(f"Wrote {centers_out}  ({ys.size} pixels)", flush=True)

    print(f"Wrote {args.out}  ({(time.perf_counter() - t0) / 60:.1f} min wall)", flush=True)
    k_fin = k_map[np.isfinite(k_map)]
    if k_fin.size:
        print(
            f"K stats: min={k_fin.min():.0f} med={np.median(k_fin):.0f} "
            f"max={k_fin.max():.0f} mean={k_fin.mean():.2f}",
            flush=True,
        )

    if not args.no_plot:
        fig_out = args.fig_out or (_MOPRA / "figures" / f"{args.out.stem}.png")
        crop = None
        if args.crop_to_labels and args.dat.is_file():
            labeled = labeled_mask_from_dat(args.dat, shape=(ny, nx), wcs=wcs_full)
            crop = crop_bbox_from_mask(labeled, pad=int(args.crop_pad))
        plot_k_map_figure(
            k_map,
            wcs_header,
            out=fig_out,
            crop=crop,
            title="K_pred (heatmap K head)" if k_from_head else "K_pred (heatmap decode)",
        )
        print(f"Wrote figure {fig_out}", flush=True)

    if args.compare_scouse:
        resid_fig = _MOPRA / "figures" / f"{args.out.stem}_residual.png"
        stats = compare_k_pred_to_scouse(
            k_map,
            dat_path=args.dat,
            cube_path=args.cube,
            out_png=resid_fig,
            crop_to_labels=args.crop_to_labels,
            crop_pad=args.crop_pad,
        )
        print(
            f"Scouse compare: n={stats['n_compare']}  MAE={stats['mae']:.3f}  "
            f"median dK={stats['median_delta']:.1f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
