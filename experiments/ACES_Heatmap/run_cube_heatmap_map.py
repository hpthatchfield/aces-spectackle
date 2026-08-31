#!/usr/bin/env python
"""
Apply ACES heatmap->K (or Stage-1 heatmap) to mosaic cutout (NLW region1).

Uses mosaic spectral axis cropped to the training vel_window, spatial bounds from
--subcube-ref (WCS), same cutout helpers as NLW_Finder.

  python experiments/ACES_Heatmap/run_cube_heatmap_map.py \\
    --run-dir experiments/ACES_Heatmap/runs/aces_heatmap_k_<ts>_aces_hm_k_simple_k6_pm80 \\
    --subcube-ref data/hnco_region1_cube.fits \\
    --out data/hnco_region1_aces_hm_k_pred.fits
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
from astropy import units as u
from astropy.io import fits
from spectral_cube import SpectralCube

_SCRIPT = Path(__file__).resolve()
_ACES = _SCRIPT.parent
_REPO = _ACES.parents[1]
_SHARED = _REPO / "experiments" / "shared"
_DEFAULT_MOSAIC = _REPO / "data" / (
    "group.uid___A001_X1590_X30a9.lp_slongmore.cmz_mosaic.12m7mTP.HNCO_7m12mTP.cube.pbcor.fits"
)
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_SHARED))

from cube_cutout_utils import resolve_spatial_bounds  ### noqa: E402

from spectackle.data.generator import _make_v_axis  ### noqa: E402
from spectackle.data.preprocess import prepare_spectrum_input  ### noqa: E402
from spectackle.models import CenterHeatmapNet1DDeep, HeatmapCountNet  ### noqa: E402
from spectackle.models.center_heatmap_decode import (  ### noqa: E402
    decode_centers_batch_from_heatmap,
    decode_centers_from_heatmap,
)
from spectackle.wcs_plot import wcs_celestial, wcs_header_for_array_cutout  ### noqa: E402


def _channel_indices_for_vrange(
    vel_kms: np.ndarray, vrange: tuple[float, float]
) -> tuple[int, int]:
    """Inclusive-style slice [i0, i1) covering channels whose vel lie in vrange (approx)."""
    vlo, vhi = float(vrange[0]), float(vrange[1])
    if vlo > vhi:
        vlo, vhi = vhi, vlo
    ### Nearest indices to window edges.
    i0 = int(np.argmin(np.abs(vel_kms - vlo)))
    i1 = int(np.argmin(np.abs(vel_kms - vhi))) + 1
    i0 = max(0, min(i0, vel_kms.size - 1))
    i1 = max(i0 + 2, min(i1, vel_kms.size))
    return i0, i1


def _load_heatmap(run_dir: Path, manifest: dict, *, device: str) -> CenterHeatmapNet1DDeep:
    args = dict(manifest.get("args", {}))
    cfg = manifest["cfg"]
    kernel_size = int(manifest.get("kernel_size", args.get("kernel_size", 9)))
    coord = None
    if manifest.get("coord", {}).get("enabled"):
        v_scale = float(manifest["coord"].get("v_scale_kms", 100.0))
        coord = _make_v_axis(cfg).astype(np.float32) / v_scale
    model = CenterHeatmapNet1DDeep(
        width=int(args.get("width", 96)),
        n_blocks=int(args.get("n_blocks", 6)),
        coord=coord,
        kernel_size=kernel_size,
    )
    state = torch.load(run_dir / "center_heatmap_net.pt", map_location="cpu")
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def _load_heatmap_count(run_dir: Path, manifest: dict, *, device: str) -> HeatmapCountNet:
    stage1_dir = Path(manifest["stage1_run_dir"])
    if not stage1_dir.is_absolute():
        stage1_dir = (_REPO / stage1_dir).resolve()
    stage1_manifest = json.loads((stage1_dir / "manifest.json").read_text(encoding="utf-8"))
    heatmap = _load_heatmap(stage1_dir, stage1_manifest, device="cpu")
    args = dict(manifest.get("args", {}))
    kernel_size = int(manifest.get("kernel_size", args.get("kernel_size", 25)))
    model = HeatmapCountNet(
        heatmap,
        width=int(args.get("width", 96)),
        n_blocks=int(args.get("n_blocks", 6)),
        k_input=str(manifest.get("k_input", args.get("k_input", "spec_p"))),
        freeze_heatmap=bool(manifest.get("freeze_heatmap", True)),
        kernel_size=kernel_size,
    )
    state = torch.load(run_dir / "heatmap_count_net.pt", map_location="cpu")
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def _decode_cfg(manifest: dict) -> dict:
    dec = dict(manifest.get("decode_stage1") or manifest.get("decode") or {})
    args = dict(manifest.get("args", {}))
    return {
        "height": float(dec.get("height", 0.35)),
        "prominence": float(dec.get("prominence", 0.15)),
        "min_sep_kms": float(dec.get("min_sep_kms", 4.0)),
        "Kmax": int(dec.get("Kmax", args.get("Kmax", 6))),
    }


def _prep_aces_batch(
    spec_raw: np.ndarray, *, min_finite: int, min_peak: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ACES NaN/zero pad contract via prepare_spectrum_input."""
    B, C = spec_raw.shape
    x_norm = np.zeros((B, C), dtype=np.float32)
    chan_mask = np.zeros((B, C), dtype=np.float32)
    for i in range(B):
        xn, vm = prepare_spectrum_input(spec_raw[i])
        x_norm[i] = xn
        chan_mask[i] = vm
    n_valid = chan_mask.sum(axis=1)
    valid_bool = chan_mask > 0.5
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        peak = np.nanmax(np.where(valid_bool, spec_raw, np.nan), axis=1)
    peak = np.where(np.isfinite(peak), peak, 0.0)
    ok = (n_valid >= min_finite) & (peak > min_peak)
    return x_norm, chan_mask, ok


@torch.no_grad()
def _kv_batch(
    model,
    x_norm: np.ndarray,
    chan_mask: np.ndarray,
    *,
    device: str,
    v_axis: np.ndarray,
    decode: dict,
    k_from_head: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t = torch.from_numpy(x_norm).to(device)
    m = torch.from_numpy(chan_mask).to(device)
    Kmax = int(decode["Kmax"])
    if k_from_head:
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
                prob[i], vel, valid_mask=chan_mask[i],
                height=decode["height"], prominence=decode["prominence"],
                min_sep_kms=decode["min_sep_kms"], Kmax=ki,
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
        prob, v_axis, valid_mask=chan_mask,
        height=decode["height"], prominence=decode["prominence"],
        min_sep_kms=decode["min_sep_kms"], Kmax=Kmax,
    )
    return k.astype(np.float32), v_slots, p_slots


def main() -> None:
    parser = argparse.ArgumentParser(description="ACES heatmap->K map on mosaic cutout.")
    parser.add_argument("--cube", type=Path, default=_DEFAULT_MOSAIC, help="Full ACES mosaic.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--subcube-ref", type=Path, default=_REPO / "data" / "hnco_region1_cube.fits")
    parser.add_argument("--out", type=Path, default=_REPO / "data" / "hnco_region1_aces_hm_k_pred.fits")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--chunk-y", type=int, default=16)
    parser.add_argument("--chunk-x", type=int, default=32)
    parser.add_argument("--infer-batch", type=int, default=128)
    parser.add_argument("--min-finite-channels", type=int, default=100)
    parser.add_argument("--min-peak", type=float, default=0.02)
    parser.add_argument("--y0", type=int, default=None)
    parser.add_argument("--y1", type=int, default=None)
    parser.add_argument("--x0", type=int, default=None)
    parser.add_argument("--x1", type=int, default=None)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    variant = str(manifest.get("variant", ""))
    if variant not in ("center_heatmap", "heatmap_count"):
        raise ValueError(f"Expected center_heatmap or heatmap_count, got {variant!r}")
    k_from_head = variant == "heatmap_count"
    decode = _decode_cfg(manifest)
    vw = dict(manifest.get("vel_window") or {})
    vrange = tuple(vw.get("vrange") or manifest["cfg"]["vrange"])
    n_model = int(manifest["cfg"]["n_channels"])
    v_model = _make_v_axis(manifest["cfg"]).astype(np.float64)

    print(f"Loading mosaic (lazy): {args.cube}", flush=True)
    t0 = time.perf_counter()
    cube = SpectralCube.read(str(args.cube.resolve()), use_dask=True)
    nv, ny, nx = cube.shape
    vel_full = cube.spectral_axis.to(u.km / u.s).value.astype(np.float64)
    i0, i1 = _channel_indices_for_vrange(vel_full, (float(vrange[0]), float(vrange[1])))
    n_win = i1 - i0
    print(
        f"Velocity crop channels [{i0},{i1}) -> {n_win} ch  "
        f"(model expects {n_model}; vrange={vrange})",
        flush=True,
    )
    if n_win != n_model:
        ### Snap window length to model by adjusting i1 (keep i0).
        i1 = min(nv, i0 + n_model)
        i0 = max(0, i1 - n_model)
        n_win = i1 - i0
        print(f"Adjusted crop to [{i0},{i1}) for exact n_channels={n_model}", flush=True)
    v_axis = vel_full[i0:i1]
    ### Prefer model axis for decode if lengths match (training grid).
    if v_axis.size == v_model.size:
        v_axis = v_model

    y0, y1, x0, x1 = resolve_spatial_bounds(
        mosaic_ny=ny,
        mosaic_nx=nx,
        y0=args.y0,
        y1=args.y1,
        x0=args.x0,
        x1=args.x1,
        subcube_ref=args.subcube_ref,
        mosaic_path=args.cube,
    )
    out_ny, out_nx = y1 - y0, x1 - x0
    print(f"Spatial cutout y=[{y0},{y1}) x=[{x0},{x1}) -> {out_ny}x{out_nx}", flush=True)
    print(f"Variant={variant}  K from {'head' if k_from_head else 'peak decode'}", flush=True)

    if k_from_head:
        model = _load_heatmap_count(run_dir, manifest, device=args.device)
    else:
        model = _load_heatmap(run_dir, manifest, device=args.device)

    Kmax = int(decode["Kmax"])
    k_map = np.full((out_ny, out_nx), np.nan, dtype=np.float32)
    v_map = np.full((out_ny, out_nx, Kmax), np.nan, dtype=np.float32)
    p_map = np.full((out_ny, out_nx, Kmax), np.nan, dtype=np.float32)
    n_infer = 0
    cy, cx = max(1, args.chunk_y), max(1, args.chunk_x)
    n_chunks = ((out_ny + cy - 1) // cy) * ((out_nx + cx - 1) // cx)
    done = 0
    for ys in range(y0, y1, cy):
        ye = min(ys + cy, y1)
        for xs in range(x0, x1, cx):
            xe = min(xs + cx, x1)
            sub = cube[i0:i1, ys:ye, xs:xe]
            arr = sub.filled(np.nan)
            if hasattr(arr, "compute"):
                arr = arr.compute()
            arr = np.asarray(arr, dtype=np.float32)
            ### arr: (C, ny, nx)
            c_loc, ny_c, nx_c = arr.shape
            spec = np.transpose(arr, (1, 2, 0)).reshape(ny_c * nx_c, c_loc)
            k_flat = np.full(ny_c * nx_c, np.nan, dtype=np.float32)
            v_flat = np.full((ny_c * nx_c, Kmax), np.nan, dtype=np.float32)
            p_flat = np.full((ny_c * nx_c, Kmax), np.nan, dtype=np.float32)
            for b0 in range(0, spec.shape[0], args.infer_batch):
                b1 = min(b0 + args.infer_batch, spec.shape[0])
                x_norm, chan_mask, ok = _prep_aces_batch(
                    spec[b0:b1],
                    min_finite=args.min_finite_channels,
                    min_peak=args.min_peak,
                )
                if not np.any(ok):
                    continue
                k, vs, ps = _kv_batch(
                    model, x_norm[ok], chan_mask[ok],
                    device=args.device, v_axis=v_axis, decode=decode, k_from_head=k_from_head,
                )
                block = k_flat[b0:b1]
                block[ok] = k
                k_flat[b0:b1] = block
                vb = v_flat[b0:b1]
                pb = p_flat[b0:b1]
                vb[ok] = vs
                pb[ok] = ps
                v_flat[b0:b1] = vb
                p_flat[b0:b1] = pb
                n_infer += int(ok.sum())
            ly0, lx0 = ys - y0, xs - x0
            k_map[ly0 : ly0 + ny_c, lx0 : lx0 + nx_c] = k_flat.reshape(ny_c, nx_c)
            v_map[ly0 : ly0 + ny_c, lx0 : lx0 + nx_c, :] = v_flat.reshape(ny_c, nx_c, Kmax)
            p_map[ly0 : ly0 + ny_c, lx0 : lx0 + nx_c, :] = p_flat.reshape(ny_c, nx_c, Kmax)
            done += 1
            if done == 1 or done % 10 == 0 or done == n_chunks:
                print(f"  chunk {done}/{n_chunks}", flush=True)

    wcs_full = wcs_celestial(cube.header)
    wcs_header = wcs_header_for_array_cutout(wcs_full, y0=y0, y1=y1, x0=x0, x1=x1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    hdu = fits.PrimaryHDU(data=k_map, header=wcs_header)
    hdu.header["BUNIT"] = "1"
    hdu.header["COMMENT"] = (
        "ACES HeatmapCountNet K_pred" if k_from_head else "ACES heatmap peak-decode K_pred"
    )
    hdu.header["KMAX"] = Kmax
    hdu.header["VLO"] = float(vrange[0])
    hdu.header["VHI"] = float(vrange[1])
    hdu.header["MOPRARUN"] = run_dir.name[:68]
    hdu.writeto(str(args.out.resolve()), overwrite=True)

    ys_i, xs_i = np.where(np.isfinite(k_map))
    centers_out = args.out.with_name(f"{args.out.stem}_centers.npz")
    np.savez_compressed(
        centers_out,
        yi=ys_i.astype(np.int64),
        xi=xs_i.astype(np.int64),
        k_pred=k_map[ys_i, xs_i].astype(np.float32),
        center_v_kms=v_map[ys_i, xs_i, :],
        center_prob=p_map[ys_i, xs_i, :],
        y0=np.int32(y0),
        x0=np.int32(x0),
        i0=np.int32(i0),
        i1=np.int32(i1),
        v_axis=v_axis.astype(np.float32),
        run_dir=np.asarray(str(run_dir)),
    )
    print(f"Wrote {centers_out} ({ys_i.size} pixels)", flush=True)
    print(f"Wrote {args.out} ({(time.perf_counter() - t0) / 60:.1f} min)", flush=True)
    k_fin = k_map[np.isfinite(k_map)]
    if k_fin.size:
        print(
            f"K stats: min={k_fin.min():.0f} med={np.median(k_fin):.0f} "
            f"max={k_fin.max():.0f} mean={k_fin.mean():.2f} n={k_fin.size}",
            flush=True,
        )

    if not args.no_plot:
        import matplotlib.pyplot as plt

        fig_out = _ACES / "figures" / f"{args.out.stem}.png"
        fig_out.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(7, 5))
        im = ax.imshow(k_map, origin="lower", cmap="viridis", vmin=0, vmax=max(4, Kmax))
        fig.colorbar(im, ax=ax, label="K_pred")
        ax.set_title(f"ACES heatmap->K ({run_dir.name})")
        ax.set_xlabel("x (cutout pix)")
        ax.set_ylabel("y (cutout pix)")
        fig.tight_layout()
        fig.savefig(fig_out, dpi=140)
        plt.close(fig)
        print(f"Wrote {fig_out}", flush=True)


if __name__ == "__main__":
    main()
