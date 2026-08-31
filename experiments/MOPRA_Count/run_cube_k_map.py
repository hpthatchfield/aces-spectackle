#!/usr/bin/env python
"""
Apply a trained MOPRA Count model to a real PPV cube -> 2D K_pred map (l-b plane).

One-shot example (smoothed cube, Scouse-label mask, cropped PNG, Scouse residual):
  python experiments/MOPRA_Count/run_cube_k_map.py \\
    --cube data/CMZ_3mm_HNCO_60.fits \\
    --run-dir experiments/MOPRA_Count/runs/mopra_count_<timestamp>_scouse_smooth60 \\
    --infer-on-scouse-labels \\
    --out data/mopra_cmz_k_pred_scouse_smooth60_labels.fits \\
    --compare-scouse

Outputs:
  data/mopra_cmz_k_pred_smooth60_snr5.fits          K map
  experiments/MOPRA_Count/figures/<stem>.png        cropped K_pred (default)
  experiments/MOPRA_Count/figures/<stem>_residual.png  delta-K vs Scouse (if --compare-scouse)
"""
from __future__ import annotations

import os

### Cap BLAS/OpenMP before numpy/torch import (macOS teardown segfaults otherwise).
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

from spectackle.data.mopra_preprocess import (  ### noqa: E402
    NORM_MODES,
    prepare_mopra_input,
    snr_peak_rms_mopra,
    snr_peak_scouse_mopra,
)
from spectackle.data.mopra_resample import resample_spec_batch, velocity_axis_kms  ### noqa: E402
from spectackle.models import CountNet1DDeep  ### noqa: E402
from spectackle.wcs_plot import wcs_celestial, wcs_header_for_array_cutout  ### noqa: E402


def _load_model(run_dir: Path, manifest: dict, *, device: str) -> CountNet1DDeep:
    args = dict(manifest.get("args", {}))
    if "width" not in args or "n_blocks" not in args:
        init_dir = manifest.get("init_run_dir")
        if init_dir:
            init_manifest = json.loads((Path(init_dir) / "manifest.json").read_text(encoding="utf-8"))
            for key in ("width", "n_blocks", "Kmax"):
                if key not in args and key in init_manifest.get("args", {}):
                    args[key] = init_manifest["args"][key]
    model = CountNet1DDeep(width=int(args["width"]), n_blocks=int(args["n_blocks"]))
    state = torch.load(run_dir / "count_net.pt", map_location="cpu")
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


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
    """
    spec_raw: (B, C). Returns (spec_norm, valid_mask, pixel_ok) for model input.
    """
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
        raise ValueError(f"Unknown snr_method {snr_method!r}; use 'scouse' or 'global'.")
    pixel_ok = (n_valid >= min_finite) & (peak > min_peak)
    if min_snr > 0.0:
        pixel_ok &= snr >= min_snr
    return x_norm, chan_mask, pixel_ok


@torch.no_grad()
def _k_batch(
    model,
    x_norm: np.ndarray,
    chan_mask: np.ndarray,
    *,
    device: str,
    Kmax: int,
    round_output: bool,
) -> np.ndarray:
    t = torch.from_numpy(x_norm).to(device)
    m = torch.from_numpy(chan_mask).to(device)
    k_hat = model(t, m)
    if round_output:
        k_out = torch.clamp(torch.round(k_hat), 0, Kmax)
    else:
        k_out = torch.clamp(k_hat, 0.0, float(Kmax))
    return k_out.cpu().numpy().astype(np.float32)


def _process_spatial_chunk(
    arr_vyx: np.ndarray,
    model,
    *,
    device: str,
    Kmax: int,
    min_finite: int,
    min_peak: float,
    min_snr: float,
    blank_value: float,
    infer_batch: int,
    round_output: bool,
    v_src: np.ndarray | None = None,
    v_tgt: np.ndarray | None = None,
    snr_vel_range: tuple[float, float] | None = None,
    snr_method: str = "scouse",
    norm_mode: str = "zscore",
    infer_mask_yx: np.ndarray | None = None,
) -> tuple[np.ndarray, int, int]:
    """arr_vyx: (n_vel, ny, nx) -> K map (ny, nx), NaN where skipped."""
    nv, ny, nx = arr_vyx.shape
    spec = np.transpose(arr_vyx, (1, 2, 0)).reshape(ny * nx, nv).astype(np.float64, copy=False)
    k_flat = np.full(ny * nx, np.nan, dtype=np.float32)

    n_infer = 0
    for i0 in range(0, spec.shape[0], infer_batch):
        i1 = min(i0 + infer_batch, spec.shape[0])
        block = spec[i0:i1]
        if v_tgt is not None:
            block = resample_spec_batch(block, v_src, v_tgt, blank_value=blank_value)
        vel_for_snr = v_tgt if v_tgt is not None else v_src
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
        k = _k_batch(
            model,
            x_norm[ok],
            chan_mask[ok],
            device=device,
            Kmax=Kmax,
            round_output=round_output,
        )
        block = k_flat[i0:i1]
        block[ok] = k
        k_flat[i0:i1] = block
        n_infer += int(ok.sum())

    n_skip = ny * nx - n_infer
    return k_flat.reshape(ny, nx), n_infer, n_skip


def _estimate_runtime(
    cube: SpectralCube,
    model,
    *,
    device: str,
    Kmax: int,
    y0: int,
    y1: int,
    x0: int,
    x1: int,
    min_finite: int,
    min_peak: float,
    min_snr: float,
    blank_value: float,
    infer_batch: int,
    round_output: bool,
    sample_pixels: int,
    v_src: np.ndarray | None = None,
    v_tgt: np.ndarray | None = None,
    snr_vel_range: tuple[float, float] | None = None,
    snr_method: str = "scouse",
    norm_mode: str = "zscore",
    infer_mask_yx: np.ndarray | None = None,
    rng_seed: int = 0,
) -> None:
    ### Time a random subset of spectra, extrapolate to the output region.
    ny, nx = y1 - y0, x1 - x0
    n_region = ny * nx
    rng = np.random.default_rng(rng_seed)
    if infer_mask_yx is not None:
        labeled_flat = np.flatnonzero(infer_mask_yx.ravel())
        if labeled_flat.size == 0:
            print("=== Runtime estimate ===", flush=True)
            print("No Scouse-labeled pixels in output region.", flush=True)
            return
        n_sample = min(sample_pixels, labeled_flat.size)
        flat_idx = rng.choice(labeled_flat, size=n_sample, replace=False)
        sy = flat_idx // nx + y0
        sx = flat_idx % nx + x0
    else:
        n_sample = min(sample_pixels, n_region)
        flat_idx = rng.choice(n_region, size=n_sample, replace=False)
        sy = flat_idx // nx + y0
        sx = flat_idx % nx + x0

    t0 = time.perf_counter()
    n_ok = 0
    for i in range(0, n_sample, infer_batch):
        batch_y = sy[i : i + infer_batch]
        batch_x = sx[i : i + infer_batch]
        specs = []
        for y, x in zip(batch_y, batch_x):
            sl = cube[:, int(y), int(x)]
            arr = sl.filled(np.nan)
            if hasattr(arr, "compute"):
                arr = arr.compute()
            specs.append(np.asarray(arr, dtype=np.float64))
        spec_b = np.stack(specs, axis=0)
        if v_tgt is not None:
            spec_b = resample_spec_batch(spec_b, v_src, v_tgt, blank_value=blank_value)
        vel_for_snr = v_tgt if v_tgt is not None else v_src
        x_norm, chan_mask, ok = _prep_batch(
            spec_b,
            min_finite=min_finite,
            min_peak=min_peak,
            min_snr=min_snr,
            blank_value=blank_value,
            vel_kms=vel_for_snr,
            snr_vel_range=snr_vel_range,
            snr_method=snr_method,
            norm_mode=norm_mode,
        )
        if np.any(ok):
            _k_batch(
                model,
                x_norm[ok],
                chan_mask[ok],
                device=device,
                Kmax=Kmax,
                round_output=round_output,
            )
            n_ok += int(ok.sum())
    elapsed = time.perf_counter() - t0
    frac_ok = n_ok / max(1, n_sample)
    sec_per_spec = elapsed / max(1, n_ok)
    n_infer_est = int(round(n_region * frac_ok))
    total_sec = sec_per_spec * n_infer_est

    print("=== Runtime estimate ===", flush=True)
    print(f"Output region: {ny}x{nx} = {n_region:,} sky pixels", flush=True)
    print(f"Sampled {n_sample} spectra ({n_ok} passed mask) in {elapsed:.2f}s", flush=True)
    print(f"Throughput: {sec_per_spec * 1000:.2f} ms/spectrum (inferred only)", flush=True)
    print(f"Estimated inferred pixels: {n_infer_est:,} ({100 * frac_ok:.1f}% of region)", flush=True)
    print(f"Estimated inference time: {total_sec:.0f}s ({total_sec / 60:.1f} min)", flush=True)
    print("(Excludes cube I/O and FITS write; add ~10-30% for chunked reads.)", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="MOPRA Count K map from a real PPV cube.")
    parser.add_argument("--cube", type=Path, default=_DEFAULT_CUBE)
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Train folder with count_net.pt + manifest.json.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=_REPO / "data" / "mopra_cmz_k_pred.fits",
        help="Output 2D FITS (K per sky pixel; NaN = not inferred).",
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--chunk-y", type=int, default=32)
    parser.add_argument("--chunk-x", type=int, default=64)
    parser.add_argument("--infer-batch", type=int, default=512)
    parser.add_argument(
        "--min-finite-channels",
        type=int,
        default=100,
        help="Min valid channels (finite, not BLANK, not 0) to infer.",
    )
    parser.add_argument(
        "--min-peak",
        type=float,
        default=0.02,
        help="Min max|T| over valid channels (K) to infer.",
    )
    parser.add_argument(
        "--min-snr",
        type=float,
        default=3.0,
        help="Min peak SNR to infer (0 disables). Default method: Scouse-style in --snr-vel-range.",
    )
    parser.add_argument(
        "--snr-method",
        choices=("scouse", "global"),
        default="scouse",
        help="SNR definition: scouse=(max-median)/sigma_rms in vel window; global=legacy max|T|/RMS.",
    )
    parser.add_argument(
        "--snr-vel-min",
        type=float,
        default=40.0,
        help="SNR evaluation window lower bound (km/s); used with --snr-method scouse.",
    )
    parser.add_argument(
        "--snr-vel-max",
        type=float,
        default=140.0,
        help="SNR evaluation window upper bound (km/s); used with --snr-method scouse.",
    )
    parser.add_argument(
        "--snr-full-band",
        action="store_true",
        help="Evaluate Scouse SNR over all valid channels (ignore --snr-vel-min/max).",
    )
    parser.add_argument(
        "--blank-value",
        type=float,
        default=-1.0,
        help="FITS BLANK sentinel treated as invalid.",
    )
    parser.add_argument(
        "--continuous",
        action="store_true",
        help="Write raw K_hat (float) instead of rounded integer K.",
    )
    parser.add_argument("--y0", type=int, default=None)
    parser.add_argument("--y1", type=int, default=None)
    parser.add_argument("--x0", type=int, default=None)
    parser.add_argument("--x1", type=int, default=None)
    parser.add_argument(
        "--estimate-only",
        action="store_true",
        help="Print runtime estimate for the output region and exit.",
    )
    parser.add_argument(
        "--estimate-samples",
        type=int,
        default=2048,
        help="Spectra to time for --estimate-only.",
    )
    parser.add_argument(
        "--fig-out",
        type=Path,
        default=None,
        help="Publication PNG for K map (default: same stem as --out, .png).",
    )
    parser.add_argument("--no-plot", action="store_true", help="Skip publication figure.")
    parser.add_argument("--plot-title", type=str, default=None, help="Optional figure title.")
    parser.add_argument("--plot-cmap", type=str, default="Blues")
    parser.add_argument(
        "--ref-cube",
        type=Path,
        default=None,
        help="Reference cube for model velocity axis when --cube has a different channel count "
        "(default: training cube from manifest). Spectra are linearly resampled in velocity.",
    )
    parser.add_argument(
        "--dat",
        type=Path,
        default=_REPO / "data" / "final_fits_updated.dat",
        help="Scouse label table (for crop bbox and --compare-scouse).",
    )
    parser.add_argument(
        "--crop-to-labels",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Crop K_pred PNG to Scouse-labeled region (default: on).",
    )
    parser.add_argument("--crop-pad", type=int, default=0)
    parser.add_argument(
        "--compare-scouse",
        action="store_true",
        help="Also write cropped dK residual PNG/FITS/JSON vs --dat labels.",
    )
    parser.add_argument(
        "--infer-on-scouse-labels",
        action="store_true",
        help="Only infer at (l,b) in --dat (Scouse/Henshaw fits). Disables --min-snr gate; "
        "recommended for Scouse comparison when SNR window drops labeled pixels.",
    )
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
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        hint = ""
        runs_root = _MOPRA / "runs"
        if runs_root.is_dir():
            candidates = sorted(runs_root.glob("mopra_count_*"), key=lambda p: p.stat().st_mtime, reverse=True)
            if candidates:
                hint = "\n  Recent runs:\n" + "\n".join(f"    {p.name}" for p in candidates[:5])
        raise FileNotFoundError(
            f"manifest.json not found under --run-dir: {run_dir}{hint}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    n_channels = int(manifest["cfg"]["n_channels"])
    Kmax = int(manifest["args"].get("Kmax", manifest["cfg"].get("max_components", 10)))
    norm_mode = str(
        args.norm_mode
        or manifest.get("norm_mode")
        or manifest.get("cfg", {}).get("norm_mode")
        or manifest.get("args", {}).get("norm_mode")
        or "zscore"
    )

    print(f"Loading cube (lazy): {args.cube}", flush=True)
    t0 = time.perf_counter()
    cube = SpectralCube.read(str(args.cube.resolve()), use_dask=True)
    nv, ny, nx = cube.shape
    print(f"  shape (v,y,x)=({nv}, {ny}, {nx})  ({time.perf_counter() - t0:.1f}s)", flush=True)

    v_src = cube.spectral_axis.to("km/s").value.astype(np.float64)
    v_tgt: np.ndarray | None = None
    ref_cube_path: Path | None = None
    if nv != n_channels:
        meta = manifest.get("cfg", {}).get("mopra_meta", {})
        ref_default = meta.get("cube_path") or str(args.cube)
        ref_cube_path = (args.ref_cube or Path(ref_default)).resolve()
        v_tgt = velocity_axis_kms(ref_cube_path)
        if v_tgt.size != n_channels:
            raise ValueError(
                f"Reference cube {ref_cube_path} has {v_tgt.size} channels but model expects "
                f"{n_channels} (from {run_dir / 'manifest.json'})."
            )
        overlap = (max(v_src.min(), v_tgt.min()), min(v_src.max(), v_tgt.max()))
        n_ov = int(((v_tgt >= overlap[0]) & (v_tgt <= overlap[1])).sum())
        print(
            f"Resampling {nv} ch -> {n_channels} ch using velocity axis from {ref_cube_path.name}",
            flush=True,
        )
        print(
            f"  src v=[{v_src.min():.1f},{v_src.max():.1f}] km/s  "
            f"tgt v=[{v_tgt.min():.1f},{v_tgt.max():.1f}] km/s  "
            f"overlap channels on tgt: {n_ov}/{n_channels}",
            flush=True,
        )

    y0 = 0 if args.y0 is None else int(args.y0)
    y1 = ny if args.y1 is None else int(args.y1)
    x0 = 0 if args.x0 is None else int(args.x0)
    x1 = nx if args.x1 is None else int(args.x1)
    y0, y1 = max(0, y0), min(ny, y1)
    x0, x1 = max(0, x0), min(nx, x1)
    out_ny, out_nx = y1 - y0, x1 - x0
    if out_ny <= 0 or out_nx <= 0:
        raise ValueError(f"Empty cutout: y=[{y0},{y1}) x=[{x0},{x1})")

    wcs_full = wcs_celestial(cube.header)
    scouse_mask_full: np.ndarray | None = None
    min_snr_infer = float(args.min_snr)
    if args.infer_on_scouse_labels:
        if not args.dat.is_file():
            raise FileNotFoundError(f"--infer-on-scouse-labels requires --dat: {args.dat}")
        scouse_mask_full = labeled_mask_from_dat(args.dat, shape=(ny, nx), wcs=wcs_full)
        n_lab = int(scouse_mask_full[y0:y1, x0:x1].sum())
        min_snr_infer = 0.0
        print(
            f"Infer-on-Scouse-labels: {int(scouse_mask_full.sum())} on cube grid, "
            f"{n_lab} in output region (min_snr gate disabled)",
            flush=True,
        )

    print(f"Output region y=[{y0},{y1}) x=[{x0},{x1}) -> {out_ny}x{out_nx}", flush=True)
    snr_note = ""
    if min_snr_infer > 0.0:
        if args.snr_method == "scouse" and snr_vel_range is not None:
            snr_note = (
                f"  min_snr={min_snr_infer} ({args.snr_method}, "
                f"v=[{snr_vel_range[0]:g},{snr_vel_range[1]:g}] km/s)"
            )
        else:
            snr_note = f"  min_snr={min_snr_infer} ({args.snr_method})"
    elif args.infer_on_scouse_labels:
        snr_note = "  mask=Scouse .dat labels"
    print(
        f"Inference mask: min_finite={args.min_finite_channels}  min_peak={args.min_peak}{snr_note}",
        flush=True,
    )
    print(f"Loading model from {run_dir}  (Kmax={Kmax})", flush=True)
    print(f"Norm mode: {norm_mode}", flush=True)
    model = _load_model(run_dir, manifest, device=args.device)

    if args.estimate_only:
        _estimate_runtime(
            cube,
            model,
            device=args.device,
            Kmax=Kmax,
            y0=y0,
            y1=y1,
            x0=x0,
            x1=x1,
            min_finite=args.min_finite_channels,
            min_peak=args.min_peak,
            min_snr=min_snr_infer,
            blank_value=args.blank_value,
            infer_batch=args.infer_batch,
            round_output=not args.continuous,
            sample_pixels=args.estimate_samples,
            v_src=v_src,
            v_tgt=v_tgt,
            snr_vel_range=snr_vel_range,
            snr_method=args.snr_method,
            norm_mode=norm_mode,
            infer_mask_yx=scouse_mask_full[y0:y1, x0:x1] if scouse_mask_full is not None else None,
        )
        return

    k_map = np.full((out_ny, out_nx), np.nan, dtype=np.float32)
    n_infer = 0
    n_skip = 0
    cy, cx = max(1, args.chunk_y), max(1, args.chunk_x)
    n_chunks = ((out_ny + cy - 1) // cy) * ((out_nx + cx - 1) // cx)
    done = 0
    t_infer = time.perf_counter()

    for ys in range(y0, y1, cy):
        ye = min(ys + cy, y1)
        for xs in range(x0, x1, cx):
            xe = min(xs + cx, x1)
            sub = cube[:, ys:ye, xs:xe]
            arr = sub.filled(np.nan)
            if hasattr(arr, "compute"):
                arr = arr.compute()
            else:
                arr = np.asarray(arr, dtype=np.float32)
            chunk_mask = None
            if scouse_mask_full is not None:
                chunk_mask = scouse_mask_full[ys:ye, xs:xe]
            chunk_k, ni, ns = _process_spatial_chunk(
                arr,
                model,
                device=args.device,
                Kmax=Kmax,
                min_finite=args.min_finite_channels,
                min_peak=args.min_peak,
                min_snr=min_snr_infer,
                blank_value=args.blank_value,
                infer_batch=args.infer_batch,
                round_output=not args.continuous,
                v_src=v_src,
                v_tgt=v_tgt,
                snr_vel_range=snr_vel_range,
                snr_method=args.snr_method,
                norm_mode=norm_mode,
                infer_mask_yx=chunk_mask,
            )
            oy0, oy1 = ys - y0, ye - y0
            ox0, ox1 = xs - x0, xe - x0
            k_map[oy0:oy1, ox0:ox1] = chunk_k
            n_infer += ni
            n_skip += ns
            done += 1
            if done == 1 or done % 10 == 0 or done == n_chunks:
                elapsed = time.perf_counter() - t_infer
                rate = n_infer / max(elapsed, 1e-6)
                print(
                    f"  chunk {done}/{n_chunks}  y={ys}:{ye} x={xs}:{xe}  "
                    f"inferred={ni} skipped={ns}  ({rate:.0f} spec/s cumul)",
                    flush=True,
                )

    wcs_header = wcs_header_for_array_cutout(wcs_full, y0=y0, y1=y1, x0=x0, x1=x1)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    hdu = fits.PrimaryHDU(data=k_map, header=wcs_header)
    hdu.header["BUNIT"] = "1"
    if args.continuous:
        hdu.header["COMMENT"] = "Scheme B K_hat (continuous, clamped 0..Kmax); NaN = not inferred"
    else:
        hdu.header["COMMENT"] = "Scheme B K_pred (rounded); NaN = not inferred"
    hdu.header["KMAX"] = Kmax
    hdu.header["NORMMODE"] = str(norm_mode)
    hdu.header["MOPRARUN"] = run_dir.name[:68]
    if min_snr_infer > 0.0:
        hdu.header["MINSNR"] = float(min_snr_infer)
        hdu.header["SNRMETH"] = args.snr_method[:8]
        if snr_vel_range is not None:
            hdu.header["SNRVLO"] = float(snr_vel_range[0])
            hdu.header["SNRVHI"] = float(snr_vel_range[1])
    if args.infer_on_scouse_labels:
        hdu.header["SCOUSMSK"] = 1
    if v_tgt is not None and ref_cube_path is not None:
        hdu.header["RESAMPLED"] = 1
        hdu.header["REFCUBE"] = ref_cube_path.name[:68]
        hdu.header["SRCHN"] = nv
        hdu.header["TGTCHN"] = n_channels
    hdu.writeto(str(args.out.resolve()), overwrite=True)

    elapsed = time.perf_counter() - t0
    infer_elapsed = time.perf_counter() - t_infer
    print(f"Wrote {args.out}  ({elapsed / 60:.1f} min wall)", flush=True)
    print(
        f"Pixels inferred: {n_infer}  skipped: {n_skip}  "
        f"frac inferred: {n_infer / max(1, n_infer + n_skip):.3f}",
        flush=True,
    )
    print(f"Inference throughput: {n_infer / max(infer_elapsed, 1e-6):.0f} spectra/s", flush=True)
    k_fin = k_map[np.isfinite(k_map)]
    if k_fin.size:
        print(
            f"K stats on inferred pixels: min={k_fin.min():.1f} med={np.median(k_fin):.1f} "
            f"max={k_fin.max():.1f} mean={k_fin.mean():.2f}",
            flush=True,
        )

    if not args.no_plot:
        fig_out = args.fig_out
        if fig_out is None:
            fig_out = _MOPRA / "figures" / f"{args.out.stem}.png"
        title = args.plot_title
        if title is None:
            if args.infer_on_scouse_labels:
                mask_note = ", Scouse labels"
            elif min_snr_infer > 0:
                mask_note = f", SNR>={min_snr_infer:g}"
            else:
                mask_note = ""
            title = f"MOPRA CMZ HNCO: K_pred ({run_dir.name}{mask_note})"

        crop = None
        if args.crop_to_labels and args.dat.is_file():
            wcs_crop = wcs_celestial(hdu.header)
            labeled = labeled_mask_from_dat(args.dat, shape=k_map.shape, wcs=wcs_crop)
            if labeled.any():
                crop = crop_bbox_from_mask(labeled, pad=int(args.crop_pad))

        fig_path = plot_k_map_figure(
            k_map,
            hdu.header,
            out=fig_out,
            Kmax=Kmax,
            title=title,
            cmap_name=args.plot_cmap,
            cbar_label="K_pred" if not args.continuous else r"$\hat{K}$ (continuous)",
            crop=crop,
        )
        print(f"Wrote figure {fig_path}", flush=True)

    if args.compare_scouse:
        if not args.dat.is_file():
            raise FileNotFoundError(f"--compare-scouse requires --dat file: {args.dat}")
        ### Full-grid k_map embedded in output WCS (undo cutout offset if needed).
        k_full = np.full((ny, nx), np.nan, dtype=np.float32)
        k_full[y0:y1, x0:x1] = k_map
        fig_stem = (args.fig_out or (_MOPRA / "figures" / f"{args.out.stem}.png")).stem
        residual_png = _MOPRA / "figures" / f"{fig_stem}_residual.png"
        residual_fits = args.out.with_name(f"{args.out.stem}_residual.fits")
        report = compare_k_pred_to_scouse(
            k_full,
            dat_path=args.dat,
            cube_path=args.cube,
            out_png=residual_png,
            out_fits=residual_fits,
            crop_to_labels=args.crop_to_labels,
            crop_pad=args.crop_pad,
        )
        print(
            f"Scouse compare: n={report['n_compare']}  MAE={report['mae']:.3f}  "
            f"median dK={report['median_delta']:.1f}",
            flush=True,
        )
        print(f"Wrote residual figure {report['residual_png']}", flush=True)


if __name__ == "__main__":
    main()
