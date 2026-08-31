#!/usr/bin/env python
"""
Two-level SAA inference on a real cube: stage1 K on parent SAAs, stage2 K on pixels.

No human input for stage 1 - SAA grid comes from scousepy-style coverage on the cube.

Example:
  python experiments/MOPRA_Count/SAA_TwoLevel/setup_saa_grid_smooth60.py

  python experiments/MOPRA_Count/SAA_TwoLevel/run_cube_twolevel.py \\
    --cube data/CMZ_3mm_HNCO_60.fits \\
    --saa-dir experiments/MOPRA_Count/SAA_TwoLevel/runs/saa_grid_smooth60 \\
    --stage1-run-dir experiments/MOPRA_Count/SAA_TwoLevel/runs/saa2_stage1_<ts> \\
    --stage2-run-dir experiments/MOPRA_Count/SAA_TwoLevel/runs/saa2_stage2_<ts> \\
    --out data/mopra_cmz_k_pred_saa2.fits \\
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
_EXP = _SCRIPT.parent
_MOPRA = _EXP.parent
_REPO = _MOPRA.parents[1]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_MOPRA / "plots"))

from plot_crop_utils import crop_bbox_from_mask, labeled_mask_from_dat  ### noqa: E402
from plot_k_map_image import plot_k_map_figure  ### noqa: E402
from plot_k_residual_map import compare_k_pred_to_scouse  ### noqa: E402

from spectackle.data.mopra_preprocess import prepare_mopra_input  ### noqa: E402
from spectackle.data.mopra_resample import resample_spec_batch, velocity_axis_kms  ### noqa: E402
from spectackle.models import CountNet1DDeep, CountNet1DDeepSaaCond  ### noqa: E402
from spectackle.wcs_plot import wcs_celestial, wcs_header_for_array_cutout  ### noqa: E402


def _load_stage1(run_dir: Path, *, device: str) -> tuple[CountNet1DDeep, dict]:
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    args = manifest["args"]
    model = CountNet1DDeep(width=int(args["width"]), n_blocks=int(args["n_blocks"]))
    model.load_state_dict(torch.load(run_dir / "count_net.pt", map_location="cpu"))
    model.to(device).eval()
    return model, manifest


def _load_stage2(run_dir: Path, *, device: str) -> tuple[CountNet1DDeepSaaCond, dict]:
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    args = manifest["args"]
    model = CountNet1DDeepSaaCond(
        width=int(args["width"]), n_blocks=int(args["n_blocks"]), Kmax=int(args["Kmax"]),
    )
    model.load_state_dict(torch.load(run_dir / "count_net_saa_cond.pt", map_location="cpu"))
    model.to(device).eval()
    return model, manifest


@torch.no_grad()
def _k_stage1_batch(model, x_norm, mask, *, device: str, Kmax: int) -> np.ndarray:
    t = torch.from_numpy(x_norm).to(device)
    m = torch.from_numpy(mask).to(device)
    k_hat = model(t, m)
    return torch.clamp(torch.round(k_hat), 0, Kmax).cpu().numpy().astype(np.int64)


@torch.no_grad()
def _k_stage2_batch(
    model, x_norm, parent_norm, k_parent, mask, *, device: str, Kmax: int,
) -> np.ndarray:
    t = torch.from_numpy(x_norm).to(device)
    p = torch.from_numpy(parent_norm).to(device)
    kp = torch.from_numpy(k_parent.astype(np.int64)).to(device)
    m = torch.from_numpy(mask).to(device)
    k_hat = model(t, p, kp, m)
    return torch.clamp(torch.round(k_hat), 0, Kmax).cpu().numpy().astype(np.int64)


def main() -> None:
    parser = argparse.ArgumentParser(description="Two-level SAA cube inference.")
    parser.add_argument("--cube", type=Path, default=_REPO / "data" / "CMZ_3mm_HNCO_60.fits")
    parser.add_argument("--saa-dir", type=Path, required=True)
    parser.add_argument("--stage1-run-dir", type=Path, required=True)
    parser.add_argument("--stage2-run-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=_REPO / "data" / "mopra_cmz_k_pred_saa2.fits")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--infer-batch", type=int, default=256)
    parser.add_argument("--blank-value", type=float, default=-1.0)
    parser.add_argument("--dat", type=Path, default=_REPO / "data" / "final_fits_updated.dat")
    parser.add_argument("--compare-scouse", action="store_true")
    parser.add_argument("--infer-on-scouse-labels", action="store_true")
    parser.add_argument("--min-finite-channels", type=int, default=100)
    args = parser.parse_args()

    catalog = json.loads((args.saa_dir / "saa_catalog.json").read_text(encoding="utf-8"))
    saa_npz = np.load(args.saa_dir / "saa_spectra.npz")
    parent_specs = saa_npz["spectra"].astype(np.float64)
    n_saa = parent_specs.shape[0]

    stage1_model, m1 = _load_stage1(args.stage1_run_dir, device=args.device)
    stage2_model, m2 = _load_stage2(args.stage2_run_dir, device=args.device)
    Kmax = int(m2["args"]["Kmax"])
    n_channels = int(m1["cfg"]["n_channels"])

    print(f"Loading cube: {args.cube}", flush=True)
    cube = SpectralCube.read(str(args.cube.resolve()), use_dask=False)
    arr = np.asarray(cube.filled(np.nan), dtype=np.float32)
    if arr.ndim == 4:
        arr = arr[0]
    nv, ny, nx = arr.shape
    v_src = cube.spectral_axis.to("km/s").value.astype(np.float64)
    v_tgt = velocity_axis_kms(_REPO / "data" / "CMZ_3mm_HNCO_60.fits")
    if v_src.size != n_channels:
        resample = True
    else:
        resample = nv != n_channels
    wcs_full = wcs_celestial(cube.header)

    ### Stage 1: K_parent per SAA from precomputed parent spectra.
    print(f"Stage 1: {n_saa} parent SAAs", flush=True)
    k_parent_saa = np.zeros(n_saa, dtype=np.int64)
    t0 = time.perf_counter()
    for i0 in range(0, n_saa, args.infer_batch):
        i1 = min(i0 + args.infer_batch, n_saa)
        block = parent_specs[i0:i1]
        if resample:
            block = resample_spec_batch(block, v_src, v_tgt, blank_value=args.blank_value)
        x_norm, mask = [], []
        for row in block:
            sn, vm = prepare_mopra_input(row, blank_value=args.blank_value)
            x_norm.append(sn)
            mask.append(vm)
        x_norm = np.stack(x_norm, axis=0)
        mask = np.stack(mask, axis=0)
        ok = mask.sum(axis=1) >= args.min_finite_channels
        kp = np.zeros(i1 - i0, dtype=np.int64)
        if np.any(ok):
            kp[ok] = _k_stage1_batch(stage1_model, x_norm[ok], mask[ok], device=args.device, Kmax=Kmax)
        k_parent_saa[i0:i1] = kp
    print(f"  stage1 done in {time.perf_counter()-t0:.1f}s  K_parent median={np.median(k_parent_saa):.0f}", flush=True)

    k_map = np.full((ny, nx), np.nan, dtype=np.float32)
    scouse_mask = None
    if args.infer_on_scouse_labels:
        scouse_mask = labeled_mask_from_dat(args.dat, shape=(ny, nx), wcs=wcs_full)

    ### Stage 2: pixels within each SAA footprint.
    print("Stage 2: pixel inference within SAAs", flush=True)
    n_infer = 0
    t1 = time.perf_counter()
    for si, saa in enumerate(catalog["saa"]):
        saa_id = int(saa["saa_id"])
        parent_raw = parent_specs[saa_id]
        if resample:
            parent_raw = resample_spec_batch(
                parent_raw.reshape(1, -1), v_src, v_tgt, blank_value=args.blank_value,
            )[0]
        parent_norm, _ = prepare_mopra_input(parent_raw, blank_value=args.blank_value)
        k_par = int(k_parent_saa[saa_id])
        ys = np.asarray(saa["pixel_y"], dtype=np.int32)
        xs = np.asarray(saa["pixel_x"], dtype=np.int32)
        for j0 in range(0, ys.size, args.infer_batch):
            j1 = min(j0 + args.infer_batch, ys.size)
            yb, xb = ys[j0:j1], xs[j0:j1]
            if scouse_mask is not None:
                keep = scouse_mask[yb, xb]
                yb, xb = yb[keep], xb[keep]
                if yb.size == 0:
                    continue
            specs = arr[:, yb, xb].T.astype(np.float64)
            if resample:
                specs = resample_spec_batch(specs, v_src, v_tgt, blank_value=args.blank_value)
            x_norm, masks = [], []
            for row in specs:
                sn, vm = prepare_mopra_input(row, blank_value=args.blank_value)
                x_norm.append(sn)
                masks.append(vm)
            x_norm = np.stack(x_norm, axis=0)
            masks = np.stack(masks, axis=0)
            parent_batch = np.tile(parent_norm.reshape(1, -1), (x_norm.shape[0], 1))
            kp_batch = np.full(x_norm.shape[0], k_par, dtype=np.int64)
            ok = masks.sum(axis=1) >= args.min_finite_channels
            if not np.any(ok):
                continue
            k_pred = _k_stage2_batch(
                stage2_model,
                x_norm[ok],
                parent_batch[ok],
                kp_batch[ok],
                masks[ok],
                device=args.device,
                Kmax=Kmax,
            )
            k_map[yb[ok], xb[ok]] = k_pred.astype(np.float32)
            n_infer += int(ok.sum())
        if (si + 1) % 50 == 0 or si == 0:
            print(f"  SAA {si+1}/{n_saa}  pixels inferred so far: {n_infer}", flush=True)
    print(f"Stage 2 done in {time.perf_counter()-t1:.1f}s  total pixels: {n_infer}", flush=True)

    wcs_header = wcs_header_for_array_cutout(wcs_full, y0=0, y1=ny, x0=0, x1=nx)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    hdu = fits.PrimaryHDU(data=k_map, header=wcs_header)
    hdu.header["COMMENT"] = "Two-level SAA K_pred; NaN outside SAA footprints"
    hdu.header["SAA2STG1"] = args.stage1_run_dir.name[:56]
    hdu.header["SAA2STG2"] = args.stage2_run_dir.name[:56]
    hdu.writeto(str(args.out.resolve()), overwrite=True)
    print(f"Wrote {args.out}", flush=True)

    fig_out = _MOPRA / "figures" / f"{args.out.stem}.png"
    crop = None
    if args.dat.is_file():
        labeled = labeled_mask_from_dat(args.dat, shape=(ny, nx), wcs=wcs_full)
        crop = crop_bbox_from_mask(labeled, pad=0)
    plot_k_map_figure(k_map, wcs_header, out=fig_out, crop=crop, title="K_pred (SAA two-level)")
    print(f"Wrote {fig_out}", flush=True)

    if args.compare_scouse:
        resid = _MOPRA / "figures" / f"{args.out.stem}_residual.png"
        stats = compare_k_pred_to_scouse(
            k_map, dat_path=args.dat, cube_path=args.cube, out_png=resid,
        )
        print(
            f"Scouse compare: n={stats['n_compare']} MAE={stats['mae']:.3f} "
            f"median dK={stats['median_delta']:.1f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
