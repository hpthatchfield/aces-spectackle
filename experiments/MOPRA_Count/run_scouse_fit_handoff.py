#!/usr/bin/env python
"""
ScousePy handoff: ML K_pred -> multi-Gaussian component table (Henshaw-style .dat).

Fits each target spectrum with K = round(K_pred) Gaussians using scipy
(no pyspeckit/scousepy required). Default targets are Henshaw-labeled (l,b).

Example (quick test, 50 pixels):
  python experiments/MOPRA_Count/run_scouse_fit_handoff.py \\
    --k-pred data/mopra_cmz_k_pred_scouse_ft_v1.fits \\
    --cube data/CMZ_3mm_HNCO_60.fits \\
    --dat data/final_fits_updated.dat \\
    --max-pixels 50 \\
    --out data/mopra_cmz_scouse_ft_v1_handoff.dat

Full Henshaw-pixel run:
  python experiments/MOPRA_Count/run_scouse_fit_handoff.py \\
    --k-pred data/mopra_cmz_k_pred_scouse_ft_v1.fits \\
    --out data/mopra_cmz_scouse_ft_v1_handoff.dat
"""
from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from spectral_cube import SpectralCube

_SCRIPT = Path(__file__).resolve()
_MOPRA = _SCRIPT.parent
_REPO = _MOPRA.parents[1]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_MOPRA / "plots"))

from plot_k_residual_map import k_true_map_from_dat  ### noqa: E402
from spectackle.data.mopra_preprocess import MOPRA_BLANK_VALUE  ### noqa: E402
from spectackle.data.scouse_fit_handoff import (  ### noqa: E402
    fit_result_to_rows,
    fit_spectrum_gaussians,
    write_handoff_dat,
)
from spectackle.wcs_plot import wcs_celestial  ### noqa: E402


def _lb_from_pixel(yi: int, xi: int, wcs: WCS) -> tuple[float, float]:
    l, b = wcs.all_pix2world(xi, yi, 0)
    return float(l), float(b)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit Gaussians at pixels using ML K_pred.")
    parser.add_argument(
        "--k-pred",
        type=Path,
        default=_REPO / "data" / "mopra_cmz_k_pred_scouse_ft_v1.fits",
    )
    parser.add_argument(
        "--cube",
        type=Path,
        default=_REPO / "data" / "CMZ_3mm_HNCO_60.fits",
    )
    parser.add_argument(
        "--dat",
        type=Path,
        default=_REPO / "data" / "final_fits_updated.dat",
        help="Henshaw .dat used to select labeled pixels (ignored if --all-finite-k).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output .dat path (default: data/<k-pred-stem>_handoff.dat).",
    )
    parser.add_argument("--vel-min", type=float, default=40.0)
    parser.add_argument("--vel-max", type=float, default=140.0)
    parser.add_argument("--blank-value", type=float, default=MOPRA_BLANK_VALUE)
    parser.add_argument("--Kmax", type=int, default=6)
    parser.add_argument(
        "--max-pixels",
        type=int,
        default=None,
        help="Optional cap for a quick test (first N labeled pixels in raster order).",
    )
    parser.add_argument(
        "--all-finite-k",
        action="store_true",
        help="Fit every finite K_pred pixel (default: only Henshaw-labeled positions).",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out = args.out
    if out is None:
        out = _REPO / "data" / f"{args.k_pred.stem}_handoff.dat"

    t0 = time.time()
    print(f"Loading K_pred: {args.k_pred}")
    with fits.open(args.k_pred) as hdul:
        k_pred = np.asarray(hdul[0].data, dtype=np.float64)
        k_hdr = hdul[0].header
    ny, nx = k_pred.shape
    wcs = wcs_celestial(k_hdr)

    print(f"Loading cube: {args.cube}")
    cube = SpectralCube.read(str(args.cube), use_dask=False)
    if cube.shape[1:] != (ny, nx):
        raise ValueError(f"Cube spatial shape {cube.shape[1:]} != K_pred {(ny, nx)}")
    vel = cube.spectral_axis.to("km/s").value.astype(np.float64)
    data = cube.unmasked_data[:].value.astype(np.float64)

    if args.all_finite_k:
        mask = np.isfinite(k_pred) & (k_pred >= 0)
    else:
        _, labeled = k_true_map_from_dat(args.dat, shape=(ny, nx), wcs=wcs)
        mask = labeled & np.isfinite(k_pred)
    ys, xs = np.where(mask)
    if args.max_pixels is not None and ys.size > args.max_pixels:
        ### Deterministic subsample for a quick test.
        rng = np.random.default_rng(args.seed)
        pick = rng.choice(ys.size, size=int(args.max_pixels), replace=False)
        pick.sort()
        ys, xs = ys[pick], xs[pick]
    print(f"Target pixels: {ys.size}")

    vel_range = (float(args.vel_min), float(args.vel_max))
    all_rows: list[np.ndarray] = []
    n_ok = 0
    n_fail = 0
    n_k0 = 0
    k_hist: dict[int, int] = {}

    for i, (yi, xi) in enumerate(zip(ys, xs)):
        if i > 0 and i % 200 == 0:
            print(f"  {i}/{ys.size}  ok={n_ok} fail={n_fail} k0={n_k0}")
        k = int(np.clip(np.rint(k_pred[yi, xi]), 0, args.Kmax))
        k_hist[k] = k_hist.get(k, 0) + 1
        if k == 0:
            n_k0 += 1
            continue
        spec = data[:, yi, xi]
        fit = fit_spectrum_gaussians(
            spec,
            vel,
            k,
            blank_value=args.blank_value,
            vel_range=vel_range,
        )
        l, b = _lb_from_pixel(int(yi), int(xi), wcs)
        rows = fit_result_to_rows(fit, l=l, b=b)
        if rows.size == 0:
            n_fail += 1
            continue
        if not fit.success:
            n_fail += 1
        else:
            n_ok += 1
        all_rows.append(rows)

    table = np.vstack(all_rows) if all_rows else np.zeros((0, 15))
    write_handoff_dat(
        out,
        table,
        header_comment=(
            "ncomps l b amp amp_err v v_err sigma sigma_err "
            "rms resid_std chi2 dof redchi aic  "
            "(sigma=Gaussian dispersion km/s; K from ML K_pred)"
        ),
    )

    summary = {
        "k_pred": str(args.k_pred),
        "cube": str(args.cube),
        "dat": str(args.dat),
        "out": str(out),
        "n_target_pixels": int(ys.size),
        "n_component_rows": int(table.shape[0]),
        "n_fit_ok": int(n_ok),
        "n_fit_fail": int(n_fail),
        "n_k0_skipped": int(n_k0),
        "k_hist": {str(k): int(v) for k, v in sorted(k_hist.items())},
        "vel_range": list(vel_range),
        "elapsed_s": float(time.time() - t0),
    }
    summary_path = out.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {out}  ({table.shape[0]} component rows)")
    print(f"Wrote {summary_path}")
    print(
        f"Done in {summary['elapsed_s']:.1f}s  "
        f"ok={n_ok} fail={n_fail} k0={n_k0}  K hist={dict(sorted(k_hist.items()))}"
    )


if __name__ == "__main__":
    main()
