#!/usr/bin/env python
"""
Apply a trained MOPRA Count model to SCouse SAA-averaged parent spectra.

Run from repo root (set thread caps on macOS to avoid exit-time segfaults):
  export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
  python experiments/MOPRA_Count/run_saa_k_predict.py \\
    --saa-dir experiments/MOPRA_Count/runs/saa_grid_cmz \\
    --run-dir experiments/MOPRA_Count/runs/mopra_count_<ts>_<tag>
"""
from __future__ import annotations

import os

### Cap BLAS/OpenMP before numpy/torch import (macOS teardown segfaults otherwise).
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import gc
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch

_SCRIPT = Path(__file__).resolve()
_MOPRA = _SCRIPT.parent
_REPO = _MOPRA.parents[1]
_DEFAULT_SAA_DIR = _MOPRA / "runs" / "saa_grid_cmz"
sys.path.insert(0, str(_REPO / "src"))

from spectackle.data.mopra_preprocess import prepare_mopra_input  ### noqa: E402
from spectackle.models import CountNet1DDeep  ### noqa: E402


def _load_model(run_dir: Path, manifest: dict, *, device: str) -> CountNet1DDeep:
    args = manifest["args"]
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
    blank_value: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_norm, chan_mask = prepare_mopra_input(spec_raw, blank_value=blank_value)
    n_valid = chan_mask.sum(axis=1)
    valid_bool = chan_mask > 0.5
    spec_for_peak = np.where(valid_bool, spec_raw, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        peak = np.nanmax(np.abs(spec_for_peak), axis=1)
    peak = np.where(np.isfinite(peak), peak, 0.0)
    ok = (n_valid >= min_finite) & (peak > min_peak)
    return x_norm, chan_mask, ok


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


def main() -> None:
    parser = argparse.ArgumentParser(description="K_pred for SCouse SAA parent spectra.")
    parser.add_argument("--saa-dir", type=Path, default=_DEFAULT_SAA_DIR)
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Train folder with count_net.pt + manifest.json.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output JSON (default: <saa-dir>/saa_k_pred.json).",
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--infer-batch", type=int, default=256)
    parser.add_argument("--min-finite-channels", type=int, default=100)
    parser.add_argument("--min-peak", type=float, default=0.02)
    parser.add_argument("--blank-value", type=float, default=-1.0)
    parser.add_argument("--continuous", action="store_true")
    parser.add_argument(
        "--cube",
        type=Path,
        default=_REPO / "data" / "CMZ_3mm_HNCO.fits",
        help="Cube FITS for WCS when writing K map at SAA centres.",
    )
    parser.add_argument(
        "--k-map-out",
        type=Path,
        default=None,
        help="Optional 2D FITS: K_pred at SAA centres (NaN elsewhere).",
    )
    args = parser.parse_args()

    saa_dir = args.saa_dir.resolve()
    catalog_path = saa_dir / "saa_catalog.json"
    spectra_path = saa_dir / "saa_spectra.npz"
    if not catalog_path.is_file() or not spectra_path.is_file():
        raise FileNotFoundError(f"Missing saa_catalog.json or saa_spectra.npz in {saa_dir}")

    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    with np.load(spectra_path) as npz:
        spectra = npz["spectra"].astype(np.float64)
        saa_id = npz["saa_id"]
        center_x = npz["center_x"]
        center_y = npz["center_y"]
        n_pixels = npz["n_pixels"]
    n_saa = spectra.shape[0]
    n_ch = spectra.shape[1]

    run_dir = args.run_dir.resolve()
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    n_channels = int(manifest["cfg"]["n_channels"])
    Kmax = int(manifest["args"].get("Kmax", manifest["cfg"].get("max_components", 10)))
    if n_ch != n_channels:
        raise ValueError(f"SAA spectra have {n_ch} channels but model expects {n_channels}")

    print(f"SAA grid: {saa_dir}  ({n_saa} parent spectra)", flush=True)
    print(f"Model: {run_dir.name}  Kmax={Kmax}", flush=True)
    model = _load_model(run_dir, manifest, device=args.device)

    k_pred = np.full(n_saa, np.nan, dtype=np.float32)
    k_hat_cont = np.full(n_saa, np.nan, dtype=np.float32)
    inferred = np.zeros(n_saa, dtype=bool)

    t0 = time.perf_counter()
    n_ok = 0
    bs = max(1, args.infer_batch)
    for i0 in range(0, n_saa, bs):
        i1 = min(i0 + bs, n_saa)
        x_norm, chan_mask, ok = _prep_batch(
            spectra[i0:i1],
            min_finite=args.min_finite_channels,
            min_peak=args.min_peak,
            blank_value=args.blank_value,
        )
        if not np.any(ok):
            continue
        k = _k_batch(
            model,
            x_norm[ok],
            chan_mask[ok],
            device=args.device,
            Kmax=Kmax,
            round_output=not args.continuous,
        )
        idx_local = np.where(ok)[0]
        rows = np.arange(i0, i1)[idx_local]
        k_hat_cont[rows] = k
        if args.continuous:
            k_pred[rows] = k
        else:
            k_pred[rows] = np.round(k).astype(np.float32)
        inferred[rows] = True
        n_ok += int(ok.sum())

    elapsed = time.perf_counter() - t0
    print(f"Inferred {n_ok}/{n_saa} SAAs in {elapsed:.1f}s  ({n_ok / max(elapsed, 1e-6):.0f} spec/s)", flush=True)

    records = []
    for i in range(n_saa):
        records.append(
            {
                "saa_id": int(saa_id[i]),
                "center_x": int(center_x[i]),
                "center_y": int(center_y[i]),
                "n_pixels": int(n_pixels[i]),
                "inferred": bool(inferred[i]),
                "K_pred": None if not inferred[i] else float(k_pred[i]),
                "K_hat": None if not inferred[i] else float(k_hat_cont[i]),
            }
        )

    k_fin = k_pred[inferred]
    if k_fin.size:
        print(
            f"K stats: min={k_fin.min():.0f} med={np.median(k_fin):.0f} "
            f"max={k_fin.max():.0f} mean={k_fin.mean():.2f}",
            flush=True,
        )

    out_json = args.out if args.out is not None else saa_dir / "saa_k_pred.json"
    out_json = Path(out_json)
    report = {
        "saa_dir": str(saa_dir),
        "run_dir": str(run_dir),
        "n_saa": n_saa,
        "n_inferred": int(n_ok),
        "Kmax": Kmax,
        "continuous": bool(args.continuous),
        "records": records,
    }
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {out_json}", flush=True)

    if args.k_map_out is not None:
        from astropy.io import fits
        from astropy.wcs import WCS

        header = fits.getheader(args.cube)
        wcs = WCS(header).celestial
        ny = int(header["NAXIS2"])
        nx = int(header["NAXIS1"])
        k_map = np.full((ny, nx), np.nan, dtype=np.float32)
        for i in range(n_saa):
            if not inferred[i]:
                continue
            y, x = int(center_y[i]), int(center_x[i])
            if 0 <= y < ny and 0 <= x < nx:
                k_map[y, x] = k_pred[i]
        k_map_path = Path(args.k_map_out)
        k_map_path.parent.mkdir(parents=True, exist_ok=True)
        hdu = fits.PrimaryHDU(data=k_map, header=wcs.to_header())
        hdu.header["BUNIT"] = "1"
        hdu.header["COMMENT"] = "K_pred at SAA centres only; NaN elsewhere"
        hdu.header["KMAX"] = Kmax
        hdu.header["MOPRARUN"] = run_dir.name[:68]
        hdu.header["SAAGRID"] = saa_dir.name[:68]
        hdu.writeto(str(k_map_path.resolve()), overwrite=True)
        print(f"Wrote SAA-centre map {k_map_path}", flush=True)

    del model
    gc.collect()
if __name__ == "__main__":
    main()
