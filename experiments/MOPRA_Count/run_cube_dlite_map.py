#!/usr/bin/env python
"""
Apply a trained MOPRA D-lite model (CenterNet1DDeep) to a real PPV cube -> K_pred map.

Mirrors run_cube_k_map.py but for the two-head D-lite model: K_pred is argmax of the Scheme C
K logits (integer 0..Kmax). Velocity slots are also available and written as a sidecar .npz for
later velocity-vs-Henshaw comparison, but this script's headline output is the K map + Scouse
residual so it is directly comparable to the Scheme B non-fine-tuned baseline (MAE 0.78).

Inference runs only at Scouse-labeled (l,b) positions (like the 0.78 comparison), so the output
is sparse and cropped to the labeled region.

Run from repo root:
  python experiments/MOPRA_Count/run_cube_dlite_map.py \\
    --cube data/CMZ_3mm_HNCO_60.fits \\
    --run-dir experiments/MOPRA_Count/runs/mopra_dlite_<ts>_dlite_coord_baseline \\
    --out data/mopra_cmz_k_pred_dlite_coord.fits --compare-scouse
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
import torch
from astropy.io import fits
from spectral_cube import SpectralCube

_SCRIPT = Path(__file__).resolve()
_MOPRA = _SCRIPT.parent
_REPO = _MOPRA.parents[1]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_MOPRA / "plots"))

from plot_crop_utils import crop_bbox_from_mask, labeled_mask_from_dat  ### noqa: E402
from plot_k_map_image import plot_k_map_figure  ### noqa: E402
from plot_k_residual_map import compare_k_pred_to_scouse  ### noqa: E402

from spectackle.data.generator import _make_v_axis  ### noqa: E402
from spectackle.data.mopra_preprocess import prepare_mopra_input  ### noqa: E402
from spectackle.data.mopra_resample import resample_spec_batch, velocity_axis_kms  ### noqa: E402
from spectackle.models import CenterNet1DDeep  ### noqa: E402
from spectackle.wcs_plot import wcs_celestial  ### noqa: E402


def _load_dlite_model(run_dir: Path, manifest: dict, *, device: str) -> tuple[CenterNet1DDeep, int]:
    args = dict(manifest.get("args", {}))
    cfg = manifest["cfg"]
    Kmax = int(args.get("Kmax", cfg.get("max_components", 6)))
    k_mode = str(args.get("k_mode", "ce"))
    coord = None
    if manifest.get("coord", {}).get("enabled"):
        v_scale = float(manifest["coord"].get("v_scale_kms", 100.0))
        coord = _make_v_axis(cfg).astype(np.float32) / v_scale
    model = CenterNet1DDeep(
        Kmax=Kmax,
        width=int(args["width"]),
        n_blocks=int(args["n_blocks"]),
        coord=coord,
        k_mode=k_mode,
    )
    state = torch.load(run_dir / "center_net.pt", map_location="cpu")
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, Kmax


@torch.no_grad()
def _predict_kv(
    model: CenterNet1DDeep,
    spec_raw: np.ndarray,
    *,
    device: str,
    blank_value: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """spec_raw: (N, C) at the model axis. Returns (K_pred, v_slots, valid_channel_count)."""
    x_norm, chan_mask = prepare_mopra_input(spec_raw, blank_value=blank_value)
    t = torch.from_numpy(x_norm).float().to(device)
    m = torch.from_numpy(chan_mask).float().to(device)
    k_out, v_pred = model(t, m)
    if getattr(model, "k_mode", "ce") == "reg":
        k_pred = torch.clamp(torch.round(k_out), 0, model.Kmax).long()
    else:
        k_pred = k_out.argmax(dim=1)
    k_pred = k_pred.cpu().numpy().astype(np.int16)
    v_slots = v_pred.cpu().numpy().astype(np.float32)
    n_valid = chan_mask.sum(axis=1).astype(np.int32)
    return k_pred, v_slots, n_valid


def main() -> None:
    parser = argparse.ArgumentParser(description="MOPRA D-lite K map from a real PPV cube.")
    parser.add_argument("--cube", type=Path, default=_REPO / "data" / "CMZ_3mm_HNCO_60.fits")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=_REPO / "data" / "mopra_cmz_k_pred_dlite.fits")
    parser.add_argument("--dat", type=Path, default=_REPO / "data" / "final_fits_updated.dat")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--infer-batch", type=int, default=512)
    parser.add_argument("--min-finite-channels", type=int, default=100)
    parser.add_argument("--min-peak", type=float, default=0.02)
    parser.add_argument("--blank-value", type=float, default=-1.0)
    parser.add_argument("--compare-scouse", action="store_true", help="Write dK residual vs --dat.")
    parser.add_argument("--crop-pad", type=int, default=0)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    n_channels = int(manifest["cfg"]["n_channels"])

    print(f"Loading model from {run_dir}", flush=True)
    model, Kmax = _load_dlite_model(run_dir, manifest, device=args.device)
    print(
        f"  CenterNet1DDeep Kmax={Kmax} k_mode={manifest.get('args', {}).get('k_mode', 'ce')} "
        f"coord={manifest.get('coord', {}).get('enabled')}",
        flush=True,
    )

    print(f"Loading cube: {args.cube}", flush=True)
    t0 = time.perf_counter()
    cube = SpectralCube.read(str(args.cube.resolve()), use_dask=False)
    arr = np.asarray(cube.filled(np.nan), dtype=np.float32)
    if arr.ndim == 4:
        arr = arr[0]
    nv, ny, nx = arr.shape
    v_src = cube.spectral_axis.to("km/s").value.astype(np.float64)
    print(f"  shape (v,y,x)=({nv},{ny},{nx})  ({time.perf_counter()-t0:.1f}s)", flush=True)

    v_tgt: np.ndarray | None = None
    if nv != n_channels:
        meta = manifest.get("cfg", {}).get("mopra_meta", {})
        ref = Path(meta.get("axis_cube") or meta.get("cube_path") or args.cube)
        v_tgt = velocity_axis_kms(ref)
        if v_tgt.size != n_channels:
            raise ValueError(f"Ref axis {ref} has {v_tgt.size} ch, model expects {n_channels}.")
        print(f"  resampling {nv} -> {n_channels} ch via {ref.name}", flush=True)

    wcs = wcs_celestial(cube.header)
    if not args.dat.is_file():
        raise FileNotFoundError(f"--dat not found: {args.dat}")
    labeled = labeled_mask_from_dat(args.dat, shape=(ny, nx), wcs=wcs)
    ys, xs = np.where(labeled)
    print(f"Inferring at {ys.size} Scouse-labeled pixels", flush=True)

    k_map = np.full((ny, nx), np.nan, dtype=np.float32)
    v_map = np.full((ny, nx, Kmax), np.nan, dtype=np.float32)
    n_infer = 0
    for i0 in range(0, ys.size, args.infer_batch):
        i1 = min(i0 + args.infer_batch, ys.size)
        by, bx = ys[i0:i1], xs[i0:i1]
        spec = arr[:, by, bx].T.astype(np.float64)  ### (n, nv)
        if v_tgt is not None:
            spec = resample_spec_batch(spec, v_src, v_tgt, blank_value=args.blank_value)
        finite = np.isfinite(np.where(spec == args.blank_value, np.nan, spec))
        n_valid = finite.sum(axis=1)
        peak = np.where(finite, spec, -np.inf).max(axis=1)
        ok = (n_valid >= args.min_finite_channels) & (peak > args.min_peak)
        if not np.any(ok):
            continue
        k_pred, v_slots, _ = _predict_kv(model, spec[ok], device=args.device, blank_value=args.blank_value)
        k_map[by[ok], bx[ok]] = k_pred.astype(np.float32)
        v_map[by[ok], bx[ok], :] = v_slots
        n_infer += int(ok.sum())

    print(f"Inferred {n_infer}/{ys.size} labeled pixels", flush=True)

    wcs_header = wcs.to_header()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    hdu = fits.PrimaryHDU(data=k_map, header=wcs_header)
    hdu.header["BUNIT"] = "1"
    hdu.header["COMMENT"] = "D-lite K_pred (argmax K logits); NaN = not inferred"
    hdu.header["KMAX"] = Kmax
    hdu.header["MOPRARUN"] = run_dir.name[:68]
    hdu.header["DLITE"] = 1
    hdu.writeto(str(args.out.resolve()), overwrite=True)
    print(f"Wrote {args.out}", flush=True)

    ### Sidecar velocity slots for later v-vs-Henshaw comparison (not decoded here).
    v_out = args.out.with_name(f"{args.out.stem}_vslots.npz")
    np.savez_compressed(v_out, yi=ys, xi=xs, v_slots=v_map[ys, xs], k_pred=k_map[ys, xs])
    print(f"Wrote {v_out}", flush=True)

    k_fin = k_map[np.isfinite(k_map)]
    if k_fin.size:
        print(
            f"K stats: min={k_fin.min():.0f} med={np.median(k_fin):.0f} "
            f"max={k_fin.max():.0f} mean={k_fin.mean():.2f}",
            flush=True,
        )

    if not args.no_plot:
        fig_out = _MOPRA / "figures" / f"{args.out.stem}.png"
        crop = crop_bbox_from_mask(labeled, pad=int(args.crop_pad)) if labeled.any() else None
        plot_k_map_figure(
            k_map,
            hdu.header,
            out=fig_out,
            Kmax=Kmax,
            title=f"MOPRA CMZ HNCO: D-lite K_pred ({run_dir.name})",
            cmap_name="Blues",
            cbar_label="K_pred",
            crop=crop,
        )
        print(f"Wrote figure {fig_out}", flush=True)

    if args.compare_scouse:
        fig_stem = (_MOPRA / "figures" / f"{args.out.stem}.png").stem
        report = compare_k_pred_to_scouse(
            k_map,
            dat_path=args.dat,
            cube_path=args.cube,
            out_png=_MOPRA / "figures" / f"{fig_stem}_residual.png",
            out_fits=args.out.with_name(f"{args.out.stem}_residual.fits"),
            crop_to_labels=True,
            crop_pad=args.crop_pad,
        )
        print(
            f"Scouse compare: n={report['n_compare']}  MAE={report['mae']:.3f}  "
            f"exact={report['exact_match_frac']:.3f}  median dK={report['median_delta']:.1f}  "
            f"over={report['frac_over']:.3f} under={report['frac_under']:.3f}",
            flush=True,
        )
        print(f"Wrote residual figure {report['residual_png']}", flush=True)


if __name__ == "__main__":
    main()
