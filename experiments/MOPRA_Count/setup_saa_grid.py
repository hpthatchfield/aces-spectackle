#!/usr/bin/env python
"""
Build SCouse-style SAA grid on the MOPRA CMZ cube (stage-1 coverage).

Uses default scousepy tiling: regular coverage, Nyquist centre spacing (wsaa/2),
fillfactor=0.5, moment-0 mask at itol_sigma x sigma_rms (CMZ default 3sigma).

Example (repo root):
  python experiments/MOPRA_Count/setup_saa_grid.py \\
    --out experiments/MOPRA_Count/runs/saa_grid_cmz_baseline
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from spectral_cube import SpectralCube

_SCRIPT = Path(__file__).resolve()
_REPO = _SCRIPT.parents[2]
_DEFAULT_CUBE = _REPO / "data" / "CMZ_3mm_HNCO.fits"
sys.path.insert(0, str(_REPO / "src"))

from spectackle.data.scouse_saa import (  ### noqa: E402
    CMZ_R_SAA_DEG,
    CMZ_SCOUSE_TOL,
    ScouseSaaConfig,
    build_saa_grid,
    estimate_cube_rms,
    wsaa_from_r_saa,
)
from spectackle.wcs_plot import (  ### noqa: E402
    style_galactic_wcs_axes,
    suppress_wcsaxes_format_warnings,
    wcs_celestial,
)


def _write_coverage_config(out_dir: Path, cfg: ScouseSaaConfig, wsaa: int, mask_below: float) -> None:
    ### Minimal coverage.config for future scousepy stage_1 (non-interactive).
    lines = [
        "[DEFAULT]",
        "nrefine = 1",
        f"mask_below = {mask_below}",
        "mask_coverage = None",
        f"x_range = [{cfg.xmin}, {cfg.xmax if cfg.xmax is not None else 'None'}]",
        f"y_range = [{cfg.ymin}, {cfg.ymax if cfg.ymax is not None else 'None'}]",
        "vel_range = [None, None]",
        f"wsaa = [{wsaa}]",
        f"fillfactor = [{cfg.fillfactor}]",
        "samplesize = 0",
        f"covmethod = '{cfg.covmethod}'",
        f"spacing = '{cfg.spacing}'",
        "speccomplexity = 'momdiff'",
        "",
    ]
    (out_dir / "coverage.config").write_text("\n".join(lines), encoding="utf-8")

    tol_str = ", ".join(str(t) for t in cfg.tol)
    scouse_lines = [
        "[DEFAULT]",
        f"tol = [{tol_str}]",
        "",
    ]
    (out_dir / "scouse_tol_snippet.config").write_text("\n".join(scouse_lines), encoding="utf-8")


def _plot_coverage(
    moment_mask: np.ndarray,
    coverage: np.ndarray,
    saa_list: list[dict],
    wsaa: int,
    *,
    out: Path,
    wcs_header,
) -> None:
    fig = plt.figure(figsize=(9, 6))
    wcs = wcs_celestial(wcs_header)
    ax = fig.add_subplot(111, projection=wcs)
    ax.imshow(moment_mask.astype(float), origin="lower", cmap="gray_r", vmin=0, vmax=1, alpha=0.85)
    for row in coverage:
        if not row[2]:
            continue
        cx, cy = row[0], row[1]
        bl_x, bl_y = cx - wsaa / 2, cy - wsaa / 2
        rect = plt.Rectangle(
            (bl_x, bl_y),
            wsaa,
            wsaa,
            fill=False,
            edgecolor="cyan",
            linewidth=0.35,
            transform=ax.get_transform("pixel"),
        )
        ax.add_patch(rect)
    style_galactic_wcs_axes(ax, wcs=wcs, shape_yx=moment_mask.shape)
    ax.set_title(f"SAA grid (kept={len(saa_list)}, wsaa={wsaa} px, Nyquist spacing)")
    fig.tight_layout()
    with suppress_wcsaxes_format_warnings():
        fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="SCouse-style SAA grid for MOPRA CMZ.")
    parser.add_argument("--cube", type=Path, default=_DEFAULT_CUBE)
    parser.add_argument("--out", type=Path, default=_SCRIPT.parent / "runs" / "saa_grid_cmz")
    parser.add_argument("--r-saa-deg", type=float, default=CMZ_R_SAA_DEG)
    parser.add_argument(
        "--r-is-radius",
        action="store_true",
        help="Interpret r-saa-deg as radius (default: square width = Henshaw/Jones CMZ).",
    )
    parser.add_argument("--wsaa-pix", type=int, default=None, help="Override computed wsaa.")
    parser.add_argument("--itol-sigma", type=float, default=3.0, help="Moment mask: keep T > itol * sigma_rms.")
    parser.add_argument("--fillfactor", type=float, default=0.5)
    parser.add_argument("--spacing", type=str, default="nyquist", choices=["nyquist", "regular"])
    parser.add_argument("--rms-samples", type=int, default=512)
    parser.add_argument("--blank-value", type=float, default=-1.0)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"Loading cube: {args.cube}", flush=True)
    cube = SpectralCube.read(str(args.cube.resolve()), use_dask=True)
    nv, ny, nx = cube.shape
    wcs_2d = WCS(cube.header).celestial

    arr = cube.filled(np.nan)
    if hasattr(arr, "compute"):
        arr = arr.compute()
    cube_vyx = np.asarray(arr, dtype=np.float32)

    dx_deg, _ = abs(float(wcs_2d.wcs.cdelt[0])), abs(float(wcs_2d.wcs.cdelt[1]))
    wsaa_est = (
        args.wsaa_pix
        if args.wsaa_pix is not None
        else wsaa_from_r_saa(args.r_saa_deg, dx_deg, r_is_radius=args.r_is_radius)
    )
    cube_rms = estimate_cube_rms(cube_vyx, sample_pixels=args.rms_samples)

    cfg = ScouseSaaConfig(
        r_saa_deg=args.r_saa_deg,
        r_is_radius=args.r_is_radius,
        wsaa_pix=args.wsaa_pix,
        itol_sigma=args.itol_sigma,
        fillfactor=args.fillfactor,
        spacing=args.spacing,
        tol=CMZ_SCOUSE_TOL,
        xmax=nx,
        ymax=ny,
    )

    print(
        f"  shape (v,y,x)=({nv},{ny},{nx})  pixel_scale~{dx_deg:.5f} deg/pix  "
        f"wsaa~{wsaa_est} pix  sigma_rms~{cube_rms:.4f} K  mask_below~{args.itol_sigma * cube_rms:.4f} K",
        flush=True,
    )

    result = build_saa_grid(
        cube_vyx,
        cfg,
        pixel_scale_x_deg=dx_deg,
        cube_rms=cube_rms,
        blank_value=args.blank_value,
    )

    wsaa = int(result["wsaa"])
    mask_below = float(result["mask_below"])
    saa_list = result["saa_list"]

    ### Save spectra + indices (compact catalog).
    catalog = {
        "cube": str(args.cube.resolve()),
        "wsaa": wsaa,
        "spacing": result["spacing"],
        "mask_below": mask_below,
        "cube_rms": cube_rms,
        "itol_sigma": args.itol_sigma,
        "r_saa_deg": args.r_saa_deg,
        "r_is_radius": args.r_is_radius,
        "r_is_width": not args.r_is_radius,
        "n_saa_kept": len(saa_list),
        "n_saa_total": int(result["n_saa_total"]),
        "tol": list(cfg.tol),
        "saa": [
            {
                "saa_id": s["saa_id"],
                "center_x": int(s["center_x"]),
                "center_y": int(s["center_y"]),
                "n_pixels": int(s["n_pixels"]),
                "pixel_y": s["pixel_y"].tolist(),
                "pixel_x": s["pixel_x"].tolist(),
            }
            for s in saa_list
        ],
    }
    (args.out / "saa_catalog.json").write_text(json.dumps(catalog, indent=2), encoding="utf-8")

    if not saa_list:
        raise RuntimeError(
            "No SAAs passed fillfactor/mask checks. Try lowering --itol-sigma or --fillfactor."
        )
    spectra = np.stack([s["spectrum"] for s in saa_list], axis=0).astype(np.float32)
    np.savez_compressed(
        args.out / "saa_spectra.npz",
        spectra=spectra,
        saa_id=np.array([s["saa_id"] for s in saa_list], dtype=np.int32),
        center_x=np.array([s["center_x"] for s in saa_list], dtype=np.int32),
        center_y=np.array([s["center_y"] for s in saa_list], dtype=np.int32),
        n_pixels=np.array([s["n_pixels"] for s in saa_list], dtype=np.int32),
        wsaa=wsaa,
        mask_below=mask_below,
    )

    _write_coverage_config(args.out, cfg, wsaa, mask_below)

    if not args.no_plot:
        trim = result["trim"]
        hdr = wcs_2d[trim["xmin"] : trim["xmax"], trim["ymin"] : trim["ymax"]].to_header()
        _plot_coverage(
            result["moment_mask"],
            result["coverage"],
            saa_list,
            wsaa,
            out=args.out / "saa_coverage.png",
            wcs_header=hdr,
        )

    print(
        f"Kept {len(saa_list)} / {result['n_saa_total']} SAAs  "
        f"(spacing={result['spacing']:.2f} px, fill>={args.fillfactor})",
        flush=True,
    )
    print(f"Wrote {args.out}/", flush=True)


if __name__ == "__main__":
    main()
