#!/usr/bin/env python
"""
K_pred - K_true residual map (ML vs Scouse/Henshaw labels from .dat).

Example:
  python experiments/MOPRA_Count/plots/plot_k_residual_map.py \\
    --k-pred data/mopra_cmz_k_pred_biased_low_sep.fits \\
    --dat data/final_fits_updated.dat \\
    --cube data/CMZ_3mm_HNCO_60.fits \\
    --out experiments/MOPRA_Count/figures/mopra_k_residual_biased_low_sep.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPT = Path(__file__).resolve()
_REPO = _SCRIPT.parents[3]
sys.path.insert(0, str(_SCRIPT.parent))
sys.path.insert(0, str(_REPO / "src"))

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from matplotlib.colors import TwoSlopeNorm

from plot_crop_utils import crop_bbox_from_mask, labeled_mask_from_dat
from spectackle.wcs_plot import (  ### noqa: E402
    style_galactic_wcs_axes,
    suppress_wcsaxes_format_warnings,
    wcs_celestial,
)


def k_true_map_from_dat(
    dat_path: Path,
    *,
    shape: tuple[int, int],
    wcs: WCS,
    use_row_count: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Rasterize Scouse K labels onto a 2D grid.

    Returns (k_true, labeled_mask) where labeled_mask is True at Scouse pixels.
    """
    arr = np.loadtxt(dat_path)
    ny, nx = shape
    k_true = np.full((ny, nx), np.nan, dtype=np.float32)
    labeled = np.zeros((ny, nx), dtype=bool)

    ### Group rows by sky position; col0 = ncomps, or use row count if requested.
    from collections import defaultdict

    by_pos: dict[tuple[float, float], list[np.ndarray]] = defaultdict(list)
    for row in arr:
        key = (round(float(row[1]), 5), round(float(row[2]), 5))
        by_pos[key].append(row)

    for (l, b), rows in by_pos.items():
        xp, yp = wcs.all_world2pix([[l, b]], 0)[0]
        xi, yi = int(round(xp)), int(round(yp))
        if not (0 <= xi < nx and 0 <= yi < ny):
            continue
        if use_row_count:
            k = len(rows)
        else:
            k = int(rows[0][0])
        k_true[yi, xi] = float(k)
        labeled[yi, xi] = True

    return k_true, labeled


def plot_residual_figure(
    residual: np.ndarray,
    header,
    *,
    out: Path,
    title: str | None = None,
    dpi: int = 200,
    crop: tuple[int, int, int, int] | None = None,
    sym_lim_min: float = 5.0,
) -> Path:
    wcs = wcs_celestial(header)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    data = residual
    if crop is not None:
        x0, x1, y0, y1 = crop
        data = residual[y0:y1, x0:x1]
        wcs = wcs[y0:y1, x0:x1]

    finite = np.isfinite(data)
    if not np.any(finite):
        raise ValueError("Residual map contains no finite pixels.")

    r = data[finite]
    vmax = max(float(np.nanmax(np.abs(r))), float(sym_lim_min))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#d9d9d9", alpha=1.0)

    ny, nx = data.shape
    aspect = nx / max(ny, 1)
    fig_w = min(9.0, max(5.5, 4.2 * aspect))
    fig_h = min(7.0, max(3.8, fig_w / max(aspect, 0.25)))

    n_rows = 2 if title else 1
    height_ratios = [0.07, 1.0] if title else [1.0]
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(
        n_rows,
        2,
        height_ratios=height_ratios,
        width_ratios=[1.0, 0.045],
        hspace=0.04 if title else 0.0,
        wspace=0.06,
        left=0.14,
        right=0.93,
        top=0.95,
        bottom=0.11,
    )
    map_row = 1 if title else 0
    if title:
        ax_title = fig.add_subplot(gs[0, :])
        ax_title.axis("off")
        ax_title.text(0.5, 0.5, title, ha="center", va="center", fontsize=11)

    ax = fig.add_subplot(gs[map_row, 0], projection=wcs)
    cax = fig.add_subplot(gs[map_row, 1])
    im = ax.imshow(data, origin="lower", cmap=cmap, norm=norm, interpolation="nearest")
    style_galactic_wcs_axes(ax, wcs=wcs, shape_yx=data.shape, lon_minpad=0.35, lat_minpad=0.35)

    cb = fig.colorbar(im, cax=cax, orientation="vertical")
    cax.yaxis.set_ticks_position("right")
    cax.yaxis.set_label_position("right")
    cax.xaxis.set_visible(False)
    cb.set_label(r"$\Delta K = K_{\rm pred} - K_{\rm Scouse}$")

    with suppress_wcsaxes_format_warnings():
        fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return out


def compare_k_pred_to_scouse(
    k_pred: np.ndarray,
    *,
    dat_path: Path,
    cube_path: Path,
    out_png: Path,
    out_fits: Path | None = None,
    report_json: Path | None = None,
    title: str | None = None,
    crop_to_labels: bool = True,
    crop_pad: int = 0,
    use_row_count: bool = False,
    sym_lim_min: float = 5.0,
) -> dict:
    """
    Build dK = K_pred - K_Scouse maps and write FITS, PNG, JSON report.

    k_pred: 2D array aligned with cube_path spatial grid.
    """
    cube_header = fits.getheader(cube_path)
    wcs = wcs_celestial(cube_header)
    if k_pred.shape != (cube_header["NAXIS2"], cube_header["NAXIS1"]):
        raise ValueError(
            f"k-pred shape {k_pred.shape} != cube spatial "
            f"({cube_header['NAXIS2']}, {cube_header['NAXIS1']})"
        )

    k_true, labeled = k_true_map_from_dat(
        dat_path,
        shape=k_pred.shape,
        wcs=wcs,
        use_row_count=use_row_count,
    )
    compare = labeled & np.isfinite(k_pred)
    residual = np.full(k_pred.shape, np.nan, dtype=np.float32)
    residual[compare] = k_pred[compare] - k_true[compare]

    n = int(compare.sum())
    if n == 0:
        raise ValueError("No overlapping labeled + inferred pixels for comparison.")

    diff = residual[compare]
    report = {
        "n_compare": n,
        "exact_match_frac": float(np.mean(diff == 0)),
        "mae": float(np.mean(np.abs(diff))),
        "median_delta": float(np.median(diff)),
        "frac_over": float(np.mean(diff > 0)),
        "frac_under": float(np.mean(diff < 0)),
        "k_true_median": float(np.median(k_true[compare])),
        "k_pred_median": float(np.median(k_pred[compare])),
        "dat_path": str(dat_path.resolve()),
    }

    wcs_header = wcs[:, :].to_header()
    out_fits = out_fits if out_fits is not None else out_png.with_suffix(".fits")
    hdu = fits.PrimaryHDU(data=residual, header=wcs_header)
    hdu.header["BUNIT"] = "1"
    hdu.header["COMMENT"] = "K_pred - K_Scouse; NaN outside overlap"
    out_fits.parent.mkdir(parents=True, exist_ok=True)
    hdu.writeto(str(out_fits.resolve()), overwrite=True)

    crop = None
    if crop_to_labels:
        crop = crop_bbox_from_mask(labeled, pad=int(crop_pad))
        report["crop_bbox"] = {"x0": crop[0], "x1": crop[1], "y0": crop[2], "y1": crop[3]}

    if title is None:
        title = r"$\Delta K$: ML vs Scouse ($K_{\rm pred} - K_{\rm true}$)"
    fig_path = plot_residual_figure(
        residual,
        wcs_header,
        out=out_png,
        title=title,
        crop=crop,
        sym_lim_min=sym_lim_min,
    )

    report_path = report_json if report_json is not None else out_fits.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["residual_fits"] = str(out_fits.resolve())
    report["residual_png"] = str(fig_path.resolve())
    report["report_json"] = str(report_path.resolve())
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="ML vs Scouse K residual map.")
    parser.add_argument("--k-pred", type=Path, required=True)
    parser.add_argument("--dat", type=Path, default=_REPO / "data" / "final_fits_updated.dat")
    parser.add_argument(
        "--cube",
        type=Path,
        default=_REPO / "data" / "CMZ_3mm_HNCO_60.fits",
        help="Cube for WCS when building K_true from .dat (must match k-pred grid).",
    )
    parser.add_argument("--out", type=Path, default=None, help="Output PNG.")
    parser.add_argument("--out-fits", type=Path, default=None, help="Output residual FITS.")
    parser.add_argument("--use-row-count", action="store_true", help="K_true = n component rows.")
    parser.add_argument("--title", type=str, default=None)
    parser.add_argument("--report-json", type=Path, default=None)
    parser.add_argument(
        "--crop-to-labels",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Crop PNG to Scouse-labeled region (default: on).",
    )
    parser.add_argument("--crop-pad", type=int, default=0, help="Pixel padding around crop bbox.")
    parser.add_argument(
        "--sym-lim-min",
        type=float,
        default=5.0,
        help="Minimum half-range for symmetric dK color scale (default: 5).",
    )
    args = parser.parse_args()

    k_pred = fits.getdata(args.k_pred).astype(np.float32)
    out_png = args.out
    if out_png is None:
        out_png = _SCRIPT.parent.parent / "figures" / f"{args.k_pred.stem}_residual.png"

    report = compare_k_pred_to_scouse(
        k_pred,
        dat_path=args.dat,
        cube_path=args.cube,
        out_png=out_png,
        out_fits=args.out_fits,
        report_json=args.report_json,
        title=args.title,
        crop_to_labels=args.crop_to_labels,
        crop_pad=args.crop_pad,
        use_row_count=args.use_row_count,
        sym_lim_min=args.sym_lim_min,
    )

    print(f"Compared {report['n_compare']} pixels")
    print(f"  exact match: {report['exact_match_frac']:.3f}")
    print(f"  MAE: {report['mae']:.3f}")
    print(f"  median dK: {report['median_delta']:.1f}")
    print(f"Wrote {report['residual_fits']}")
    print(f"Wrote {report['residual_png']}")
    print(f"Wrote {report['report_json']}")


if __name__ == "__main__":
    main()
