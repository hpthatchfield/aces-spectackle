#!/usr/bin/env python
"""
Publication-style plot for a 2D MOPRA K_pred map FITS (Galactic WCS).

Example:
  python experiments/MOPRA_Count/plots/plot_k_map_image.py \\
    --k-map data/mopra_cmz_k_pred.fits \\
    --out experiments/MOPRA_Count/figures/mopra_cmz_k_pred.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPT = Path(__file__).resolve()
sys.path.insert(0, str(_SCRIPT.parent))
sys.path.insert(0, str(_SCRIPT.parents[3] / "src"))

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from matplotlib.colors import BoundaryNorm, ListedColormap

from plot_crop_utils import crop_bbox_from_mask, labeled_mask_from_dat
from spectackle.wcs_plot import (  ### noqa: E402
    style_galactic_wcs_axes,
    suppress_wcsaxes_format_warnings,
    wcs_celestial,
)


def _nice_length_arcsec(target_arcsec: float) -> float:
    if target_arcsec <= 0:
        return 1.0
    exp = int(np.floor(np.log10(target_arcsec)))
    base = 10.0**exp
    for m in (5.0, 2.0, 1.0):
        if m * base <= target_arcsec:
            return m * base
    return base


def _pixel_scales_deg(w: WCS) -> tuple[float, float]:
    cdelt = w.wcs.cdelt
    if cdelt is None or len(cdelt) < 2:
        raise ValueError("WCS missing CDELT for 2D celestial axes.")
    return abs(float(cdelt[0])), abs(float(cdelt[1]))


def _k_cmap(
    k_min: int,
    k_max: int,
    *,
    cmap_name: str = "Blues",
    bad_color: str = "#d9d9d9",
) -> tuple[ListedColormap, BoundaryNorm]:
    ### One discrete color per integer K in [k_min, k_max] (inclusive).
    if k_max < k_min:
        raise ValueError(f"k_max must be >= k_min, got ({k_min}, {k_max})")
    n = k_max - k_min + 1
    base = plt.get_cmap(cmap_name, max(n, 1))
    colors = [base(i) for i in range(n)]
    cmap = ListedColormap(colors)
    cmap.set_bad(bad_color, alpha=1.0)
    bounds = np.arange(k_min - 0.5, k_max + 1.5, 1.0)
    norm = BoundaryNorm(bounds, cmap.N)
    return cmap, norm


def plot_k_map_figure(
    k_map: np.ndarray,
    header,
    *,
    out: Path,
    Kmax: int | None = None,
    title: str | None = None,
    cmap_name: str = "Blues",
    cbar_label: str = "K_pred (components)",
    scalebar_frac: float = 0.25,
    dpi: int = 200,
    crop: tuple[int, int, int, int] | None = None,
    scale_to_data: bool = True,
    blank_as_zero: bool = False,
) -> Path:
    """
    Render a publication-style K map with Galactic WCS axes.
    k_map: 2D float array (NaN = not inferred).
    header: astropy FITS header with celestial WCS.
    scale_to_data: if True, colorbar spans min..max K in finite pixels (default).
    blank_as_zero: render NaN / not-inferred pixels as K=0 (white when scale starts at 0).
    """
    if k_map.ndim != 2:
        raise ValueError(f"Expected 2D map, got shape {k_map.shape}")

    wcs = wcs_celestial(header)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    data = k_map
    if crop is not None:
        x0, x1, y0, y1 = crop
        data = k_map[y0:y1, x0:x1]
        ### Numpy is [y, x]; astropy WCS slices in the same axis order.
        wcs = wcs[y0:y1, x0:x1]

    finite = np.isfinite(data)
    if not np.any(finite):
        raise ValueError("K map contains no finite pixels.")

    k_fin = data[finite]
    k_int_min = int(np.nanmin(k_fin))
    k_int_max = int(np.nanmax(k_fin))
    if Kmax is None:
        kmax_hdr = header.get("KMAX")
        Kmax = int(kmax_hdr) if kmax_hdr is not None else k_int_max
    Kmax = max(int(Kmax), k_int_max)

    if scale_to_data:
        k_lo, k_hi = k_int_min, k_int_max
    else:
        k_lo, k_hi = 0, Kmax

    bad_color = "#ffffff" if blank_as_zero else "#d9d9d9"
    cmap, norm = _k_cmap(k_lo, k_hi, cmap_name=cmap_name, bad_color=bad_color)
    plot_data = np.where(np.isfinite(data), data, 0.0) if blank_as_zero else data

    ### GridSpec: optional title row + map/cbar row share identical height.
    n_rows = 2 if title else 1
    height_ratios = [0.07, 1.0] if title else [1.0]
    ny, nx = data.shape
    aspect = nx / max(ny, 1)
    fig_w = min(9.0, max(5.5, 4.2 * aspect))
    fig_h = min(7.0, max(3.8, fig_w / max(aspect, 0.25)))
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(
        n_rows,
        2,
        height_ratios=height_ratios,
        width_ratios=[1.0, 0.045],
        hspace=0.04 if title else 0.0,
        wspace=0.06,
        left=0.16,
        right=0.93,
        top=0.95,
        bottom=0.14,
    )
    map_row = 1 if title else 0
    if title:
        ax_title = fig.add_subplot(gs[0, :])
        ax_title.axis("off")
        ax_title.text(0.5, 0.5, title, ha="center", va="center", fontsize=11)

    ax = fig.add_subplot(gs[map_row, 0], projection=wcs)
    cax = fig.add_subplot(gs[map_row, 1])
    im = ax.imshow(plot_data, origin="lower", cmap=cmap, norm=norm, interpolation="nearest")

    style_galactic_wcs_axes(ax, wcs=wcs, shape_yx=data.shape, lon_minpad=0.7, lat_minpad=0.5)

    ticks = list(range(k_lo, k_hi + 1))
    cb = fig.colorbar(im, cax=cax, orientation="vertical", ticks=ticks)
    cax.yaxis.set_ticks_position("right")
    cax.yaxis.set_label_position("right")
    cax.xaxis.set_visible(False)
    cb.set_label(cbar_label)

    dx_deg, dy_deg = _pixel_scales_deg(wcs.celestial)
    arcmin_per_pix = 60.0 * 0.5 * (dx_deg + dy_deg)
    if arcmin_per_pix > 0:
        frame_arcsec = data.shape[1] * arcmin_per_pix * 60.0
        length_arcsec = _nice_length_arcsec(float(scalebar_frac) * frame_arcsec)
        length_arcmin = length_arcsec / 60.0
        label = f"{length_arcsec:g}\"" if length_arcsec < 60.0 else f"{length_arcmin:g}'"
        n_pix = length_arcmin / arcmin_per_pix
        # bar = AnchoredSizeBar(
        #     ax.transData,
        #     n_pix,
        #     label,
        #     loc="lower left",
        #     pad=0.35,
        #     color="white",
        #     frameon=True,
        #     size_vertical=2.5,
        #     fontproperties={"size": 9},
        #)
        # bar.patch.set_alpha(0.45)
        # bar.patch.set_facecolor("k")
        # bar.patch.set_edgecolor("k")
        # ax.add_artist(bar)

    with suppress_wcsaxes_format_warnings():
        fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot a MOPRA K_pred map as a WCS image.")
    parser.add_argument("--k-map", type=Path, required=True, help="2D K_pred FITS map.")
    parser.add_argument("--out", type=Path, default=None, help="Output PNG/PDF (default: alongside input).")
    parser.add_argument("--Kmax", type=int, default=None, help="Model Kmax for --full-kmax-scale (default: FITS header).")
    parser.add_argument(
        "--full-kmax-scale",
        action="store_true",
        help="Colorbar 0..Kmax from header (default: only K values present in map).",
    )
    parser.add_argument(
        "--blank-as-zero",
        action="store_true",
        help="Show not-inferred (NaN) pixels as K=0 with white background.",
    )
    parser.add_argument("--cmap", type=str, default="Blues")
    parser.add_argument("--title", type=str, default=None)
    parser.add_argument("--dpi", type=int, default=200)
    _REPO = Path(__file__).resolve().parents[3]
    parser.add_argument("--dat", type=Path, default=_REPO / "data" / "final_fits_updated.dat")
    parser.add_argument(
        "--cube",
        type=Path,
        default=_REPO / "data" / "CMZ_3mm_HNCO_60.fits",
        help="Reference cube WCS for --crop-to-labels (must match k-map grid).",
    )
    parser.add_argument(
        "--crop-to-labels",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Crop PNG to Scouse-labeled region (default: on).",
    )
    parser.add_argument("--crop-pad", type=int, default=0)
    args = parser.parse_args()

    data = fits.getdata(args.k_map, memmap=True).astype(np.float32)
    header = fits.getheader(args.k_map)
    out = args.out if args.out is not None else args.k_map.with_suffix(".png")

    crop = None
    if args.crop_to_labels and args.dat.is_file():
        cube_header = fits.getheader(args.cube)
        wcs = wcs_celestial(cube_header)
        if data.shape != (cube_header["NAXIS2"], cube_header["NAXIS1"]):
            raise ValueError(
                f"k-map shape {data.shape} != cube spatial "
                f"({cube_header['NAXIS2']}, {cube_header['NAXIS1']})"
            )
        labeled = labeled_mask_from_dat(args.dat, shape=data.shape, wcs=wcs)
        crop = crop_bbox_from_mask(labeled, pad=int(args.crop_pad))

    path = plot_k_map_figure(
        data,
        header,
        out=out,
        Kmax=args.Kmax,
        title=args.title,
        cmap_name=args.cmap,
        dpi=args.dpi,
        crop=crop,
        scale_to_data=not args.full_kmax_scale,
        blank_as_zero=args.blank_as_zero,
    )
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
