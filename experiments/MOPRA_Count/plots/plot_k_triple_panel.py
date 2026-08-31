#!/usr/bin/env python
"""
Clean 3-panel vertical figure: Scouse K, K_pred, dK residual.

Example:
  python experiments/MOPRA_Count/plots/plot_k_triple_panel.py \\
    --k-pred data/mopra_cmz_k_pred_heatmap_realamp_k6_20k.fits \\
    --out experiments/MOPRA_Count/figures/mopra_cmz_k_pred_heatmap_realamp_k6_20k_triple.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPT = Path(__file__).resolve()
_REPO = _SCRIPT.parents[3]
sys.path.insert(0, str(_SCRIPT.parent))
sys.path.insert(0, str(_REPO / "src"))

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from matplotlib.colors import BoundaryNorm, ListedColormap, TwoSlopeNorm

from plot_crop_utils import crop_bbox_from_mask, labeled_mask_from_dat
from plot_k_residual_map import k_true_map_from_dat
from spectackle.wcs_plot import (
    style_galactic_wcs_axes,
    suppress_wcsaxes_format_warnings,
    wcs_celestial,
)


def _k_cmap(k_lo: int, k_hi: int, *, cmap_name: str = "Blues") -> tuple[ListedColormap, BoundaryNorm]:
    n = k_hi - k_lo + 1
    base = plt.get_cmap(cmap_name, max(n, 1))
    cmap = ListedColormap([base(i) for i in range(n)])
    cmap.set_bad("#d9d9d9", alpha=1.0)
    bounds = np.arange(k_lo - 0.5, k_hi + 1.5, 1.0)
    return cmap, BoundaryNorm(bounds, cmap.N)


def plot_triple(
    *,
    k_pred: np.ndarray,
    k_true: np.ndarray,
    header,
    out: Path,
    crop: tuple[int, int, int, int] | None = None,
    dpi: int = 200,
    k_cmap_name: str = "Blues",
    titles: tuple[str, str, str] = (
        "Scouse / Henshaw K",
        r"$K_{\mathrm{pred}}$",
        r"$\Delta K = K_{\mathrm{pred}} - K_{\mathrm{Scouse}}$",
    ),
) -> Path:
    residual = k_pred.astype(np.float64) - k_true.astype(np.float64)
    residual = np.where(np.isfinite(k_pred) & np.isfinite(k_true), residual, np.nan)

    wcs = wcs_celestial(header)
    if crop is not None:
        x0, x1, y0, y1 = crop
        k_pred = k_pred[y0:y1, x0:x1]
        k_true = k_true[y0:y1, x0:x1]
        residual = residual[y0:y1, x0:x1]
        wcs = wcs[y0:y1, x0:x1]

    k_stack = np.concatenate([k_true[np.isfinite(k_true)], k_pred[np.isfinite(k_pred)]])
    k_lo = int(np.min(k_stack))
    k_hi = int(np.max(k_stack))
    cmap_k, norm_k = _k_cmap(k_lo, k_hi, cmap_name=k_cmap_name)

    r_fin = residual[np.isfinite(residual)]
    vmax = max(float(np.nanmax(np.abs(r_fin))), 3.0)
    norm_r = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    cmap_r = plt.get_cmap("RdBu_r").copy()
    cmap_r.set_bad("#d9d9d9", alpha=1.0)

    ny, nx = k_pred.shape
    aspect = nx / max(ny, 1)
    fig_w = min(8.5, max(5.8, 3.8 * aspect))
    fig_h = fig_w / max(aspect, 0.2) * 3.15
    fig_h = min(14.0, max(9.0, fig_h))

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(
        3,
        2,
        height_ratios=[1.0, 1.0, 1.0],
        width_ratios=[1.0, 0.04],
        hspace=0.18,
        wspace=0.06,
        left=0.14,
        right=0.92,
        top=0.97,
        bottom=0.04,
    )

    panels = [
        (k_true, titles[0], cmap_k, norm_k, f"K ({k_lo}..{k_hi})", list(range(k_lo, k_hi + 1))),
        (k_pred, titles[1], cmap_k, norm_k, f"K ({k_lo}..{k_hi})", list(range(k_lo, k_hi + 1))),
        (residual, titles[2], cmap_r, norm_r, r"$\Delta K$", None),
    ]

    for i, (data, title, cmap, norm, cbar_label, ticks) in enumerate(panels):
        ax = fig.add_subplot(gs[i, 0], projection=wcs)
        cax = fig.add_subplot(gs[i, 1])
        im = ax.imshow(data, origin="lower", cmap=cmap, norm=norm, interpolation="nearest")
        style_galactic_wcs_axes(ax, wcs=wcs, shape_yx=data.shape, lon_minpad=0.35, lat_minpad=0.35)
        ax.set_title(title, fontsize=11, pad=6)
        if i < 2:
            ### Drop lon labels on upper panels to reduce clutter.
            ax.coords[0].set_ticklabel_visible(False)
        cb = fig.colorbar(im, cax=cax, orientation="vertical", ticks=ticks)
        cax.yaxis.set_ticks_position("right")
        cax.yaxis.set_label_position("right")
        cax.xaxis.set_visible(False)
        cb.set_label(cbar_label, fontsize=9)

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with suppress_wcsaxes_format_warnings():
        fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="3-panel Scouse / K_pred / residual map.")
    parser.add_argument(
        "--k-pred",
        type=Path,
        default=_REPO / "data" / "mopra_cmz_k_pred_heatmap_realamp_k6_20k.fits",
    )
    parser.add_argument("--dat", type=Path, default=_REPO / "data" / "final_fits_updated.dat")
    parser.add_argument("--cube", type=Path, default=_REPO / "data" / "CMZ_3mm_HNCO_60.fits")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--crop-pad", type=int, default=2)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--cmap", type=str, default="Blues")
    args = parser.parse_args()

    with fits.open(args.k_pred) as hdul:
        k_pred = np.asarray(hdul[0].data, dtype=np.float64)
        header = hdul[0].header

    ### Prefer cube WCS if spatial shape matches (k-pred may be header-light).
    cube_hdr = fits.getheader(args.cube)
    if k_pred.shape == (int(cube_hdr["NAXIS2"]), int(cube_hdr["NAXIS1"])):
        header = cube_hdr

    wcs = wcs_celestial(header)
    k_true, _ = k_true_map_from_dat(args.dat, shape=k_pred.shape, wcs=wcs)
    labeled = labeled_mask_from_dat(args.dat, shape=k_pred.shape, wcs=wcs)
    crop = crop_bbox_from_mask(labeled, pad=int(args.crop_pad))

    out = args.out
    if out is None:
        out = (
            _REPO
            / "experiments"
            / "MOPRA_Count"
            / "figures"
            / f"{args.k_pred.stem}_triple.png"
        )

    path = plot_triple(
        k_pred=k_pred,
        k_true=k_true,
        header=header,
        out=out,
        crop=crop,
        dpi=args.dpi,
        k_cmap_name=args.cmap,
    )
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
