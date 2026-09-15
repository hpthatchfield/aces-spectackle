#!/usr/bin/env python
"""
HNCO region1 K_pred map with Galactic WCS (from an existing cube-map FITS).

  python experiments/ACES_Heatmap/plots/plot_region_k_map.py \\
    --k-pred experiments/ACES_Heatmap/records/<stage2>/hnco_region1_aces_hm_k_pred_islands.fits \\
    --out experiments/ACES_Heatmap/figures/region1/islands_k_map.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPT = Path(__file__).resolve()
_ACES = _SCRIPT.parents[1]
_REPO = _ACES.parents[1]
sys.path.insert(0, str(_SCRIPT.parent))
sys.path.insert(0, str(_REPO / "src"))

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

from plot_region_style import K_PRED, k_discrete_cmap
from spectackle.wcs_plot import (  ### noqa: E402
    style_galactic_wcs_axes,
    suppress_wcsaxes_format_warnings,
    wcs_celestial,
)


def save_k_map_figure(
    k_map: np.ndarray,
    header,
    out: Path,
    *,
    title: str = "HNCO region1",
    dpi: int = 160,
) -> Path:
    if k_map.ndim != 2:
        raise ValueError(f"Expected 2D map, got shape {k_map.shape}")
    finite = np.isfinite(k_map)
    if not np.any(finite):
        raise ValueError("K map contains no finite pixels.")
    k_lo = int(np.nanmin(k_map[finite]))
    k_hi = int(np.nanmax(k_map[finite]))
    cmap, norm, ticks = k_discrete_cmap(k_lo, k_hi)
    wcs = wcs_celestial(header)

    fig = plt.figure(figsize=(6.4, 5.2), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.045], wspace=0.05)
    ax = fig.add_subplot(gs[0, 0], projection=wcs)
    cax = fig.add_subplot(gs[0, 1])
    im = ax.imshow(k_map, origin="lower", cmap=cmap, norm=norm, interpolation="nearest")
    style_galactic_wcs_axes(ax, wcs=wcs, shape_yx=k_map.shape, lon_minpad=0.6, lat_minpad=0.5)
    ax.set_title(title, fontsize=11)
    cb = fig.colorbar(im, cax=cax, orientation="vertical", ticks=ticks)
    cax.yaxis.set_ticks_position("right")
    cb.set_label(K_PRED)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with suppress_wcsaxes_format_warnings():
        fig.savefig(out, dpi=dpi)
    plt.close(fig)
    print(f"Wrote {out}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Region1 K_pred map with Galactic axes.")
    parser.add_argument("--k-pred", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--title", type=str, default="HNCO region1")
    args = parser.parse_args()
    k_map = fits.getdata(args.k_pred).astype(np.float32)
    header = fits.getheader(args.k_pred)
    out = args.out or (_ACES / "figures" / "region1" / f"{args.k_pred.stem}.png")
    save_k_map_figure(k_map, header, out, title=args.title)


if __name__ == "__main__":
    main()
