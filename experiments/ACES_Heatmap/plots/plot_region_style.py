### Shared labels / Galactic helpers for ACES region1 figures.
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from astropy.wcs import WCS
from matplotlib.colors import BoundaryNorm, ListedColormap

K_PRED = r"$K_{\mathrm{pred}}$"
K_HAT = r"$\hat{K}$"
P_CENTER = r"$P(\mathrm{center})$"


def galactic_lb(wcs: WCS, x: float, y: float) -> tuple[float, float]:
    """0-indexed pixel -> Galactic (l, b) in deg, l wrapped to +/-180."""
    sky = wcs.pixel_to_world(float(x), float(y))
    g = sky.galactic
    l = float(g.l.wrap_at("180d").deg)
    b = float(g.b.deg)
    return l, b


def lb_title(l_deg: float, b_deg: float) -> str:
    return rf"$l={l_deg:.3f}$, $b={b_deg:.3f}$"


def k_discrete_cmap(k_lo: int, k_hi: int, *, cmap_name: str = "viridis"):
    n = max(1, int(k_hi) - int(k_lo) + 1)
    base = plt.get_cmap(cmap_name, n)
    cmap = ListedColormap([base(i) for i in range(n)])
    cmap.set_bad("#d9d9d9", alpha=1.0)
    bounds = np.arange(k_lo - 0.5, k_hi + 1.5, 1.0)
    norm = BoundaryNorm(bounds, cmap.N)
    return cmap, norm, list(range(int(k_lo), int(k_hi) + 1))
