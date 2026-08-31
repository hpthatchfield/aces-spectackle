### Shared crop helpers for MOPRA K map figures.
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
from astropy.wcs import WCS


def crop_bbox_from_mask(mask: np.ndarray, *, pad: int = 0) -> tuple[int, int, int, int]:
    """Return x0, x1, y0, y1 pixel bounds (x1/y1 exclusive) with optional padding."""
    ys, xs = np.where(mask)
    if ys.size == 0:
        raise ValueError("Cannot crop: mask is empty.")
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(mask.shape[0], int(ys.max()) + pad + 1)
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(mask.shape[1], int(xs.max()) + pad + 1)
    return x0, x1, y0, y1


def labeled_mask_from_dat(
    dat_path: Path,
    *,
    shape: tuple[int, int],
    wcs: WCS,
) -> np.ndarray:
    """True at pixel locations present in a Scouse .dat table."""
    arr = np.loadtxt(dat_path)
    ny, nx = shape
    labeled = np.zeros((ny, nx), dtype=bool)

    by_pos: dict[tuple[float, float], list] = defaultdict(list)
    for row in arr:
        key = (round(float(row[1]), 5), round(float(row[2]), 5))
        by_pos[key].append(row)

    for (l, b) in by_pos:
        xp, yp = wcs.all_world2pix([[l, b]], 0)[0]
        xi, yi = int(round(xp)), int(round(yp))
        if 0 <= xi < nx and 0 <= yi < ny:
            labeled[yi, xi] = True

    return labeled
