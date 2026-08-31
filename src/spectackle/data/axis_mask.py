### ALMA-style spectral-axis padding - valid island, NaN moat, exact-zero outer pad.
from __future__ import annotations

import numpy as np


def draw_valid_island(c: int, gen: dict, rng: np.random.Generator) -> tuple[int, int, int, int]:
    """
    Return (i0, i1, nan_left, nan_right): the valid-island channel span [i0, i1) plus the
    NaN-moat widths just outside it on each side (remaining outer channels are exact-zero pad).
    Fully valid (no mask) => (0, c, 0, 0).
    """
    if rng.random() >= float(gen.get("mask_prob", 0.0)):
        return 0, c, 0, 0
    f = float(rng.uniform(*gen["valid_frac_range"]))
    wlen = max(2, min(c, int(round(f * c))))
    i0 = int(rng.integers(0, c - wlen + 1))
    i1 = i0 + wlen
    mf = float(rng.uniform(*gen["nan_moat_frac_range"]))
    nan_left = int(round(mf * i0))
    nan_right = int(round(mf * (c - i1)))
    return i0, i1, nan_left, nan_right


def apply_axis_mask(arr: np.ndarray, i0: int, i1: int, nan_left: int, nan_right: int) -> np.ndarray:
    """Set channels outside [i0, i1) to NaN (moat adjacent to island) then 0.0 (outer pad)."""
    c = int(arr.shape[0])
    if i0 <= 0 and i1 >= c:
        return arr
    out = arr.copy()
    if i0 > 0:
        moat_start = max(0, i0 - nan_left)
        out[:moat_start] = 0.0
        out[moat_start:i0] = np.nan
    if i1 < c:
        moat_end = min(c, i1 + nan_right)
        out[i1:moat_end] = np.nan
        out[moat_end:] = 0.0
    return out


__all__ = ["draw_valid_island", "apply_axis_mask"]
