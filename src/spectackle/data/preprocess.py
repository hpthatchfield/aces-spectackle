### Shared spectrum preprocessing: training and real-cube inference use the same contract.
from __future__ import annotations

import numpy as np

### Channels are "valid" (real measurements) iff finite AND not an exact-zero pad.
### ALMA mosaics pad with NaN moats and exact 0.0 outside coverage; real pbcor flux is
### essentially never exactly 0.0, so this distinguishes pad from signal.


def valid_mask(spec_raw: np.ndarray) -> np.ndarray:
    """Boolean validity mask, same shape as spec_raw. True = real measured channel."""
    a = np.asarray(spec_raw)
    return np.isfinite(a) & (a != 0.0)


def prepare_spectrum_input(spec_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Normalize a spectrum (or batch) using statistics over VALID channels only.

    Accepts shape (C,) or (B, C). Returns (x, mask) with the same shape:
      x    : float32, (value - mean_valid) / std_valid; invalid channels set to 0
      mask : float32, 1.0 on valid channels, 0.0 on NaN/zero-pad

    Valid-only stats keep norm independent of pad fraction; mask pools over real channels only.
    """
    a = np.asarray(spec_raw, dtype=np.float64)
    one_d = a.ndim == 1
    if one_d:
        a = a[None, :]

    valid = np.isfinite(a) & (a != 0.0)
    a_filled = np.where(valid, a, 0.0)
    cnt = valid.sum(axis=1, keepdims=True).clip(min=1)
    mu = a_filled.sum(axis=1, keepdims=True) / cnt
    var = np.where(valid, (a - mu) ** 2, 0.0).sum(axis=1, keepdims=True) / cnt
    sd = np.sqrt(var) + 1e-6
    x = np.where(valid, (a - mu) / sd, 0.0).astype(np.float32)
    m = valid.astype(np.float32)

    if one_d:
        return x[0], m[0]
    return x, m


__all__ = ["valid_mask", "prepare_spectrum_input"]
