### Resample MOPRA spectra onto the training velocity axis (e.g. smoothed -> native grid).
from __future__ import annotations

from pathlib import Path

import numpy as np

from spectackle.data.mopra_preprocess import MOPRA_BLANK_VALUE


def velocity_axis_kms(cube_path: Path) -> np.ndarray:
    """LSR km/s axis from a FITS cube (authoritative via spectral_cube)."""
    from spectral_cube import SpectralCube

    cube = SpectralCube.read(str(cube_path.resolve()), use_dask=False)
    return cube.spectral_axis.to("km/s").value.astype(np.float64)


def _ensure_increasing(v: np.ndarray, spec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    v = np.asarray(v, dtype=np.float64)
    if v.size >= 2 and v[1] < v[0]:
        return v[::-1].copy(), spec[..., ::-1].copy()
    return v, spec


def resample_spec_batch(
    spec: np.ndarray,
    v_src: np.ndarray,
    v_tgt: np.ndarray,
    *,
    blank_value: float = MOPRA_BLANK_VALUE,
) -> np.ndarray:
    """
    Linear velocity resampling for a batch of spectra.

    spec: (B, n_src) or (n_src,). Channels outside the src velocity span are
    set to blank_value so preprocess masks them out.
    """
    one_d = spec.ndim == 1
    if one_d:
        spec = spec[None, :]

    v_src, spec = _ensure_increasing(v_src, np.asarray(spec, dtype=np.float64))
    v_tgt = np.asarray(v_tgt, dtype=np.float64)
    out = np.full((spec.shape[0], v_tgt.size), blank_value, dtype=np.float64)

    vmin, vmax = float(v_src.min()), float(v_src.max())
    in_range = (v_tgt >= vmin) & (v_tgt <= vmax)
    if not np.any(in_range):
        return out[0] if one_d else out

    vi = v_tgt[in_range]
    for i in range(spec.shape[0]):
        row = spec[i]
        ok = np.isfinite(row) & (row != blank_value) & (row != 0.0)
        if ok.sum() < 2:
            continue
        vs = v_src[ok]
        ys = row[ok]
        order = np.argsort(vs)
        out[i, in_range] = np.interp(vi, vs[order], ys[order], left=blank_value, right=blank_value)

    return out[0] if one_d else out


__all__ = ["velocity_axis_kms", "resample_spec_batch"]
