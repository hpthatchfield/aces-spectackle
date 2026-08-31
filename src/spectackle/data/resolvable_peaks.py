### Resolvable-peak counting and blend statistics for generator calibration diagnostics.
from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks

from spectackle.data.generator import _make_v_axis, channel_width_kms
from spectackle.data.mopra_preprocess import MOPRA_BLANK_VALUE, valid_mask_mopra
from spectackle.data.scouse_saa import estimate_spectrum_rms

_FWHM_TO_SIGMA = 2.0 * np.sqrt(2.0 * np.log(2.0))  ### ~ 2.355


def fwhm_kms_to_sigma(fwhm_kms: np.ndarray | float) -> np.ndarray:
    return np.asarray(fwhm_kms, dtype=np.float64) / _FWHM_TO_SIGMA


def count_resolvable_peaks(
    spec: np.ndarray,
    vel_kms: np.ndarray | None = None,
    *,
    blank_value: float | None = MOPRA_BLANK_VALUE,
    vel_range: tuple[float, float] | None = (40.0, 140.0),
    prominence_sigma: float = 3.0,
    min_sep_kms: float = 4.0,
    prominence_mode: str = "fixed_sigma",
    peak_frac: float = 0.25,
) -> tuple[int, float]:
    """
    Prominence-based peak count in a velocity window (proxy for by-eye resolvable bumps).

    prominence_mode:
      - "fixed_sigma": height/prominence = prominence_sigma * sigma_rms (default)
      - "adaptive": height/prominence = max(prominence_sigma * sigma_rms, peak_frac * peak_amp)
        (matches earlier MOPRA generator-gap diagnostic)

    Returns (n_peaks, sigma_rms). Uses Scouse-style robust RMS on baseline-subtracted spectrum.
    """
    spec = np.asarray(spec, dtype=np.float64).reshape(-1)
    vel = np.asarray(vel_kms, dtype=np.float64).reshape(-1) if vel_kms is not None else None
    if vel is not None and vel.size != spec.size:
        raise ValueError(f"spec length {spec.size} != vel length {vel.size}")

    valid = np.isfinite(spec)
    if blank_value is not None:
        valid &= valid_mask_mopra(spec, blank_value=blank_value)
    if vel is not None and vel_range is not None:
        vlo, vhi = float(vel_range[0]), float(vel_range[1])
        valid &= (vel >= vlo) & (vel <= vhi)
    if not np.any(valid):
        return 0, float("nan")

    y_full = np.where(valid, spec, np.nan)
    med = float(np.nanmedian(y_full))
    y = y_full[valid] - med
    sigma = float(estimate_spectrum_rms(y))
    if not np.isfinite(sigma) or sigma <= 0.0:
        sigma = float(np.nanstd(y) + 1e-12)

    if vel is not None:
        dv = float(np.median(np.diff(vel[valid]))) if valid.sum() > 1 else 2.0
    else:
        dv = 2.0
    min_dist = max(1, int(round(float(min_sep_kms) / max(abs(dv), 1e-6))))

    peak_amp = float(np.nanmax(y)) if y.size else 0.0
    if prominence_mode == "adaptive":
        height = max(float(prominence_sigma) * sigma, float(peak_frac) * peak_amp)
    elif prominence_mode == "fixed_sigma":
        height = float(prominence_sigma) * sigma
    else:
        raise ValueError(f'prominence_mode must be "fixed_sigma" or "adaptive", got {prominence_mode!r}')

    peaks, _ = find_peaks(y, height=height, prominence=height, distance=min_dist)
    return int(peaks.size), sigma


def blend_stats_from_components(
    v_kms: np.ndarray,
    amps: np.ndarray,
    fwhm_kms: np.ndarray,
) -> dict[str, float]:
    """
  Pairwise blend stats for K>=2 (sorted by v).

    Returns min_sep_sigma, min_amp_ratio, median_sep_sigma; NaNs if K<2.
    """
    v = np.asarray(v_kms, dtype=np.float64).reshape(-1)
    a = np.asarray(amps, dtype=np.float64).reshape(-1)
    w = np.asarray(fwhm_kms, dtype=np.float64).reshape(-1)
    k = int(min(v.size, a.size, w.size))
    if k < 2:
        return {
            "min_sep_sigma": float("nan"),
            "median_sep_sigma": float("nan"),
            "min_amp_ratio": float("nan"),
        }
    v, a, w = v[:k], a[:k], w[:k]
    sig = fwhm_kms_to_sigma(w)
    order = np.argsort(v)
    v, a, sig = v[order], a[order], sig[order]

    sep_sig = []
    for i in range(k - 1):
        sep_kms = abs(v[i + 1] - v[i])
        sep_sig.append(sep_kms / max(sig[i] + sig[i + 1], 1e-6))
    amax = float(np.max(a))
    amp_ratio = float(np.min(a) / amax) if amax > 0 else float("nan")
    sep_arr = np.asarray(sep_sig, dtype=np.float64)
    return {
        "min_sep_sigma": float(np.min(sep_arr)),
        "median_sep_sigma": float(np.median(sep_arr)),
        "min_amp_ratio": amp_ratio,
    }


def blend_stats_from_dat_rows(rows: list[np.ndarray]) -> dict[str, float]:
    """
    Henshaw .dat rows for one (l,b): col5=v (km/s), col6=amp, col8=linewidth FWHM (km/s).

    col8 reconstructs the observed narrow-line spectrum; col7 is a broader metadata width.
    """
    if len(rows) < 2:
        return blend_stats_from_components(np.zeros(0), np.zeros(0), np.zeros(0))
    v = np.array([float(r[5]) for r in rows], dtype=np.float64)
    a = np.array([float(r[6]) for r in rows], dtype=np.float64)
    w = np.array([float(r[8]) for r in rows], dtype=np.float64)
    return blend_stats_from_components(v, a, w)


def blend_stats_from_synth_example(ex: dict) -> dict[str, float]:
    """Synthetic generator dict after scouse acceptance (component_* arrays, ex['k'])."""
    k = int(ex["k"])
    if k < 2:
        return blend_stats_from_components(np.zeros(0), np.zeros(0), np.zeros(0))
    v = ex["component_v_kms"][:k]
    a = ex["component_amp"][:k]
    ### component_sigma is always Gaussian sigma (km/s), even when width_mode=fwhm at draw time.
    sig = ex["component_sigma"][:k]
    w = sig * _FWHM_TO_SIGMA
    return blend_stats_from_components(v, a, w)


def sample_synthetic_by_k(
    cfg: dict,
    *,
    n_per_k: int = 400,
    k_values: tuple[int, ...] = (1, 2, 3, 4),
    seed: int = 0,
    max_attempts_factor: int = 200,
) -> list[dict]:
    """
    Draw scouse_dat-like spectra until we have n_per_k examples per target K label.
    """
    from spectackle.data.mopra_generator import generate_mopra_spectrum

    v_axis = _make_v_axis(cfg)
    rng = np.random.default_rng(seed)
    buckets: dict[int, list[dict]] = {int(k): [] for k in k_values}
    max_attempts = int(n_per_k) * int(max_attempts_factor) * len(k_values)
    for _ in range(max_attempts):
        if all(len(buckets[k]) >= n_per_k for k in k_values):
            break
        ex = generate_mopra_spectrum(cfg, rng, v_axis=v_axis)
        k = int(ex["k"])
        if k in buckets and len(buckets[k]) < n_per_k:
            buckets[k].append(ex)
    out: list[dict] = []
    for k in k_values:
        out.extend(buckets[int(k)])
    return out


__all__ = [
    "blend_stats_from_components",
    "blend_stats_from_dat_rows",
    "blend_stats_from_synth_example",
    "channel_width_kms",
    "count_resolvable_peaks",
    "fwhm_kms_to_sigma",
    "sample_synthetic_by_k",
]
