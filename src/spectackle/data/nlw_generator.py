### Narrow line width (NLW) synthetic spectra - sibling to generator.py
### Labels: binary presence of >=1 NLW Gaussian (narrow sigma) vs wide-only / empty backgrounds.
from __future__ import annotations

import warnings
from copy import deepcopy

import numpy as np

from spectackle.config import deep_update

from .axis_mask import apply_axis_mask, draw_valid_island
from .generator import (
    FWHM_TO_SIGMA_KMS,
    _make_v_axis,
    channel_width_kms,
    fwhm_kms_to_sigma_kms,
)


NLW_GEN_DEFAULT = dict(
    nlw_prob=0.5,
    ### Ultra-narrow NLW: FWHM 0.7-1.0 km/s (ACES dv~0.21 km/s -> ~3.4-4.8 channels FWHM).
    ### Upper cap matches the science case (genuinely narrow lines); lower bound stays a few channels
    ### wide so features remain resolved (not single-channel spikes) and above the effective resolution.
    nlw_fwhm_min_kms=0.7,
    nlw_fwhm_max_kms=1.0,
    nlw_count_max=2,
    ### Wide background must stay clearly broader than nlw_fwhm_max (validator enforces sigma gap).
    bg_fwhm_min_kms=3.0,
    bg_fwhm_max_kms=50.0,
    bg_count_max=5,
    p_zero_bg=0.10,
    ### Narrow (NLW) centroid mu uniform in central fraction of the valid island (wide mu full island).
    nlw_v_inner_fraction=0.9,
    ### --- Brightness in SNR units. Absolute scale is irrelevant (model input is per-spectrum
    ### normalized), so we fix the noise sigma and draw each component peak directly as peak/sigma = SNR.
    ### Narrow features are bright by construction (the scientific target); broad spans low->high SNR.
    ### Floor lowered to 5 so positives span a wider brightness range - this decouples "bright" from
    ### the label and forces the model to key on line *width*, not just amplitude.
    noise_sigma=1.0,
    nlw_snr_range=(5.0, 30.0),
    bg_snr_range=(2.0, 25.0),
    ### --- Optional smooth continuum (survives per-spectrum normalization, so it is meaningful).
    baseline_poly_prob=0.5,
    baseline_max_slope=0.02,
    baseline_max_quad=0.0002,
    ### --- Spectral-axis masking (ALMA mosaic padding): a contiguous valid island at a random
    ### position, flanked by NaN moats then exact-zero pads. Defaults bracket the region-1 test
    ### cube (valid fraction ~ 0.683). Set mask_prob=0.0 (or valid_frac_range=(1,1)) to disable.
    mask_prob=1.0,
    valid_frac_range=(0.6, 0.75),
    nan_moat_frac_range=(0.3, 0.9),
)


NLW_BASE_CFG = dict(
    ### Match ACES HNCO header in data/aces_hnco_header.txt:
    ### NAXIS3=1400, CDELT3=0.20818593 km/s, CRPIX3=700, CRVAL3=0 km/s
    ### => vmin=(1-700)*0.20818593=-145.52196507 km/s, vmax=(1400-700)*0.20818593=145.730151 km/s
    n_channels=1400,
    vrange=(-145.52196507, 145.730151),
    nlw_gen=deepcopy(NLW_GEN_DEFAULT),
)


def nlw_cfg_velocity_window(
    *,
    n_channels: int | None = None,
    v_half_width_kms: float | None = None,
    v_center_kms: float = 0.0,
) -> dict:
    """
    One fixed velocity "window": same per-channel dv as `NLW_BASE_CFG` (ACES HNCO spacing),
    but fewer channels => narrower (vmin, vmax). Analogous to a single frame of a sliding window
    over the full cube axis - no regridding, only a contiguous span at native resolution.

    Specify exactly one of `n_channels` or `v_half_width_kms` (half-width of [center-h, center+h]
    after snapping span to an integer number of channels).
    """
    if (n_channels is None) == (v_half_width_kms is None):
        raise ValueError("Specify exactly one of n_channels or v_half_width_kms (not both, not neither).")
    ref = NLW_BASE_CFG
    cw = channel_width_kms(ref)
    out = deepcopy(ref)
    if n_channels is not None:
        n = int(n_channels)
        if n < 2:
            raise ValueError("n_channels must be >= 2")
        span = cw * float(n - 1)
    else:
        half = float(v_half_width_kms)
        if half <= 0.0:
            raise ValueError("v_half_width_kms must be positive")
        span_target = 2.0 * half
        n = max(2, int(round(span_target / cw)) + 1)
        span = cw * float(n - 1)
    vc = float(v_center_kms)
    vmin = vc - 0.5 * span
    vmax = vc + 0.5 * span
    out["n_channels"] = n
    out["vrange"] = (vmin, vmax)
    return out


def build_nlw_base_cfg(
    *,
    nlw_n_channels: int | None = None,
    nlw_v_half_kms: float | None = None,
    nlw_v_center_kms: float = 0.0,
) -> dict:
    ### Thin wrapper for scripts: default full-axis cfg, or a narrowed window (see nlw_cfg_velocity_window).
    if nlw_n_channels is not None and nlw_v_half_kms is not None:
        raise ValueError("Pass at most one of nlw_n_channels or nlw_v_half_kms.")
    if nlw_n_channels is None and nlw_v_half_kms is None:
        return deepcopy(NLW_BASE_CFG)
    if nlw_n_channels is not None:
        return nlw_cfg_velocity_window(
            n_channels=nlw_n_channels,
            v_center_kms=nlw_v_center_kms,
        )
    return nlw_cfg_velocity_window(
        v_half_width_kms=nlw_v_half_kms,
        v_center_kms=nlw_v_center_kms,
    )


def _validate_nlw_gen(gen: dict, cfg: dict) -> None:
    ### Ensure wide sigma strictly exceeds narrow sigma so labels are not ambiguous at the sigma level.
    sigma_nlw_max = fwhm_kms_to_sigma_kms(gen["nlw_fwhm_max_kms"])
    sigma_bg_min = fwhm_kms_to_sigma_kms(gen["bg_fwhm_min_kms"])
    if sigma_bg_min <= sigma_nlw_max:
        raise ValueError(
            "NLW generator: background sigma min must exceed narrow sigma max. "
            f"Got bg_fwhm_min_kms -> sigma={sigma_bg_min:.4g} km/s, "
            f"nlw_fwhm_max_kms -> sigma={sigma_nlw_max:.4g} km/s. "
            "Increase bg_fwhm_min_kms or decrease nlw_fwhm_max_kms."
        )
    if gen["nlw_fwhm_min_kms"] > gen["nlw_fwhm_max_kms"]:
        raise ValueError("nlw_fwhm_min_kms must be <= nlw_fwhm_max_kms")
    if gen["bg_fwhm_min_kms"] > gen["bg_fwhm_max_kms"]:
        raise ValueError("bg_fwhm_min_kms must be <= bg_fwhm_max_kms")
    cw = channel_width_kms(cfg)
    if gen["nlw_fwhm_min_kms"] < cw:
        warnings.warn(
            f"nlw_fwhm_min_kms ({gen['nlw_fwhm_min_kms']}) is below one channel "
            f"({cw:.4g} km/s); narrow features may be poorly sampled.",
            stacklevel=2,
        )
    frac = gen.get("nlw_v_inner_fraction")
    if frac is not None:
        f = float(frac)
        if not (0.0 < f <= 1.0):
            raise ValueError(f"nlw_v_inner_fraction must be in (0, 1], got {frac}")

    ### Brightness (SNR) knobs.
    if float(gen["noise_sigma"]) <= 0.0:
        raise ValueError(f"noise_sigma must be > 0, got {gen['noise_sigma']}")
    _validate_positive_range(gen["nlw_snr_range"], "nlw_snr_range")
    _validate_positive_range(gen["bg_snr_range"], "bg_snr_range")

    ### Spectral-axis masking knobs.
    mp = float(gen.get("mask_prob", 1.0))
    if not (0.0 <= mp <= 1.0):
        raise ValueError(f"mask_prob must be in [0, 1], got {mp}")
    _validate_fraction_range(gen["valid_frac_range"], "valid_frac_range", allow_zero=False)
    _validate_fraction_range(gen["nan_moat_frac_range"], "nan_moat_frac_range", allow_zero=True)


def _validate_positive_range(rng_val, name: str) -> None:
    ### (lo, hi) with 0 < lo <= hi.
    lo, hi = float(rng_val[0]), float(rng_val[1])
    if not (0.0 < lo <= hi):
        raise ValueError(f"{name} must be (lo, hi) with 0 < lo <= hi, got {rng_val}")


def _validate_fraction_range(rng_val, name: str, *, allow_zero: bool) -> None:
    ### (lo, hi) with 0(<)= lo <= hi <= 1.
    lo, hi = float(rng_val[0]), float(rng_val[1])
    lo_ok = (0.0 <= lo) if allow_zero else (0.0 < lo)
    if not (lo_ok and lo <= hi <= 1.0):
        bound = "0 <=" if allow_zero else "0 <"
        raise ValueError(f"{name} must be (lo, hi) with {bound} lo <= hi <= 1, got {rng_val}")


def _nlw_narrow_mu_bounds(vmin: float, vmax: float, inner_fraction: float) -> tuple[float, float]:
    ### Central fraction of [vmin, vmax] for narrow Gaussian centers (e.g. 0.9 -> inner 90%).
    span = float(vmax) - float(vmin)
    if span <= 0.0:
        raise ValueError(f"vrange must have positive span, got ({vmin}, {vmax})")
    f = float(inner_fraction)
    if f >= 1.0:
        return float(vmin), float(vmax)
    margin = 0.5 * span * (1.0 - f)
    return float(vmin) + margin, float(vmax) - margin


def generate_nlw_spectrum(cfg: dict, rng: np.random.Generator, v_axis=None) -> dict:
    """
    Synthetic pixel spectrum with optional narrow (NLW) Gaussian components.

    User-facing widths are **FWHM in km/s** (same units as cfg[\"vrange\"]). Component peaks are
    drawn directly as SNR (peak/noise_sigma): narrow from `nlw_snr_range`, wide from `bg_snr_range`.
    All components live inside a contiguous **valid island** (see draw_valid_island); channels
    outside it are NaN moat then exact-zero pad, mirroring ALMA mosaic spectral padding. Narrow
    centers mu are uniform in the central `nlw_v_inner_fraction` of the island; wide mu span the island.

    Returns dict keys:
      spec, spec_clean : (C,) float32 - noisy vs Gaussian-only; both carry NaN/0 pads outside island
      has_nlw          : bool
      n_nlw, n_bg      : int counts of injected narrow vs wide Gaussians
      narrowest_sigma_kms : float (nan if no Gaussian component)
      peak_clean       : max(spec_clean) over the island (~ peak SNR since noise_sigma sets the scale)
      nlw_peak_clean   : max narrow-only clean peak over the island (~ narrow SNR; 0 if no NLW)
      noise_std        : (1,) float32 - the noise_sigma used
      v_axis           : (C,) float32
      component_sigma_kms : (k_total,) float32 (wide components first, then narrow)
      component_is_narrow : (k_total,) bool
      component_v_kms       : (k_total,) float32 - Gaussian center velocity (km/s), same order
    """
    gen = deep_update(deepcopy(NLW_GEN_DEFAULT), cfg.get("nlw_gen", {}))
    _validate_nlw_gen(gen, cfg)

    c = int(cfg["n_channels"])
    v = v_axis if v_axis is not None else _make_v_axis(cfg)

    sigma_nlw_lo = fwhm_kms_to_sigma_kms(gen["nlw_fwhm_min_kms"])
    sigma_nlw_hi = fwhm_kms_to_sigma_kms(gen["nlw_fwhm_max_kms"])
    sigma_bg_lo = fwhm_kms_to_sigma_kms(gen["bg_fwhm_min_kms"])
    sigma_bg_hi = fwhm_kms_to_sigma_kms(gen["bg_fwhm_max_kms"])

    has_nlw = bool(rng.random() < float(gen["nlw_prob"]))
    k_narrow = int(rng.integers(1, int(gen["nlw_count_max"]) + 1)) if has_nlw else 0
    k_wide = 0 if rng.random() < float(gen["p_zero_bg"]) else int(rng.integers(1, int(gen["bg_count_max"]) + 1))

    ### Valid island first: all components and noise live inside [i0, i1).
    i0, i1, nan_left, nan_right = draw_valid_island(c, gen, rng)
    v_isl_lo = float(min(v[i0], v[i1 - 1]))
    v_isl_hi = float(max(v[i0], v[i1 - 1]))

    mus_wide = rng.uniform(v_isl_lo, v_isl_hi, size=k_wide).astype(np.float32) if k_wide > 0 else np.zeros(0, dtype=np.float32)
    sigs_wide = (
        rng.uniform(sigma_bg_lo, sigma_bg_hi, size=k_wide).astype(np.float32) if k_wide > 0 else np.zeros(0, dtype=np.float32)
    )
    if k_narrow > 0:
        nlw_frac = gen.get("nlw_v_inner_fraction")
        if nlw_frac is None:
            mu_n_lo, mu_n_hi = v_isl_lo, v_isl_hi
        else:
            mu_n_lo, mu_n_hi = _nlw_narrow_mu_bounds(v_isl_lo, v_isl_hi, float(nlw_frac))
        mus_narrow = rng.uniform(mu_n_lo, mu_n_hi, size=k_narrow).astype(np.float32)
    else:
        mus_narrow = np.zeros(0, dtype=np.float32)
    sigs_narrow = (
        rng.uniform(sigma_nlw_lo, sigma_nlw_hi, size=k_narrow).astype(np.float32)
        if k_narrow > 0
        else np.zeros(0, dtype=np.float32)
    )

    mus = np.concatenate([mus_wide, mus_narrow])
    sigs = np.concatenate([sigs_wide, sigs_narrow])
    k_total = int(mus.shape[0])
    is_narrow = np.concatenate([np.zeros(k_wide, dtype=bool), np.ones(k_narrow, dtype=bool)])

    ### Peaks drawn directly in SNR units (peak = SNR * noise_sigma).
    noise_sigma = float(gen["noise_sigma"])
    amps_wide = noise_sigma * rng.uniform(*gen["bg_snr_range"], size=k_wide) if k_wide > 0 else np.zeros(0)
    amps_narrow = noise_sigma * rng.uniform(*gen["nlw_snr_range"], size=k_narrow) if k_narrow > 0 else np.zeros(0)
    amps = np.concatenate([amps_wide, amps_narrow]).astype(np.float32)

    spec_clean = np.zeros(c, dtype=np.float32)
    spec_nlw_clean = np.zeros(c, dtype=np.float32)
    for i in range(k_total):
        dv = (v - mus[i]) / (sigs[i] + 1e-6)
        g = (amps[i] * np.exp(-0.5 * dv * dv)).astype(np.float32)
        spec_clean += g
        if bool(is_narrow[i]):
            spec_nlw_clean += g

    baseline_term = np.zeros(c, dtype=np.float32)
    if rng.random() < float(gen["baseline_poly_prob"]):
        x = np.linspace(-1, 1, c, dtype=np.float32)
        slope = rng.uniform(-gen["baseline_max_slope"], gen["baseline_max_slope"])
        quad = rng.uniform(-gen["baseline_max_quad"], gen["baseline_max_quad"])
        baseline_term = (slope * x + quad * (x**2)).astype(np.float32)

    noise = rng.normal(0.0, noise_sigma, size=c).astype(np.float32)
    spec = (spec_clean + baseline_term + noise).astype(np.float32)

    ### Diagnostics measured on the island (before masking pads to NaN/0).
    narrowest = float(np.min(sigs)) if k_total > 0 else float("nan")
    peak_clean = float(spec_clean.max()) if k_total > 0 else 0.0
    nlw_peak_clean = float(spec_nlw_clean.max()) if k_narrow > 0 else 0.0

    ### Apply ALMA-style padding: NaN moat then exact-zero pad outside the valid island.
    spec = apply_axis_mask(spec, i0, i1, nan_left, nan_right)
    spec_clean = apply_axis_mask(spec_clean, i0, i1, nan_left, nan_right)

    return dict(
        spec=spec,
        spec_clean=spec_clean,
        has_nlw=has_nlw,
        n_nlw=k_narrow,
        n_bg=k_wide,
        narrowest_sigma_kms=narrowest,
        peak_clean=peak_clean,
        nlw_peak_clean=nlw_peak_clean,
        noise_std=np.array([noise_sigma], dtype=np.float32),
        v_axis=v,
        component_sigma_kms=sigs.astype(np.float32),
        component_is_narrow=is_narrow,
        component_v_kms=mus.astype(np.float32),
    )


__all__ = [
    "FWHM_TO_SIGMA_KMS",
    "NLW_BASE_CFG",
    "NLW_GEN_DEFAULT",
    "channel_width_kms",
    "fwhm_kms_to_sigma_kms",
    "generate_nlw_spectrum",
]
