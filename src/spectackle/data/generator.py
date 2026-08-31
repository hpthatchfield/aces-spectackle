### Synthetic spectrum generator
from __future__ import annotations

import math
from copy import deepcopy

import numpy as np

from spectackle.config import deep_update

from .axis_mask import apply_axis_mask, draw_valid_island

### FWHM (km/s) <-> Gaussian sigma (km/s): FWHM = 2 * sqrt(2 * ln(2)) * sigma
FWHM_TO_SIGMA_KMS = 2.0 * math.sqrt(2.0 * math.log(2.0))

DEFAULT_GEN = dict(
    k_mode="uniform",
    p_zero=0.10,   ### explicit non-detection fraction so model sees empty spectra
    ### k_mode="biased_low": sample K in 1..k_low_max (weighted) or k_tail_prob for k_low_max+1..Kmax.
    k_low_max=5,
    k_low_weights=None,  ### if None, decreasing weights over 1..k_low_max
    min_component_separation=None,  ### factor * (sigma_i+sigma_{i+1}); also floored by min_sep_channels * dv
    min_sep_channels=None,  ### minimum peak-center separation in spectral channels (x channel_width_kms)
    k_tail_prob=0.0,
    k_mean=3.0,
    k_tail_min=6,
    amp_lognorm_mu=0.0,
    amp_lognorm_sigma=1.0,
    ### amp_mode="lognorm" (default) | "snr" | "snr_rank"
    ###   snr: each component SNR ~ independent draw in snr_range
    ###   snr_rank: one primary SNR, then secondaries as amp_ratio * primary
    amp_mode="lognorm",
    snr_range=(2.0, 25.0),
    snr_sample="uniform",  ### "uniform" | "log_uniform" (only for amp_mode="snr" / "snr_rank")
    ### snr_rank only: secondary/tertiary amp as fraction of the primary.
    amp_ratio_range=(0.12, 0.95),
    amp_ratio_sample="log_uniform",  ### "uniform" | "log_uniform"
    noise_sigma=None,  ### if set with amp_mode="snr", fixes noise scale for amp + noise draw
    # If set (0..1], enforce min(amp)/max(amp) >= min_amp_ratio for multi-peak spectra (k>=2).
    # This prevents "nearly single-peak" mixtures that are labeled as k=2+ but are not resolvable.
    min_amp_ratio=None,
    ### width_mode="sigma" (default) | "fwhm" - component widths in sigma or FWHM (km/s).
    width_mode="sigma",
    sigma_min=0.1,
    sigma_max=10.0,
    fwhm_min_kms=None,
    fwhm_max_kms=None,
    ### fwhm_sample: "uniform" (default) | "lognormal" (clip to fwhm_min/max).
    ### lognormal uses fwhm_lognorm_mu / fwhm_lognorm_sigma on ln(FWHM).
    fwhm_sample="uniform",
    fwhm_lognorm_mu=3.17,  ### ~ ln(23.7); Henshaw HNCO median FWHM ~23 km/s
    fwhm_lognorm_sigma=0.44,
    blend_cluster_prob=0.0,
    cluster_width_range=(1.0, 30.0),
    noise_std_range=(0.02, 0.15),
    # If set, enforce that the *clean* peak height is at least
    # `min_peak_height_factor * noise_std` for non-empty spectra (k>0).
    # This effectively guarantees a minimum per-spectrum SNR.
    min_peak_height_factor=None,
    # If set (lo, hi), after forming the Gaussian stack (before baseline), scale spec so
    # max(spec_clean) ~ Uniform(lo, hi). Use to fix peak scale vs noise_std (e.g. SNR sweeps).
    peak_scale_range=None,
    baseline_poly_prob=0.5,
    baseline_max_slope=0.02,
    baseline_max_quad=0.0002,
    ### ALMA-style axis masking (mask_prob=0.0 disables; mirrors real-cube padding).
    mask_prob=0.0,
    valid_frac_range=(0.6, 0.75),
    nan_moat_frac_range=(0.3, 0.9),
)


def fwhm_kms_to_sigma_kms(fwhm_kms: float) -> float:
    return float(fwhm_kms) / FWHM_TO_SIGMA_KMS


def channel_width_kms(cfg: dict) -> float:
    vmin, vmax = cfg["vrange"]
    c = int(cfg["n_channels"])
    ### Match _make_v_axis (np.linspace inclusive endpoints): dv ~ (vmax-vmin)/(C-1)
    return float(vmax - vmin) / max(c - 1, 1)


def _make_v_axis(cfg: dict) -> np.ndarray:
    vmin, vmax = cfg["vrange"]
    C = int(cfg["n_channels"])
    return np.linspace(vmin, vmax, C, dtype=np.float32)


def _draw_component_sigmas(gen: dict, k: int, rng: np.random.Generator) -> np.ndarray:
    mode = gen.get("width_mode", "sigma")
    if mode == "sigma":
        return rng.uniform(gen["sigma_min"], gen["sigma_max"], size=k)
    if mode == "fwhm":
        fmin = gen.get("fwhm_min_kms")
        fmax = gen.get("fwhm_max_kms")
        if fmin is None or fmax is None:
            raise ValueError('width_mode="fwhm" requires gen.fwhm_min_kms and gen.fwhm_max_kms')
        if float(fmin) > float(fmax):
            raise ValueError(f"fwhm_min_kms must be <= fwhm_max_kms, got ({fmin}, {fmax})")
        sample = str(gen.get("fwhm_sample", "uniform"))
        if sample == "uniform":
            fwhm = rng.uniform(float(fmin), float(fmax), size=k)
        elif sample == "lognormal":
            mu = float(gen.get("fwhm_lognorm_mu", 3.17))
            sig = float(gen.get("fwhm_lognorm_sigma", 0.44))
            if sig <= 0:
                raise ValueError(f"gen.fwhm_lognorm_sigma must be > 0, got {sig}")
            fwhm = rng.lognormal(mean=mu, sigma=sig, size=k)
            fwhm = np.clip(fwhm, float(fmin), float(fmax))
        else:
            raise ValueError(f'gen.fwhm_sample must be "uniform" or "lognormal", got {sample!r}')
        return fwhm / FWHM_TO_SIGMA_KMS
    raise ValueError(f'gen.width_mode must be "sigma" or "fwhm", got {mode!r}')


def _draw_snr_values(gen: dict, n: int, rng: np.random.Generator) -> np.ndarray:
    snr_lo, snr_hi = float(gen["snr_range"][0]), float(gen["snr_range"][1])
    if not (0.0 < snr_lo <= snr_hi):
        raise ValueError(f"gen.snr_range must satisfy 0 < lo <= hi, got {gen['snr_range']}")
    sample = str(gen.get("snr_sample", "uniform"))
    if sample == "uniform":
        return rng.uniform(snr_lo, snr_hi, size=n)
    if sample == "log_uniform":
        return np.exp(rng.uniform(np.log(snr_lo), np.log(snr_hi), size=n))
    raise ValueError(f'gen.snr_sample must be "uniform" or "log_uniform", got {sample!r}')


def _draw_amp_ratios(gen: dict, n: int, rng: np.random.Generator) -> np.ndarray:
    """Secondary/tertiary amplitude as a fraction of the primary (snr_rank mode)."""
    if n <= 0:
        return np.zeros(0, dtype=np.float64)
    r_lo, r_hi = float(gen.get("amp_ratio_range", (0.12, 0.95))[0]), float(
        gen.get("amp_ratio_range", (0.12, 0.95))[1]
    )
    if not (0.0 < r_lo <= r_hi <= 1.0):
        raise ValueError(f"gen.amp_ratio_range must satisfy 0 < lo <= hi <= 1, got {(r_lo, r_hi)}")
    sample = str(gen.get("amp_ratio_sample", "log_uniform"))
    if sample == "uniform":
        return rng.uniform(r_lo, r_hi, size=n)
    if sample == "log_uniform":
        return np.exp(rng.uniform(np.log(r_lo), np.log(r_hi), size=n))
    raise ValueError(f'gen.amp_ratio_sample must be "uniform" or "log_uniform", got {sample!r}')


def _draw_component_amps(
    gen: dict,
    k: int,
    rng: np.random.Generator,
    *,
    noise_std: float | None,
) -> tuple[np.ndarray, float | None]:
    mode = gen.get("amp_mode", "lognorm")
    if mode == "lognorm":
        amps = rng.lognormal(mean=gen["amp_lognorm_mu"], sigma=gen["amp_lognorm_sigma"], size=k)
        amps = amps / (np.percentile(amps, 90) + 1e-6)
        return amps, noise_std
    if mode in ("snr", "snr_rank"):
        if noise_std is None:
            if gen.get("noise_sigma") is not None:
                noise_std = float(gen["noise_sigma"])
            else:
                noise_std = float(rng.uniform(*gen["noise_std_range"]))
        if mode == "snr":
            snrs = _draw_snr_values(gen, k, rng)
            amps = noise_std * snrs
            return amps, noise_std
        ### snr_rank: one primary, then weaker companions (matches Scouse amp hierarchy).
        primary_snr = float(_draw_snr_values(gen, 1, rng)[0])
        primary_amp = float(noise_std) * primary_snr
        if k == 1:
            return np.asarray([primary_amp], dtype=np.float64), noise_std
        ratios = _draw_amp_ratios(gen, k - 1, rng)
        amps = np.empty(k, dtype=np.float64)
        amps[0] = primary_amp
        amps[1:] = primary_amp * ratios
        ### Shuffle so the brightest component is not tied to draw order before mu-sort.
        rng.shuffle(amps)
        return amps, noise_std
    raise ValueError(f'gen.amp_mode must be "lognorm", "snr", or "snr_rank", got {mode!r}')


def generate_spectrum(cfg: dict, rng: np.random.Generator, v_axis=None) -> dict:
    """
    Returns dict with stable keys used across notebooks:
      spec              : noisy spectrum, (C,)
      spec_clean        : pure Gaussian spectrum (no baseline/noise), (C,)
      k                 : int, number of sampled components
      noise_std         : (1,)
      v_axis            : (C,)
      component_amp     : (Kmax,) float32 - amplitudes; unused slots are 0
      component_v_kms   : (Kmax,) float32 - centers (km/s); unused slots are 0
      component_sigma   : (Kmax,) float32 - Gaussian sigma (km/s); unused slots are 0
    """
    vmin, vmax = cfg["vrange"]
    C = int(cfg["n_channels"])
    Kmax = int(cfg["max_components"])
    min_components = int(cfg.get("min_components", 0))
    gen = deep_update(DEFAULT_GEN, cfg.get("gen", {}))
    k_tail_max = min(10, Kmax)
    v = v_axis if v_axis is not None else _make_v_axis(cfg)

    if rng.random() < gen["p_zero"]:
        k = 0
    else:
        mode = gen.get("k_mode", "poisson")
        if mode == "uniform":
            k = int(rng.integers(0, Kmax + 1))
            if k > 0:
                k = max(k, min_components)
        elif mode == "biased_low":
            k_low_max = min(int(gen.get("k_low_max", 5)), Kmax)
            p_tail = float(gen.get("k_tail_prob", 0.12))
            if p_tail > 0.0 and k_low_max < Kmax and rng.random() < p_tail:
                k = int(rng.integers(k_low_max + 1, Kmax + 1))
            else:
                weights = gen.get("k_low_weights")
                if weights is None:
                    default_w = [0.35, 0.30, 0.18, 0.12, 0.05]
                    weights = default_w[:k_low_max]
                w = np.asarray(weights, dtype=np.float64)
                if w.size != k_low_max:
                    raise ValueError(
                        f"gen.k_low_weights length {w.size} must match k_low_max={k_low_max}"
                    )
                w = w / w.sum()
                k = int(rng.choice(np.arange(1, k_low_max + 1), p=w))
            k = max(k, min_components)
        else:
            if rng.random() < gen["k_tail_prob"]:
                k = int(rng.integers(gen["k_tail_min"], k_tail_max + 1))
            else:
                k = max(1, int(rng.poisson(gen["k_mean"])))
            k = max(k, min_components) if k > 0 else 0
    k = min(k, Kmax)

    i0, i1, nan_left, nan_right = draw_valid_island(C, gen, rng)
    v_isl_lo = float(min(v[i0], v[i1 - 1]))
    v_isl_hi = float(max(v[i0], v[i1 - 1]))

    A = np.zeros(Kmax, dtype=np.float32)
    mu = np.zeros(Kmax, dtype=np.float32)
    sig = np.ones(Kmax, dtype=np.float32)

    noise_std_drawn: float | None = None

    if k > 0:
        ### Blend-cluster draws keep components in a velocity neighborhood (partial overlaps).
        ### Min-sep redraws below resample cluster mus around the same center, or else
        ### redraw non-cluster mus across the full island.
        used_blend_cluster = False
        center = 0.0
        cw = 1.0
        if rng.random() < gen["blend_cluster_prob"] and k >= 2:
            used_blend_cluster = True
            center = rng.uniform(v_isl_lo + 0.2 * (v_isl_hi - v_isl_lo), v_isl_hi - 0.2 * (v_isl_hi - v_isl_lo))
            cw = rng.uniform(*gen["cluster_width_range"])
            mus = center + rng.normal(0.0, cw, size=k)
            mus = np.clip(mus, v_isl_lo, v_isl_hi)
        else:
            mus = rng.uniform(v_isl_lo, v_isl_hi, size=k)
        sigs = _draw_component_sigmas(gen, k, rng)
        amps, noise_std_drawn = _draw_component_amps(gen, k, rng, noise_std=noise_std_drawn)
        ### Optional: enforce amplitude ratio for multi-peak spectra (k>=2).
        min_amp_ratio = gen.get("min_amp_ratio")
        if min_amp_ratio is not None and k >= 2:
            r = float(min_amp_ratio)
            if not (0.0 < r <= 1.0):
                raise ValueError(f"gen.min_amp_ratio must be in (0,1], got {r}")
            # Resample amps only (keep mus/sigs); bounded attempts to avoid infinite loops.
            for _ in range(200):
                amax = float(np.max(amps))
                amin = float(np.min(amps))
                if amax > 0 and (amin / amax) >= r:
                    break
                amps, noise_std_drawn = _draw_component_amps(gen, k, rng, noise_std=noise_std_drawn)
        order = np.argsort(mus)
        mus, sigs, amps = mus[order], sigs[order], amps[order]
        ### Optional: enforce spatial separation between adjacent peaks (sorted mu).
        ### Separation is max(factor * (sigma_i+sigma_{i+1}), min_sep_channels * dv).
        ### Cluster draws resample around the same center/width; non-cluster redraws
        ### mus across the full island.
        sep_factor = gen.get("min_component_separation")
        min_sep_ch = gen.get("min_sep_channels")
        dv_kms = channel_width_kms(cfg) if min_sep_ch is not None else None
        if (sep_factor is not None or min_sep_ch is not None) and k >= 2:
            def _min_sep_ok(mus_sorted: np.ndarray) -> bool:
                for i in range(k - 1):
                    min_sep = 0.0
                    if sep_factor is not None:
                        min_sep = max(min_sep, float(sep_factor) * (sigs[i] + sigs[i + 1]))
                    if min_sep_ch is not None:
                        min_sep = max(min_sep, float(min_sep_ch) * dv_kms)
                    if abs(mus_sorted[i + 1] - mus_sorted[i]) < min_sep:
                        return False
                return True

            for _ in range(200):
                if _min_sep_ok(mus):
                    break
                if used_blend_cluster:
                    mus = center + rng.normal(0.0, cw, size=k)
                    mus = np.clip(mus, v_isl_lo, v_isl_hi)
                else:
                    mus = rng.uniform(v_isl_lo, v_isl_hi, size=k)
                mus = np.sort(mus)
            ### Cluster can be too tight for K>=3 + floor; fall back to island draw.
            if not _min_sep_ok(mus):
                for _ in range(200):
                    mus = rng.uniform(v_isl_lo, v_isl_hi, size=k)
                    mus = np.sort(mus)
                    if _min_sep_ok(mus):
                        break
        A[:k] = amps.astype(np.float32)
        mu[:k] = mus.astype(np.float32)
        sig[:k] = sigs.astype(np.float32)

    spec = np.zeros(C, dtype=np.float32)
    for i in range(k):
        dv = (v - mu[i]) / (sig[i] + 1e-6)
        spec += A[i] * np.exp(-0.5 * dv * dv).astype(np.float32)
    spec_clean = spec.copy()

    psr = gen.get("peak_scale_range")
    if psr is not None and k > 0:
        lo, hi = float(psr[0]), float(psr[1])
        if not (0.0 < lo <= hi):
            raise ValueError(f"gen.peak_scale_range must be (lo, hi) with 0 < lo <= hi, got {psr}")
        m = float(spec_clean.max()) + 1e-12
        target = float(rng.uniform(lo, hi))
        s = target / m
        spec_clean = (spec_clean * s).astype(np.float32)
        spec = spec_clean.copy()
        A[:k] = (A[:k] * s).astype(np.float32)

    baseline_term = np.zeros(C, dtype=np.float32)
    if rng.random() < gen["baseline_poly_prob"]:
        x = np.linspace(-1, 1, C, dtype=np.float32)
        slope = rng.uniform(-gen["baseline_max_slope"], gen["baseline_max_slope"])
        quad = rng.uniform(-gen["baseline_max_quad"], gen["baseline_max_quad"])
        baseline_term = (slope * x + quad * (x**2)).astype(np.float32)
        spec += baseline_term

    ### Optional: enforce minimum peak height vs noise.
    ### We compare to spec_clean (pure Gaussian, no baseline/noise).
    ### For k=0 we keep the existing behavior (noise-only spectra).
    if noise_std_drawn is None:
        if gen.get("noise_sigma") is not None:
            noise_std_drawn = float(gen["noise_sigma"])
        else:
            noise_std_drawn = float(rng.uniform(*gen["noise_std_range"]))
    noise_std = float(noise_std_drawn)
    min_peak_factor = gen.get("min_peak_height_factor")
    if min_peak_factor is not None and k > 0:
        peak = float(spec_clean.max())
        target = float(min_peak_factor) * noise_std
        if peak < target:
            # Small epsilon to avoid division-by-zero while keeping target tight.
            scale = target / (peak + 1e-12)
            spec_clean = (spec_clean * scale).astype(np.float32)
            # Reconstruct spec: scaled Gaussian part + unscaled baseline.
            spec = (spec_clean + baseline_term).astype(np.float32)
            A[:k] = (A[:k] * scale).astype(np.float32)

    spec += rng.normal(0.0, noise_std, size=C).astype(np.float32)

    ### Mask after noise so padded channels stay exact NaN / 0.0 (see preprocess.valid_mask).
    spec = apply_axis_mask(spec, i0, i1, nan_left, nan_right)
    spec_clean = apply_axis_mask(spec_clean, i0, i1, nan_left, nan_right)

    return dict(
        spec=spec,
        spec_clean=spec_clean,
        k=k,
        noise_std=np.array([noise_std], dtype=np.float32),
        v_axis=v,
        component_amp=A,
        component_v_kms=mu,
        component_sigma=sig,
    )
