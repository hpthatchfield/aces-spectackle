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
    ### "uniform" | "log_uniform" over cluster_width_range (km/s std of mu around a shared center).
    cluster_width_sample="uniform",
    ### If True, multiply cluster std by (k/2) so nearest-neighbor spacing does not
    ### collapse when more components share the same neighborhood.
    cluster_width_scale_with_k=False,
    ### k_mode="islands": K is an outcome of n_islands x a short extra-component chain.
    ### n_island_weights[i] = P(n_islands = i+1); unused unless k_mode="islands".
    n_island_weights=(0.70, 0.22, 0.06, 0.02),
    p_secondary=0.45,
    p_tertiary=0.30,
    p_chain_more=0.15,
    p_chain_decay=0.65,
    chain_sep_sigma_range=(0.50, 1.60),
    island_min_sep_kms=12.0,
    noise_std_range=(0.02, 0.15),
    # If set, enforce that the *clean* peak height is at least
    # `min_peak_height_factor * noise_std` for non-empty spectra (k>0).
    # This effectively guarantees a minimum per-spectrum SNR.
    min_peak_height_factor=None,
    # If set (lo, hi), after forming the Gaussian stack (before baseline), scale spec so
    # max(spec_clean) ~ Uniform(lo, hi). Use to fix peak scale vs noise_std (e.g. for experimental SNR sweeps).
    peak_scale_range=None,
    baseline_poly_prob=0.5,
    baseline_max_slope=0.02,
    baseline_max_quad=0.0002,
    ### ACES data incompleteness re axis masking (mask_prob=0.0 disables; mirrors real-cube padding).
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


def _draw_cluster_width_kms(gen: dict, rng: np.random.Generator) -> float:
    """Std (km/s) of component centers around a shared cluster mean."""
    cw_lo, cw_hi = float(gen["cluster_width_range"][0]), float(gen["cluster_width_range"][1])
    if not (0.0 < cw_lo <= cw_hi):
        raise ValueError(f"gen.cluster_width_range must satisfy 0 < lo <= hi, got {gen['cluster_width_range']}")
    sample = str(gen.get("cluster_width_sample", "uniform"))
    if sample == "uniform":
        return float(rng.uniform(cw_lo, cw_hi))
    if sample == "log_uniform":
        return float(np.exp(rng.uniform(np.log(cw_lo), np.log(cw_hi))))
    raise ValueError(f'gen.cluster_width_sample must be "uniform" or "log_uniform", got {sample!r}')


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


def _draw_chain_size(gen: dict, rng: np.random.Generator, slots_left: int) -> int:
    """Primary plus a decaying extra-component chain, capped by remaining Kmax slots."""
    slots = int(slots_left)
    if slots <= 1:
        return max(0, slots)
    n = 1
    if rng.random() >= float(gen.get("p_secondary", 0.45)):
        return n
    n += 1
    if n >= slots:
        return n
    if rng.random() >= float(gen.get("p_tertiary", 0.30)):
        return n
    n += 1
    p = float(gen.get("p_chain_more", 0.15))
    decay = float(gen.get("p_chain_decay", 0.65))
    while n < slots and rng.random() < p:
        n += 1
        p *= decay
    return n


def _draw_n_islands(gen: dict, rng: np.random.Generator, kmax: int) -> int:
    w = np.asarray(gen.get("n_island_weights", (0.70, 0.22, 0.06, 0.02)), dtype=np.float64)
    if w.size < 1 or np.any(w < 0) or float(w.sum()) <= 0:
        raise ValueError(f"gen.n_island_weights must be positive, got {gen.get('n_island_weights')}")
    n_max = min(int(w.size), int(kmax))
    w = w[:n_max]
    w = w / w.sum()
    return int(rng.choice(np.arange(1, n_max + 1), p=w))


def _draw_primary_ranked_amps(
    gen: dict,
    n: int,
    rng: np.random.Generator,
    noise_std: float,
) -> np.ndarray:
    """Brightest first (primary), extras as fractions. No shuffle (island families)."""
    primary_snr = float(_draw_snr_values(gen, 1, rng)[0])
    primary_amp = float(noise_std) * primary_snr
    if n <= 1:
        return np.asarray([primary_amp], dtype=np.float64)
    ratios = _draw_amp_ratios(gen, n - 1, rng)
    extra_snr = primary_snr * ratios
    ### Keep extras above the SNR label floor so nearby peaks are not drawn then dropped.
    snr_lo = float(gen.get("glance_snr_tol", gen.get("snr_range", (3.0, 20.0))[0]))
    extra_snr = np.clip(extra_snr, snr_lo, primary_snr)
    amps = np.empty(n, dtype=np.float64)
    amps[0] = primary_amp
    amps[1:] = float(noise_std) * extra_snr
    return amps


def _place_family_mus(
    mu_primary: float,
    sigs: np.ndarray,
    rng: np.random.Generator,
    v_lo: float,
    v_hi: float,
    sep_range: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extras sit near the primary. 2nd extra prefers the opposite side.

    Never clip onto the island edge (that stacked two centers on one channel).
    Off-island candidates flip to the interior side. Extras that still cannot
    sit (too close to an existing center, or offset larger than the island)
    are dropped; K is an outcome.

    Returns mus (n_kept,) and keep indices into sigs/amps.
    """
    n = int(sigs.size)
    mus = [float(mu_primary)]
    keep = [0]
    side0 = 0.0
    lo, hi = float(sep_range[0]), float(sep_range[1])
    if not (0.0 < lo <= hi):
        raise ValueError(f"gen.chain_sep_sigma_range must satisfy 0 < lo <= hi, got {sep_range}")
    sig0 = float(sigs[0])
    for j in range(1, n):
        ### ~0.5 km/s is ~2 ACES channels / one heatmap splat; also a fraction of the
        ### intended extra offset so wide lines do not land on top of the primary.
        min_dv = max(0.5, 0.25 * lo * (sig0 + float(sigs[j])))
        for _try in range(40):
            sep_sig = float(rng.uniform(lo, hi))
            if len(mus) == 2 and side0 != 0.0:
                side = -side0
            else:
                side = float(rng.choice(np.array([-1.0, 1.0])))
            dv = side * sep_sig * (sig0 + float(sigs[j]))
            cand = float(mu_primary + dv)
            if cand < v_lo or cand > v_hi:
                cand = float(mu_primary - dv)
            if not (v_lo <= cand <= v_hi):
                continue
            if any(abs(cand - m) < min_dv for m in mus):
                continue
            mus.append(cand)
            keep.append(j)
            if len(keep) == 2:
                side0 = 1.0 if cand >= mu_primary else -1.0
            break
        ### extra j dropped if still unplaced after retries
    return np.asarray(mus, dtype=np.float64), np.asarray(keep, dtype=np.int64)


def _draw_island_components(
    gen: dict,
    rng: np.random.Generator,
    *,
    kmax: int,
    v_isl_lo: float,
    v_isl_hi: float,
    noise_std: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int, list[int], float | None]:
    """
    Place 1+ velocity islands, each with a primary and a short extra chain.

    Returns mus, sigs, amps (length k), k, n_islands, island_sizes, noise_std.
    """
    if noise_std is None:
        if gen.get("noise_sigma") is not None:
            noise_std = float(gen["noise_sigma"])
        else:
            noise_std = float(rng.uniform(*gen["noise_std_range"]))

    n_target = _draw_n_islands(gen, rng, kmax)
    min_sep = gen.get("island_min_sep_kms")
    min_sep = None if min_sep is None else float(min_sep)
    sep_range = tuple(gen.get("chain_sep_sigma_range", (0.50, 1.60)))
    amp_mode = str(gen.get("amp_mode", "snr"))

    mus_all: list[float] = []
    sigs_all: list[float] = []
    amps_all: list[float] = []
    island_sizes: list[int] = []
    primaries: list[float] = []
    remaining = int(kmax)

    for _ in range(n_target):
        if remaining <= 0:
            break
        n_here = _draw_chain_size(gen, rng, remaining)
        if n_here <= 0:
            break
        sigs = _draw_component_sigmas(gen, n_here, rng)
        if amp_mode == "snr_rank":
            amps = _draw_primary_ranked_amps(gen, n_here, rng, float(noise_std))
        else:
            amps, noise_std = _draw_component_amps(gen, n_here, rng, noise_std=noise_std)
            ### Keep the brightest as primary (slot 0) for placement.
            order_amp = np.argsort(amps)[::-1]
            amps = amps[order_amp]
            sigs = sigs[order_amp]

        mu0 = float(rng.uniform(v_isl_lo, v_isl_hi))
        if min_sep is not None and primaries:
            for _try in range(80):
                if all(abs(mu0 - p) >= min_sep for p in primaries):
                    break
                mu0 = float(rng.uniform(v_isl_lo, v_isl_hi))
        mus, keep = _place_family_mus(mu0, sigs, rng, v_isl_lo, v_isl_hi, sep_range)
        sigs = np.asarray(sigs, dtype=np.float64)[keep]
        amps = np.asarray(amps, dtype=np.float64)[keep]
        n_here = int(mus.size)
        mus_all.extend(mus.tolist())
        sigs_all.extend(sigs.tolist())
        amps_all.extend(amps.tolist())
        island_sizes.append(n_here)
        primaries.append(float(mus[0]))
        remaining -= n_here

    k = int(len(mus_all))
    if k == 0:
        return (
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.float64),
            0,
            0,
            [],
            noise_std,
        )
    mus = np.asarray(mus_all, dtype=np.float64)
    sigs = np.asarray(sigs_all, dtype=np.float64)
    amps = np.asarray(amps_all, dtype=np.float64)
    order = np.argsort(mus)
    return mus[order], sigs[order], amps[order], k, len(island_sizes), island_sizes, noise_std


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
        use_islands = False
    else:
        mode = gen.get("k_mode", "poisson")
        use_islands = mode == "islands"
        if use_islands:
            k = -1  ### filled by island draw
        elif mode == "uniform":
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
    if not use_islands:
        k = min(k, Kmax)

    i0, i1, nan_left, nan_right = draw_valid_island(C, gen, rng)
    v_isl_lo = float(min(v[i0], v[i1 - 1]))
    v_isl_hi = float(max(v[i0], v[i1 - 1]))

    A = np.zeros(Kmax, dtype=np.float32)
    mu = np.zeros(Kmax, dtype=np.float32)
    sig = np.ones(Kmax, dtype=np.float32)

    noise_std_drawn: float | None = None
    n_islands = 0
    island_sizes: list[int] = []

    if use_islands:
        mus, sigs, amps, k, n_islands, island_sizes, noise_std_drawn = _draw_island_components(
            gen,
            rng,
            kmax=Kmax,
            v_isl_lo=v_isl_lo,
            v_isl_hi=v_isl_hi,
            noise_std=noise_std_drawn,
        )
        if k > 0:
            A[:k] = amps.astype(np.float32)
            mu[:k] = mus.astype(np.float32)
            sig[:k] = sigs.astype(np.float32)
    elif k > 0:
        ### Clustered mus: K>=2 components share a velocity neighborhood and are allowed
        ### to overlap. Min-sep (if set) applies only to non-cluster island draws. do not
        ### resample a tight cluster onto the full island.
        used_blend_cluster = False
        if rng.random() < gen["blend_cluster_prob"] and k >= 2:
            used_blend_cluster = True
            center = rng.uniform(v_isl_lo + 0.2 * (v_isl_hi - v_isl_lo), v_isl_hi - 0.2 * (v_isl_hi - v_isl_lo))
            cw = _draw_cluster_width_kms(gen, rng)
            if bool(gen.get("cluster_width_scale_with_k", False)):
                cw = cw * (float(k) / 2.0)
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
        sep_factor = gen.get("min_component_separation")
        min_sep_ch = gen.get("min_sep_channels")
        dv_kms = channel_width_kms(cfg) if min_sep_ch is not None else None
        if (
            (sep_factor is not None or min_sep_ch is not None)
            and k >= 2
            and not used_blend_cluster
        ):
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
                mus = rng.uniform(v_isl_lo, v_isl_hi, size=k)
                mus = np.sort(mus)
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

    ### Optional enforcment of minimum peak height vs noise.
    ### Compare to spec_clean (pure Gaussian, no baseline/noise).
    ### For k=0 we keep the existing noise-only spectra.
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

    out = dict(
        spec=spec,
        spec_clean=spec_clean,
        k=k,
        noise_std=np.array([noise_std], dtype=np.float32),
        v_axis=v,
        component_amp=A,
        component_v_kms=mu,
        component_sigma=sig,
    )
    if use_islands:
        out["n_islands"] = int(n_islands)
        out["island_sizes"] = np.asarray(island_sizes, dtype=np.int32)
    return out
