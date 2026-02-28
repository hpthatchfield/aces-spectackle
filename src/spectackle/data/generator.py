### Synthetic spectrum generator
from copy import deepcopy

import numpy as np

from spectackle.config import deep_update

DEFAULT_GEN = dict(
    k_mode="poisson",
    p_zero=0.0,
    k_mean=3.0,
    k_tail_prob=0.50,
    k_tail_min=6,
    amp_lognorm_mu=0.0,
    amp_lognorm_sigma=1.0,
    sigma_min=0.1,
    sigma_max=10.0,
    blend_cluster_prob=0.0,
    cluster_width_range=(1.0, 30.0),
    noise_std_range=(0.02, 0.15),
    baseline_poly_prob=0.5,
    baseline_max_slope=0.02,
    baseline_max_quad=0.0002,
)


def _make_v_axis(cfg: dict) -> np.ndarray:
    vmin, vmax = cfg["vrange"]
    C = int(cfg["n_channels"])
    return np.linspace(vmin, vmax, C, dtype=np.float32)


def generate_spectrum(cfg: dict, rng: np.random.Generator, v_axis=None) -> dict:
    """
    Returns dict with stable keys used across notebooks:
      spec       : noisy spectrum, (C,)
      spec_clean : pure Gaussian spectrum (no baseline/noise), (C,)
      k          : int, number of sampled components
      noise_std  : (1,)
      v_axis     : (C,)
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
        if gen.get("k_mode", "poisson") == "uniform":
            k = int(rng.integers(0, Kmax + 1))
            if k > 0:
                k = max(k, min_components)
        else:
            if rng.random() < gen["k_tail_prob"]:
                k = int(rng.integers(gen["k_tail_min"], k_tail_max + 1))
            else:
                k = max(1, int(rng.poisson(gen["k_mean"])))
            k = max(k, min_components) if k > 0 else 0
    k = min(k, Kmax)

    A = np.zeros(Kmax, dtype=np.float32)
    mu = np.zeros(Kmax, dtype=np.float32)
    sig = np.ones(Kmax, dtype=np.float32)

    if k > 0:
        if rng.random() < gen["blend_cluster_prob"] and k >= 2:
            center = rng.uniform(vmin + 0.2 * (vmax - vmin), vmax - 0.2 * (vmax - vmin))
            cw = rng.uniform(*gen["cluster_width_range"])
            mus = center + rng.normal(0.0, cw, size=k)
            mus = np.clip(mus, vmin, vmax)
        else:
            mus = rng.uniform(vmin, vmax, size=k)
        sigs = rng.uniform(gen["sigma_min"], gen["sigma_max"], size=k)
        amps = rng.lognormal(mean=gen["amp_lognorm_mu"], sigma=gen["amp_lognorm_sigma"], size=k)
        amps = amps / (np.percentile(amps, 90) + 1e-6)
        order = np.argsort(mus)
        mus, sigs, amps = mus[order], sigs[order], amps[order]
        A[:k] = amps.astype(np.float32)
        mu[:k] = mus.astype(np.float32)
        sig[:k] = sigs.astype(np.float32)

    spec = np.zeros(C, dtype=np.float32)
    for i in range(k):
        dv = (v - mu[i]) / (sig[i] + 1e-6)
        spec += A[i] * np.exp(-0.5 * dv * dv).astype(np.float32)
    spec_clean = spec.copy()

    if rng.random() < gen["baseline_poly_prob"]:
        x = np.linspace(-1, 1, C, dtype=np.float32)
        slope = rng.uniform(-gen["baseline_max_slope"], gen["baseline_max_slope"])
        quad = rng.uniform(-gen["baseline_max_quad"], gen["baseline_max_quad"])
        spec += (slope * x + quad * (x**2)).astype(np.float32)

    noise_std = float(rng.uniform(*gen["noise_std_range"]))
    spec += rng.normal(0.0, noise_std, size=C).astype(np.float32)

    return dict(
        spec=spec,
        spec_clean=spec_clean,
        k=k,
        noise_std=np.array([noise_std], dtype=np.float32),
        v_axis=v,
    )
