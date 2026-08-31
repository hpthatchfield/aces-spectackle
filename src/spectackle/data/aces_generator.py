### ACES HNCO-axis synthetic spectra for Scheme B/C counting baselines.
### Noisy, masked, SNR-drawn peaks on the native ACES velocity grid (or a narrowed window).
from __future__ import annotations

from copy import deepcopy

import numpy as np

from spectackle.config import deep_update

from .generator import DEFAULT_GEN, channel_width_kms, generate_spectrum
from .mopra_scouse_accept import apply_glance_visible_label
from .scheme_d_dataset import cfg_velocity_window

### Match ACES HNCO header in data/aces_hnco_header.txt:
### NAXIS3=1400, CDELT3=0.20818593 km/s, CRPIX3=700, CRVAL3=0 km/s
ACES_VRANGE = (-145.52196507, 145.730151)
ACES_N_CHANNELS = 1400

ACES_GEN_DEFAULT = dict(
    k_mode="biased_low",
    k_low_max=10,
    k_low_weights=None,  ### filled per Kmax in build_aces_synth_cfg
    k_tail_prob=0.08,
    p_zero=0.10,
    ### SNR-drawn peaks (scale-invariant after per-spectrum valid-channel norm).
    amp_mode="snr",
    snr_range=(3.0, 20.0),
    noise_sigma=1.0,
    width_mode="fwhm",
    fwhm_min_kms=3.0,
    fwhm_max_kms=45.0,
    min_component_separation=1.5,
    min_sep_channels=4.0,
    min_amp_ratio=0.30,
    blend_cluster_prob=0.12,
    cluster_width_range=(6.0, 28.0),
    baseline_poly_prob=0.50,
    baseline_max_slope=0.03,
    baseline_max_quad=0.0004,
    min_peak_height_factor=3.0,
    ### ALMA mosaic padding (valid island + NaN moat + zero pad).
    mask_prob=1.0,
    valid_frac_range=(0.6, 0.75),
    nan_moat_frac_range=(0.3, 0.9),
)

### MOPRA simple physics on ACES axis: glance/resolvable labels, free placement, ALMA mask.
### glance_min_sep_kms stays in km/s (~19 ch at dv~0.208). For heatmap: label_sigma~1 km/s.
ACES_GEN_SIMPLE_GLANCE = dict(
    width_mode="fwhm",
    fwhm_min_kms=1.5,
    fwhm_max_kms=60.0,
    amp_mode="snr",
    snr_range=(3.0, 20.0),
    snr_sample="uniform",
    noise_sigma=1.0,
    k_mode="uniform",
    p_zero=0.0,
    min_component_separation=None,
    min_sep_channels=None,
    min_amp_ratio=None,
    blend_cluster_prob=0.0,
    baseline_poly_prob=0.40,
    baseline_max_slope=0.04,
    baseline_max_quad=0.0005,
    min_peak_height_factor=3.0,
    mask_prob=1.0,
    valid_frac_range=(0.6, 0.75),
    nan_moat_frac_range=(0.3, 0.9),
    glance_label_k=True,
    glance_snr_tol=3.0,
    glance_prominence_sigma=3.0,
    glance_prominence_mode="adaptive",
    glance_peak_frac=0.15,
    glance_min_sep_kms=4.0,
    glance_cap_mode="resolvable",
)

### SNR prune only (no resolvable-peak cap): keep all amp/sigma>=3 centers as heatmap targets.
### Soft draw-time separation: min_sep_channels~12 -> ~2.5 km/s at dv~0.208, plus
### 0.5 * (sigma_i+sigma_j) so wider lines stay farther apart (also applied inside blend clusters).
### blend_cluster_prob=0.5: half of K>=2 draws are velocity-clustered (std 3-10 km/s)
### so shoulders / partial doubles are sampled well; the rest stay island-uniform.
### FWHM lognormal: median exp(1.39)~4 km/s (narrow-biased); clip to [1.5, 60].
ACES_GEN_SIMPLE_SNR = dict(
    width_mode="fwhm",
    fwhm_min_kms=1.5,
    fwhm_max_kms=60.0,
    fwhm_sample="lognormal",
    fwhm_lognorm_mu=1.39,  ### ln(4) ~ median FWHM 4 km/s
    fwhm_lognorm_sigma=0.70,
    amp_mode="snr",
    snr_range=(3.0, 20.0),
    snr_sample="uniform",
    noise_sigma=1.0,
    k_mode="uniform",
    p_zero=0.0,
    min_component_separation=0.5,
    min_sep_channels=12.0,  ### ~2.5 km/s floor at native ACES dv
    min_amp_ratio=None,
    blend_cluster_prob=0.50,
    cluster_width_range=(3.0, 10.0),
    baseline_poly_prob=0.40,
    baseline_max_slope=0.04,
    baseline_max_quad=0.0005,
    min_peak_height_factor=3.0,
    mask_prob=1.0,
    valid_frac_range=(0.6, 0.75),
    nan_moat_frac_range=(0.3, 0.9),
    glance_label_k=True,
    glance_snr_tol=3.0,
    glance_cap_mode="none",  ### no resolvable bump-cap
)

ACES_GEN_PRESETS = {
    "default": None,  ### use ACES_GEN_DEFAULT (+ biased_low weights)
    "simple_glance": ACES_GEN_SIMPLE_GLANCE,
    "simple_snr": ACES_GEN_SIMPLE_SNR,
}

ACES_BASE_CFG = dict(
    n_channels=ACES_N_CHANNELS,
    min_components=0,
    max_components=10,
    vrange=ACES_VRANGE,
    gen=deepcopy(ACES_GEN_DEFAULT),
)


def _biased_low_weights(kmax: int) -> tuple[float, ...]:
    ### Decreasing weights over K=1..kmax (generator requires len == k_low_max).
    raw = np.linspace(1.0, 0.25, int(kmax), dtype=np.float64)
    raw /= raw.sum()
    return tuple(float(x) for x in raw)


def build_aces_synth_cfg(
    *,
    Kmax: int = 10,
    v_center_kms: float = 0.0,
    v_half_width_kms: float | None = 80.0,
    n_channels: int | None = None,
    full_axis: bool = False,
    gen_preset: str = "default",
    gen_overrides: dict | None = None,
) -> dict:
    """
    Build a Scheme B/C / heatmap synthetic cfg on the ACES HNCO axis.

    Default: +/-80 km/s window at native dv (~770 channels) for practical training time.
    Set full_axis=True or pass n_channels / v_half_width_kms to override the span.

    gen_preset:
      - "default": biased_low K prior (Scheme B/C)
      - "simple_glance": MOPRA-simple physics + resolvable-glance labels + ALMA mask
      - "simple_snr": same physics + SNR prune only (no resolvable cap) + soft min sep
    """
    if gen_preset not in ACES_GEN_PRESETS:
        raise ValueError(f"Unknown gen_preset {gen_preset!r}; use one of {sorted(ACES_GEN_PRESETS)}")

    cfg = deepcopy(ACES_BASE_CFG)
    cfg["max_components"] = int(Kmax)
    preset = ACES_GEN_PRESETS[gen_preset]
    if preset is None:
        gen = deepcopy(ACES_GEN_DEFAULT)
        gen["k_low_max"] = int(Kmax)
        gen["k_low_weights"] = _biased_low_weights(int(Kmax))
    else:
        gen = deepcopy(ACES_GEN_DEFAULT)
        gen.update(deepcopy(preset))
        ### uniform / glance presets do not use biased_low weights.
        if gen.get("k_mode") == "biased_low":
            gen["k_low_max"] = int(Kmax)
            gen["k_low_weights"] = _biased_low_weights(int(Kmax))
    if gen_overrides:
        gen.update(gen_overrides)
    cfg["gen"] = gen

    if full_axis:
        return cfg
    if n_channels is not None:
        return cfg_velocity_window(cfg, n_channels=int(n_channels), v_center_kms=v_center_kms)
    if v_half_width_kms is not None:
        return cfg_velocity_window(
            cfg, v_half_width_kms=float(v_half_width_kms), v_center_kms=v_center_kms
        )
    return cfg


def generate_aces_spectrum(cfg: dict, rng: np.random.Generator, v_axis=None) -> dict:
    """
    Synthetic spectrum on the ACES HNCO axis (or narrowed window in cfg).

    Applies glance_visible labeling when gen.glance_label_k is set; otherwise returns
    drawn-component K (Scheme B/C default).
    """
    merged = deepcopy(cfg)
    merged["gen"] = deep_update(deepcopy(DEFAULT_GEN), cfg.get("gen", {}))
    gen = merged["gen"]
    ex = generate_spectrum(merged, rng, v_axis=v_axis)
    if gen.get("glance_label_k"):
        ex = apply_glance_visible_label(ex, merged)
    return ex


__all__ = [
    "ACES_BASE_CFG",
    "ACES_GEN_DEFAULT",
    "ACES_GEN_PRESETS",
    "ACES_GEN_SIMPLE_GLANCE",
    "ACES_GEN_SIMPLE_SNR",
    "ACES_N_CHANNELS",
    "ACES_VRANGE",
    "build_aces_synth_cfg",
    "channel_width_kms",
    "generate_aces_spectrum",
]
