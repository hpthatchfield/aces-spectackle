### MOPRA CMZ synthetic spectra: tuned for Jones 2012 single-dish HNCO axis + noise.
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np

from spectackle.config import deep_update

from .generator import DEFAULT_GEN, _make_v_axis, generate_spectrum
from .mopra_header import build_mopra_base_cfg
from .mopra_preprocess import valid_mask_mopra
from .mopra_scouse_accept import (
    apply_glance_visible_label,
    apply_scouse_label_filter,
    apply_snr_component_label,
)
from .scouse_saa import estimate_spectrum_rms

### Legacy generator (pre smooth60 / Scouse-SNR calibration).
MOPRA_GEN_LEGACY = dict(
    width_mode="fwhm",
    fwhm_min_kms=3.5,
    fwhm_max_kms=45.0,
    amp_mode="snr",
    snr_range=(2.5, 18.0),
    noise_sigma=0.08,
    noise_std_range=(0.04, 0.14),
    k_mode="biased_low",
    k_low_max=5,
    k_low_weights=(0.40, 0.30, 0.15, 0.10, 0.05),
    k_tail_prob=0.10,
    p_zero=0.08,
    min_component_separation=2.5,
    min_sep_channels=3.0,
    min_amp_ratio=0.25,
    blend_cluster_prob=0.08,
    cluster_width_range=(8.0, 35.0),
    baseline_poly_prob=0.65,
    baseline_max_slope=0.06,
    baseline_max_quad=0.0008,
    min_peak_height_factor=2.0,
    mask_prob=0.0,
    ripple_prob=0.35,
    ripple_amp_range=(0.02, 0.12),
    ripple_period_channels_range=(40.0, 180.0),
    spike_prob=0.12,
    spike_amp_range=(0.15, 0.8),
)

MOPRA_CUBE_SMOOTH60 = "CMZ_3mm_HNCO_60.fits"

### Scouse/Henshaw .dat K distribution (5224 labeled pixels, col0 ncomps).
SCOUSE_DAT_K_WEIGHTS = (0.541, 0.321, 0.122, 0.015)

### Default: sigma_rms calibrated to CMZ_3mm_HNCO_60.fits + Scouse-style component rules.
### Measured smooth60 sky sigma_rms (ScousePy calc_rms): med~0.021, p10~0.014, p90~0.032 K.
MOPRA_GEN_DEFAULT = dict(
    width_mode="fwhm",
    fwhm_min_kms=3.5,
    fwhm_max_kms=45.0,
    amp_mode="snr",
    ### Floor matches CMZ_SCOUSE_TOL SNR=3; upper tail trimmed vs legacy 18.
    snr_range=(3.0, 15.0),
    noise_sigma=None,
    noise_std_range=(0.012, 0.032),
    k_mode="biased_low",
    k_low_max=5,
    k_low_weights=(0.40, 0.30, 0.15, 0.10, 0.05),
    k_tail_prob=0.06,
    p_zero=0.10,
    ### Stricter separation / secondary amplitude (Scouse deblend semantics).
    min_component_separation=3.0,
    min_sep_channels=5.0,
    min_amp_ratio=0.40,
    blend_cluster_prob=0.04,
    cluster_width_range=(8.0, 30.0),
    baseline_poly_prob=0.55,
    baseline_max_slope=0.04,
    baseline_max_quad=0.0005,
    min_peak_height_factor=3.0,
    mask_prob=0.0,
    ### Reduced artifacts: spikes/ripples were inflating K on real CMZ spectra.
    ripple_prob=0.12,
    ripple_amp_range=(0.008, 0.04),
    ripple_period_channels_range=(40.0, 180.0),
    spike_prob=0.04,
    spike_amp_range=(0.06, 0.20),
)

### Scouse .dat aligned: smooth60 axis, empirical K=1..4, post-draw Scouse acceptance as label.
MOPRA_GEN_SCOUSE_DAT = dict(
    width_mode="fwhm",
    fwhm_min_kms=3.5,
    fwhm_max_kms=45.0,
    amp_mode="snr",
    snr_range=(3.0, 15.0),
    noise_sigma=None,
    noise_std_range=(0.012, 0.032),
    k_mode="biased_low",
    k_low_max=4,
    k_low_weights=SCOUSE_DAT_K_WEIGHTS,
    k_tail_prob=0.0,
    p_zero=0.0,
    min_component_separation=3.0,
    min_sep_channels=5.0,
    min_amp_ratio=0.40,
    blend_cluster_prob=0.04,
    cluster_width_range=(8.0, 30.0),
    baseline_poly_prob=0.55,
    baseline_max_slope=0.04,
    baseline_max_quad=0.0005,
    min_peak_height_factor=3.0,
    mask_prob=0.0,
    ripple_prob=0.12,
    ripple_amp_range=(0.008, 0.04),
    ripple_period_channels_range=(40.0, 180.0),
    spike_prob=0.04,
    spike_amp_range=(0.06, 0.20),
    ### Label K = Scouse-accepted component count (SNR + deblend), not raw draw count.
    scouse_label_k=True,
    scouse_snr_tol=3.0,
)


### FAILED ATTEMPT (kept as a documented negative result; do not use as the default).
### Tried to reduce the bright-center under-count vs Henshaw on the non-fine-tuned model by
### up-weighting K=3/4 and loosening deblend/amp. Outcome: Scouse MAE 0.78 -> 1.26.
### Why it failed:
###   1. The deblend/amp loosening was inert: synthetic K=3/4 keep ~2.84/3.75 resolvable
###      peaks either way (the min-separation floor rarely binds for wide-axis draws), so
###      only the K prior actually changed.
###   2. The model is effectively a resolvable-peak counter. Real crowded cores show ~0.8
###      resolvable peaks vs ~3 in synthetic K=3, so raising the count prior only inflates
###      low-information (faint/edge/K=1) pixels while bright cores follow the single-peak
###      evidence and still under-count.
### See experiments/MOPRA_Count/README.md (Findings) for the full analysis.
MOPRA_GEN_SCOUSE_DAT_RELAXED = deepcopy(MOPRA_GEN_SCOUSE_DAT)
MOPRA_GEN_SCOUSE_DAT_RELAXED.update(
    min_component_separation=2.5,
    min_sep_channels=3.0,
    min_amp_ratio=0.25,
    k_low_weights=(0.42, 0.30, 0.18, 0.10),
)

### Calibrated blends from resolvable_peak_gap diagnostic (Jul 2026).
### Goal: match real CMZ morphology where Henshaw K=3 looks like ~1.3 prominence peaks, not ~3.
### Changes vs scouse_dat:
###   - Always draw multi-component spectra in a velocity cluster (not uniform on the wide axis).
###   - Allow overlapping components (no min_sep floor; the old floor rarely bound but destroyed clusters on resample).
###   - Weaker secondaries: min_amp_ratio ~ real p10 (0.12 vs 0.40).
### Tuned on noisy spec: K=3 mean n_resolvable ~1.32, 97% under-count (real: 1.35, 97%).
### FAILED on cube: Scouse MAE 0.78 -> 1.77; K=1 over-count ~93%. Too ambiguous / too weak.
MOPRA_GEN_SCOUSE_DAT_CALIBRATED = deepcopy(MOPRA_GEN_SCOUSE_DAT)
MOPRA_GEN_SCOUSE_DAT_CALIBRATED.update(
    blend_cluster_prob=1.0,
    cluster_width_range=(6.0, 25.0),
    min_component_separation=None,
    min_sep_channels=None,
    min_amp_ratio=0.12,
)

### Blend saturation (Jul 2026): middle ground between scouse_dat (too separated/strong) and
### calibrated (too ambiguous/weak). Target glance-visible blends:
###   tall primary + clear shoulder / partial double / obvious bump (amp ratio ~0.18+),
###   without flooding faint/edge K=1 with noise-like secondaries.
### Design:
###   - Usually draw K>=2 in a tight velocity cluster so overlaps/shoulders are common.
###   - Soft amp floor 0.18 (real mode is ~0.1; we stay above invisible bumps).
###   - Keep scouse_dat K prior (do not up-weight K=3/4; that caused edge over-count).
###   - Hard min-sep still applies to non-cluster draws only (see generator.py).
###   - Label deblend: keep SNR>=3 filter, but disable separation merge for labels
###     (scouse_min_sep_* = None). Otherwise accept proportional to (sigma_i+sigma_j) strips clustered K=3
###     and sample_by_k only retains wide-axis survivors (the old failure mode).
###   - cluster_width is the std (km/s) of mu placement around a shared center.
MOPRA_GEN_SCOUSE_DAT_BLEND_SAT = deepcopy(MOPRA_GEN_SCOUSE_DAT)
MOPRA_GEN_SCOUSE_DAT_BLEND_SAT.update(
    blend_cluster_prob=0.80,
    cluster_width_range=(3.0, 10.0),
    min_amp_ratio=0.18,
    min_component_separation=2.0,
    min_sep_channels=4.0,
    scouse_min_sep_factor=None,
    scouse_min_sep_channels=None,
)

### Simple free-sampling baseline (Jul 2026), MOPRA-oriented.
### Independent component placement on the full velocity window (no blend_cluster,
### no min-sep, no Scouse sep-merge). Overlaps happen only when draws land close.
### Fiducial dials match the successful simple_k6 run (Scouse MAE ~0.51):
###   uniform SNR 3-20, FWHM 1.5-60 km/s, glance/resolvable-peak labels, Kmax=6.
MOPRA_GEN_SIMPLE = dict(
    width_mode="fwhm",
    fwhm_min_kms=1.5,
    fwhm_max_kms=60.0,
    amp_mode="snr",
    snr_range=(3.0, 20.0),
    snr_sample="uniform",
    noise_sigma=None,
    noise_std_range=(0.012, 0.032),
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
    mask_prob=0.0,
    ripple_prob=0.10,
    ripple_amp_range=(0.008, 0.04),
    ripple_period_channels_range=(40.0, 180.0),
    spike_prob=0.03,
    spike_amp_range=(0.06, 0.20),
    scouse_label_k=False,
    glance_label_k=True,
    glance_snr_tol=3.0,
    glance_prominence_sigma=3.0,
    glance_prominence_mode="adaptive",
    glance_peak_frac=0.15,
    glance_min_sep_kms=4.0,
    glance_cap_mode="resolvable",
)

### simple morphology + residual-based glance cap (Jul 2026).
### Same draws as simple; SNR>=3 floor unchanged. Replaces peak-finder bump-cap with
### iterative single-Gaussian subtract + positive integrated residual flux vs noise,
### so unresolved-but-real secondaries can still raise K_label without a 2nd local max.
MOPRA_GEN_SIMPLE_RESIDUAL = deepcopy(MOPRA_GEN_SIMPLE)
MOPRA_GEN_SIMPLE_RESIDUAL.update(
    glance_cap_mode="residual",
    glance_residual_integ_snr=5.0,
    glance_residual_window_kms=30.0,
)

### simple morphology + matched-filter glance credit (Jul 2026).
### Same draws as simple. SNR>=3 floor + glance_min_sep_kms=4 hard merge kept.
### Replaces peak-finder bump-cap with matched-filter SNR on true (A, W) after
### conceptually removing other survivors. Synth-label only (needs ground truth).
MOPRA_GEN_SIMPLE_MATCHED = deepcopy(MOPRA_GEN_SIMPLE)
MOPRA_GEN_SIMPLE_MATCHED.update(
    glance_cap_mode="matched",
    glance_matched_snr_tol=3.0,
    glance_min_sep_kms=4.0,
)

### simple + clustered blends (Jul 2026).
### Keep the simple free-window prior, then mix in blend clusters so training sees
### primary+shoulder / secondary / tertiary morphology that cores actually show.
### Differences vs simple (only these dials change):
###   - blend_cluster_prob=0.50: half of K>=2 draws are velocity-clustered
###   - cluster_width 3-12 km/s: shoulders and partial doubles, not full-axis isolates
###   - min_amp_ratio=0.20: weaker secondaries, still above the invisible 0.12 that failed
###   - mild biased_low K: fewer isolated K=4..6; still a high-K tail
### Deliberately not: always-cluster, amp floor 0.12, or heavy K=3/4 up-weight
### (those presets over-counted edges; see scouse_dat_calibrated / relaxed).
MOPRA_GEN_SIMPLE_MIX = deepcopy(MOPRA_GEN_SIMPLE)
MOPRA_GEN_SIMPLE_MIX.update(
    blend_cluster_prob=0.50,
    cluster_width_range=(3.0, 12.0),
    min_amp_ratio=0.20,
    k_mode="biased_low",
    k_low_max=4,
    k_low_weights=(0.32, 0.30, 0.24, 0.14),
    k_tail_prob=0.10,
    ### Keep some noise-only spectra (biased_low never draws K=0).
    p_zero=0.08,
)

### simple + ranked amplitudes matched to Henshaw/Scouse (Jul 2026).
### Diagnosed gap vs real CMZ: i.i.d. SNR Uniform(3,20) never exceeds SNR~20
### (real primary SNR p50~17, p90~71) and secondaries are too equal
### (real secondary/primary p10~0.14; simple_mix floored at 0.20).
### Changes vs simple:
###   - amp_mode=snr_rank: one primary SNR, companions = ratio * primary
###   - primary SNR log-uniform 4-100 (covers real p10-p99)
###   - amp ratios uniform 0.14-0.95 (real secondary/primary p10/p50/p90 ~ 0.14/0.46/0.86;
###     log-uniform undershot the median when K>=3 takes min of several ratios)
###   - FWHM lognormal matched to Henshaw (mu_ln~3.17, sigma_ln~0.44; clip 10-60 km/s)
###   - Blend clusters on most multi-K draws (p=0.75, width 6-28 km/s) so mu-spans
###     land near real p10/p50/p90 ~ 10/44/76 km/s (free-window survivors keep diversity)
### Keep glance labels for a clean A/B vs simple.
MOPRA_GEN_SIMPLE_REALAMP = deepcopy(MOPRA_GEN_SIMPLE)
MOPRA_GEN_SIMPLE_REALAMP.update(
    amp_mode="snr_rank",
    snr_range=(4.0, 100.0),
    snr_sample="log_uniform",
    amp_ratio_range=(0.14, 0.95),
    amp_ratio_sample="uniform",
    min_amp_ratio=None,
    fwhm_min_kms=10.0,
    fwhm_max_kms=60.0,
    fwhm_sample="lognormal",
    fwhm_lognorm_mu=3.17,
    fwhm_lognorm_sigma=0.44,
    blend_cluster_prob=0.75,
    cluster_width_range=(6.0, 28.0),
)

### Same morphology as simple_realamp, but K_label = drawn component count.
### Glance was collapsing ~59% of multi-K blends to label<=1 (taught "blend -> K=1").
### No Scouse sep-merge either: raw draw K is the training target.
MOPRA_GEN_SIMPLE_REALAMP_RAWK = deepcopy(MOPRA_GEN_SIMPLE_REALAMP)
MOPRA_GEN_SIMPLE_REALAMP_RAWK.update(
    glance_label_k=False,
    scouse_label_k=False,
)

### simple_realamp morphology + SNR-pass component count (no resolvable-peak discount).
### K = n components with amp/sigma >= 3; close blends count. Scouse-like K prior (biased_low).
MOPRA_GEN_SIMPLE_REALAMP_SNRK = deepcopy(MOPRA_GEN_SIMPLE_REALAMP)
MOPRA_GEN_SIMPLE_REALAMP_SNRK.update(
    glance_label_k=False,
    scouse_label_k=False,
    snr_label_k=True,
    snr_label_tol=3.0,
    k_mode="biased_low",
    k_low_max=4,
    k_low_weights=SCOUSE_DAT_K_WEIGHTS,
    k_tail_prob=0.02,
    p_zero=0.05,
)

### Heatmap benchmark: realamp morph + clusters, planted centers (no glance/snr/scouse
### label surgery), Scouse-like K prior. Heatmap targets use all drawn component centers.
MOPRA_GEN_HEATMAP_REALAMP = deepcopy(MOPRA_GEN_SIMPLE_REALAMP_RAWK)
MOPRA_GEN_HEATMAP_REALAMP.update(
    k_mode="biased_low",
    k_low_max=4,
    k_low_weights=SCOUSE_DAT_K_WEIGHTS,
    k_tail_prob=0.02,
    p_zero=0.05,
)

### Same as heatmap_realamp, but drop amp/sigma < 5 before building heatmap targets.
### apply_snr_component_label also removes those Gaussians from the spectrum (same as snrk).
MOPRA_GEN_HEATMAP_REALAMP_SNR5 = deepcopy(MOPRA_GEN_HEATMAP_REALAMP)
MOPRA_GEN_HEATMAP_REALAMP_SNR5.update(
    snr_label_k=True,
    snr_label_tol=5.0,
)


def estimate_mopra_noise_from_cube(
    cube_path: Path | str,
    *,
    n_sample: int = 2000,
    seed: int = 0,
    blank_value: float = -1.0,
) -> dict[str, float]:
    """
    ScousePy-style sigma_rms over random sky pixels; use to sanity-check gen.noise_std_range.

    Returns median, p10, p90, and suggested (lo, hi) for noise_std_range.
    """
    from spectral_cube import SpectralCube

    cube = SpectralCube.read(str(Path(cube_path).resolve()), use_dask=False)
    arr = np.asarray(cube.filled(np.nan), dtype=np.float32)
    if arr.ndim == 4:
        arr = arr[0]
    nv, ny, nx = arr.shape
    rng = np.random.default_rng(seed)
    n_sample = min(int(n_sample), ny * nx)
    idx = rng.choice(ny * nx, size=n_sample, replace=False)
    rms_vals: list[float] = []
    for flat in idx:
        y, x = divmod(int(flat), nx)
        spec = arr[:, y, x].astype(np.float64)
        ok = valid_mask_mopra(spec, blank_value=blank_value)
        if ok.sum() < max(40, nv // 6):
            continue
        r = estimate_spectrum_rms(spec[ok])
        if np.isfinite(r) and r > 0.0:
            rms_vals.append(float(r))
    if not rms_vals:
        raise ValueError(f"No finite sigma_rms values from cube {cube_path}")
    a = np.asarray(rms_vals, dtype=np.float64)
    p10, med, p90 = np.percentile(a, [10, 50, 90])
    return {
        "n_used": int(a.size),
        "median": float(med),
        "p10": float(p10),
        "p90": float(p90),
        "suggested_noise_std_range": (float(p10), float(p90)),
    }


def build_mopra_synth_cfg(
    *,
    repo_root=None,
    cube_path=None,
    header_txt=None,
    axis_cube: Path | str | None = None,
    max_components: int = 10,
    gen_preset: str = "default",
    noise_calibration_cube: Path | str | None = None,
) -> dict:
    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[3]
    smooth60_default = root / "data" / MOPRA_CUBE_SMOOTH60

    _SMOOTH60_PRESETS = (
        "scouse_smooth60",
        "scouse_dat",
        "scouse_dat_relaxed",
        "scouse_dat_calibrated",
        "scouse_dat_blend_sat",
        "simple",
        "simple_residual",
        "simple_matched",
        "simple_mix",
        "simple_realamp",
        "simple_realamp_rawk",
        "simple_realamp_snrk",
        "heatmap_realamp",
        "heatmap_realamp_snr5",
    )
    if axis_cube is not None:
        axis_path = Path(axis_cube)
    elif gen_preset in _SMOOTH60_PRESETS:
        axis_path = Path(noise_calibration_cube) if noise_calibration_cube is not None else smooth60_default
    else:
        axis_path = Path(cube_path) if cube_path is not None else None

    cfg = build_mopra_base_cfg(
        repo_root=root,
        cube_path=axis_path if axis_path is not None else cube_path,
        header_txt=header_txt,
    )
    cfg["max_components"] = int(max_components)
    if gen_preset == "legacy":
        gen = deepcopy(MOPRA_GEN_LEGACY)
    elif gen_preset in ("default", "scouse_smooth60"):
        gen = deepcopy(MOPRA_GEN_DEFAULT)
    elif gen_preset == "scouse_dat":
        gen = deepcopy(MOPRA_GEN_SCOUSE_DAT)
    elif gen_preset == "scouse_dat_relaxed":
        gen = deepcopy(MOPRA_GEN_SCOUSE_DAT_RELAXED)
    elif gen_preset == "scouse_dat_calibrated":
        gen = deepcopy(MOPRA_GEN_SCOUSE_DAT_CALIBRATED)
    elif gen_preset == "scouse_dat_blend_sat":
        gen = deepcopy(MOPRA_GEN_SCOUSE_DAT_BLEND_SAT)
    elif gen_preset == "simple":
        gen = deepcopy(MOPRA_GEN_SIMPLE)
        ### MOPRA application: K=0..6 is enough; do not train up to 10.
        if max_components > 6:
            cfg["max_components"] = 6
    elif gen_preset == "simple_residual":
        gen = deepcopy(MOPRA_GEN_SIMPLE_RESIDUAL)
        if max_components > 6:
            cfg["max_components"] = 6
    elif gen_preset == "simple_matched":
        gen = deepcopy(MOPRA_GEN_SIMPLE_MATCHED)
        if max_components > 6:
            cfg["max_components"] = 6
    elif gen_preset == "simple_mix":
        gen = deepcopy(MOPRA_GEN_SIMPLE_MIX)
        if max_components > 6:
            cfg["max_components"] = 6
    elif gen_preset == "simple_realamp":
        gen = deepcopy(MOPRA_GEN_SIMPLE_REALAMP)
        if max_components > 6:
            cfg["max_components"] = 6
    elif gen_preset == "simple_realamp_rawk":
        gen = deepcopy(MOPRA_GEN_SIMPLE_REALAMP_RAWK)
        if max_components > 6:
            cfg["max_components"] = 6
    elif gen_preset == "simple_realamp_snrk":
        gen = deepcopy(MOPRA_GEN_SIMPLE_REALAMP_SNRK)
        if max_components > 6:
            cfg["max_components"] = 6
    elif gen_preset == "heatmap_realamp":
        gen = deepcopy(MOPRA_GEN_HEATMAP_REALAMP)
        if max_components > 6:
            cfg["max_components"] = 6
    elif gen_preset == "heatmap_realamp_snr5":
        gen = deepcopy(MOPRA_GEN_HEATMAP_REALAMP_SNR5)
        if max_components > 6:
            cfg["max_components"] = 6
    else:
        raise ValueError(
            f"Unknown gen_preset {gen_preset!r}; use 'default', 'scouse_smooth60', "
            "'scouse_dat', 'scouse_dat_relaxed', 'scouse_dat_calibrated', "
            "'scouse_dat_blend_sat', 'simple', 'simple_residual', 'simple_matched', "
            "'simple_mix', 'simple_realamp', 'simple_realamp_rawk', "
            "'simple_realamp_snrk', 'heatmap_realamp', 'heatmap_realamp_snr5', or 'legacy'."
        )
    if noise_calibration_cube is not None:
        stats = estimate_mopra_noise_from_cube(noise_calibration_cube)
        lo, hi = stats["suggested_noise_std_range"]
        gen["noise_sigma"] = None
        gen["noise_std_range"] = (lo, hi)
        cfg.setdefault("mopra_meta", {})["noise_calibration"] = stats
    if gen_preset in _SMOOTH60_PRESETS:
        cfg.setdefault("mopra_meta", {})["axis_cube"] = str(axis_path.resolve())
    cfg["gen"] = gen
    return cfg


def _apply_mopra_artifacts(spec: np.ndarray, gen: dict, rng: np.random.Generator) -> np.ndarray:
    out = spec.astype(np.float32, copy=True)
    c = out.size
    if rng.random() < float(gen.get("ripple_prob", 0.0)):
        amp = float(rng.uniform(*gen["ripple_amp_range"]))
        period = float(rng.uniform(*gen["ripple_period_channels_range"]))
        phase = float(rng.uniform(0.0, 2.0 * np.pi))
        ch = np.arange(c, dtype=np.float32)
        out += (amp * np.sin(2.0 * np.pi * ch / period + phase)).astype(np.float32)
    if rng.random() < float(gen.get("spike_prob", 0.0)):
        n_sp = int(rng.integers(1, 4))
        for _ in range(n_sp):
            ix = int(rng.integers(0, c))
            out[ix] += float(rng.uniform(*gen["spike_amp_range"])) * (1.0 if rng.random() < 0.5 else -1.0)
    return out


def generate_mopra_spectrum(cfg: dict, rng: np.random.Generator, v_axis=None) -> dict:
    """
    Synthetic spectrum on the MOPRA CMZ velocity axis.

    Label modes (gen flags):
      - snr_label_k: K = n components with amp/sigma >= snr_tol (close blends count)
      - glance_label_k: SNR gate + cap (resolvable peaks or residual flux; see glance_cap_mode)
      - scouse_label_k: Scouse SNR + optional sep-merge acceptance
    """
    merged = deepcopy(cfg)
    merged["gen"] = deep_update(deepcopy(DEFAULT_GEN), cfg.get("gen", {}))
    gen = merged["gen"]
    ex = generate_spectrum(merged, rng, v_axis=v_axis)
    if gen.get("snr_label_k"):
        ex = apply_snr_component_label(ex, merged)
    elif gen.get("glance_label_k"):
        ex = apply_glance_visible_label(ex, merged)
    else:
        ex = apply_scouse_label_filter(ex, merged)
    ex["spec"] = _apply_mopra_artifacts(ex["spec"], gen, rng)
    ex["spec_clean"] = ex["spec_clean"].astype(np.float32)
    return ex


MOPRA_BASE_CFG = build_mopra_synth_cfg()


__all__ = [
    "MOPRA_BASE_CFG",
    "MOPRA_CUBE_SMOOTH60",
    "MOPRA_GEN_DEFAULT",
    "MOPRA_GEN_LEGACY",
    "MOPRA_GEN_SIMPLE",
    "MOPRA_GEN_SIMPLE_RESIDUAL",
    "MOPRA_GEN_SIMPLE_MATCHED",
    "MOPRA_GEN_SIMPLE_MIX",
    "MOPRA_GEN_SIMPLE_REALAMP",
    "MOPRA_GEN_SIMPLE_REALAMP_RAWK",
    "MOPRA_GEN_SIMPLE_REALAMP_SNRK",
    "MOPRA_GEN_HEATMAP_REALAMP",
    "MOPRA_GEN_SCOUSE_DAT",
    "MOPRA_GEN_SCOUSE_DAT_BLEND_SAT",
    "MOPRA_GEN_SCOUSE_DAT_CALIBRATED",
    "MOPRA_GEN_SCOUSE_DAT_RELAXED",
    "SCOUSE_DAT_K_WEIGHTS",
    "build_mopra_synth_cfg",
    "estimate_mopra_noise_from_cube",
    "generate_mopra_spectrum",
]
