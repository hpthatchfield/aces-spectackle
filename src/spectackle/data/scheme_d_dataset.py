### Scheme D dataset: velocity-ordered slots + optional fixed velocity window.
from __future__ import annotations

from copy import deepcopy

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from spectackle.data.dataset import BASE_CFG
from spectackle.data.generator import DEFAULT_GEN, _make_v_axis, channel_width_kms, generate_spectrum
from spectackle.data.preprocess import prepare_spectrum_input

SCHEME_D_EASY_GEN = dict(
    k_mode="biased_low",
    k_low_max=6,
    k_low_weights=(0.35, 0.28, 0.18, 0.10, 0.06, 0.03),
    k_tail_prob=0.0,
    p_zero=0.12,
    min_component_separation=3.0,
    min_sep_channels=5.0,
    min_amp_ratio=0.35,
    ### Weakest component peak >= 5x noise sigma (SNR floor for Phase-1 easy preset).
    min_peak_height_factor=5.0,
    sigma_min=1.5,
    sigma_max=10.0,
    blend_cluster_prob=0.04,
    cluster_width_range=(8.0, 30.0),
    baseline_poly_prob=0.45,
    noise_std_range=(0.02, 0.12),
)


def cfg_velocity_window(
    base_cfg: dict,
    *,
    v_half_width_kms: float | None = None,
    n_channels: int | None = None,
    v_center_kms: float = 0.0,
) -> dict:
    """
    Narrow (vmin, vmax) at native dv from base_cfg. Same contract as NLW window helper.
    """
    if (v_half_width_kms is None) == (n_channels is None):
        raise ValueError("Specify exactly one of v_half_width_kms or n_channels.")
    out = deepcopy(base_cfg)
    cw = float(channel_width_kms(out))
    if n_channels is not None:
        n = int(n_channels)
        if n < 2:
            raise ValueError("n_channels must be >= 2")
        span = cw * float(n - 1)
    else:
        half = float(v_half_width_kms)
        if half <= 0.0:
            raise ValueError("v_half_width_kms must be positive")
        n = max(2, int(round(2.0 * half / cw)) + 1)
        span = cw * float(n - 1)
    vc = float(v_center_kms)
    out["n_channels"] = n
    out["vrange"] = (vc - 0.5 * span, vc + 0.5 * span)
    return out


def build_scheme_d_easy_cfg(
    *,
    Kmax: int = 6,
    v_center_kms: float = 0.0,
    v_half_width_kms: float = 80.0,
    n_channels: int | None = None,
) -> dict:
    """Easy Phase-1 preset: K <= Kmax, enforced separation, fixed velocity window."""
    cfg = deepcopy(BASE_CFG)
    cfg["min_components"] = 0
    cfg["max_components"] = int(Kmax)
    gen = deepcopy(DEFAULT_GEN)
    gen.update(deepcopy(SCHEME_D_EASY_GEN))
    gen["k_low_max"] = int(Kmax)
    cfg["gen"] = gen
    if n_channels is not None:
        cfg = cfg_velocity_window(cfg, n_channels=n_channels, v_center_kms=v_center_kms)
    else:
        cfg = cfg_velocity_window(cfg, v_half_width_kms=v_half_width_kms, v_center_kms=v_center_kms)
    return cfg


def _order_component_slots(
    amps: np.ndarray,
    mus: np.ndarray,
    sigs: np.ndarray,
    *,
    k: int,
    Kmax: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sort by increasing v; tie-break by decreasing amplitude."""
    pad_v = np.zeros((Kmax,), dtype=np.float32)
    pad_sig = np.zeros((Kmax,), dtype=np.float32)
    pad_amp = np.zeros((Kmax,), dtype=np.float32)
    pad_ok = np.zeros((Kmax,), dtype=np.float32)
    if k <= 0:
        return pad_v, pad_sig, pad_amp, pad_ok
    idx = np.arange(int(k))
    order = np.lexsort((-amps[idx], mus[idx]))
    for j, src in enumerate(order):
        pad_v[j] = float(mus[src])
        pad_sig[j] = float(sigs[src])
        pad_amp[j] = float(amps[src])
        pad_ok[j] = 1.0
    return pad_v, pad_sig, pad_amp, pad_ok


class SchemeDOracleDataset(Dataset):
    """
    Synthetic spectra with velocity-ordered Gaussian slot labels for oracle-K training.

    Targets per slot: v (km/s), log_sigma, amp_norm (peak / per-spectrum sigma from preprocess).
    """

    def __init__(self, cfg: dict, n_samples: int, base_seed: int = 0):
        self.cfg = cfg
        self.v_axis = _make_v_axis(cfg).astype(np.float32)
        self.n_samples = int(n_samples)
        self.base_seed = int(base_seed)
        self.k_component_max = int(cfg["max_components"])

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> dict:
        rng = np.random.default_rng(self.base_seed + int(idx))
        ex = generate_spectrum(self.cfg, rng=rng, v_axis=self.v_axis)
        spec_norm, valid_mask = prepare_spectrum_input(ex["spec"])
        k = int(ex["k"])
        Kmax = self.k_component_max

        ### Per-spectrum sigma for normalized amplitude labels.
        valid = valid_mask > 0.5
        spec_f = np.where(valid, ex["spec"], 0.0)
        cnt = max(1, int(valid.sum()))
        sd = float(np.sqrt(np.sum((spec_f - spec_f.sum() / cnt) ** 2) / cnt) + 1e-6)

        amps = ex["component_amp"][:k].astype(np.float64) if k > 0 else np.zeros(0)
        mus = ex["component_v_kms"][:k].astype(np.float64) if k > 0 else np.zeros(0)
        sigs = ex["component_sigma"][:k].astype(np.float64) if k > 0 else np.zeros(0)

        pad_v, pad_sig, pad_amp, pad_ok = _order_component_slots(
            amps, mus, sigs, k=k, Kmax=Kmax
        )
        amp_norm = (pad_amp / sd).astype(np.float32)
        log_sig = np.log(np.clip(pad_sig, 1e-3, None)).astype(np.float32)

        return {
            "spec": torch.from_numpy(ex["spec"]).float(),
            "spec_norm": torch.from_numpy(spec_norm).float(),
            "valid_mask": torch.from_numpy(valid_mask).float(),
            "spec_clean": torch.from_numpy(ex["spec_clean"]).float(),
            "K_true": torch.tensor([k], dtype=torch.long),
            "component_v_kms": torch.from_numpy(pad_v).float(),
            "component_log_sigma": torch.from_numpy(log_sig).float(),
            "component_amp_norm": torch.from_numpy(amp_norm).float(),
            "component_valid": torch.from_numpy(pad_ok).float(),
            "v_axis": torch.from_numpy(self.v_axis).float(),
        }


def make_scheme_d_oracle_loaders(
    cfg: dict,
    *,
    n_train: int = 10_000,
    n_val: int = 2_000,
    bs_train: int = 128,
    bs_val: int = 256,
    num_workers: int = 0,
    shuffle_seed: int | None = None,
):
    train_ds = SchemeDOracleDataset(cfg, n_samples=n_train, base_seed=0)
    val_ds = SchemeDOracleDataset(cfg, n_samples=n_val, base_seed=10_000_000)
    loader_kw: dict = dict(batch_size=bs_train, shuffle=True, num_workers=num_workers, pin_memory=False)
    if shuffle_seed is not None:
        loader_kw["generator"] = torch.Generator().manual_seed(int(shuffle_seed))
    train_loader = DataLoader(train_ds, **loader_kw)
    val_loader = DataLoader(
        val_ds,
        batch_size=bs_val,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
    return train_loader, val_loader
