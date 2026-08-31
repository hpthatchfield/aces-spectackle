### MOPRA CMZ synthetic dataset for Scheme B (K regression).
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from spectackle.data.generator import _make_v_axis
from spectackle.data.mopra_generator import generate_mopra_spectrum
from spectackle.data.mopra_preprocess import prepare_mopra_input_from_base


class MOPRASpectraDataset(Dataset):
    """
    Deterministic synthetic MOPRA-like dataset indexed by base_seed + idx.

    Returned keys:
      spec, spec_clean, spec_norm, valid_mask, K_true,
      component_amp, component_v_kms, component_sigma, component_valid
    """

    def __init__(self, cfg: dict, n_samples: int, base_seed: int = 0, *, norm_mode: str | None = None):
        self.cfg = cfg
        self.v_axis = _make_v_axis(cfg)
        self.n_samples = int(n_samples)
        self.base_seed = int(base_seed)
        self.k_component_max = int(cfg["max_components"])
        ### Prefer explicit arg; else cfg["norm_mode"]; else legacy zscore.
        self.norm_mode = str(norm_mode or cfg.get("norm_mode") or "zscore")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx: int):
        rng = np.random.default_rng(self.base_seed + int(idx))
        ex = generate_mopra_spectrum(self.cfg, rng, v_axis=self.v_axis)
        spec_norm, valid_mask = prepare_mopra_input_from_base(ex["spec"], norm_mode=self.norm_mode)
        k = int(ex["k"])
        Kmax = self.k_component_max
        pad_ok = np.zeros((Kmax,), dtype=np.float32)
        if k > 0:
            pad_ok[:k] = 1.0
        K_true = np.array([k], dtype=np.int64)
        return {
            "spec": torch.from_numpy(ex["spec"]).float(),
            "spec_norm": torch.from_numpy(spec_norm).float(),
            "valid_mask": torch.from_numpy(valid_mask).float(),
            "spec_clean": torch.from_numpy(ex["spec_clean"]).float(),
            "K_true": torch.from_numpy(K_true).long(),
            "component_amp": torch.from_numpy(ex["component_amp"].astype(np.float32)),
            "component_v_kms": torch.from_numpy(ex["component_v_kms"].astype(np.float32)),
            "component_sigma": torch.from_numpy(ex["component_sigma"].astype(np.float32)),
            "component_valid": torch.from_numpy(pad_ok),
        }


def make_mopra_loaders(
    cfg: dict,
    *,
    n_train: int = 10_000,
    n_val: int = 2_000,
    bs_train: int = 128,
    bs_val: int = 256,
    num_workers: int = 0,
    shuffle_seed: int | None = 42,
    norm_mode: str | None = None,
):
    train_ds = MOPRASpectraDataset(cfg, n_samples=n_train, base_seed=0, norm_mode=norm_mode)
    val_ds = MOPRASpectraDataset(cfg, n_samples=n_val, base_seed=10_000_000, norm_mode=norm_mode)
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


__all__ = ["MOPRASpectraDataset", "make_mopra_loaders"]
