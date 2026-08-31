### NLW binary dataset - pairs with nlw_generator.generate_nlw_spectrum
from __future__ import annotations

from copy import deepcopy

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from spectackle.config import deep_update

from spectackle.data.generator import _make_v_axis
from spectackle.data.nlw_generator import NLW_GEN_DEFAULT, generate_nlw_spectrum
from spectackle.data.preprocess import prepare_spectrum_input


class NLWSpectraDataset(Dataset):
    """
    Deterministic synthetic NLW dataset indexed by base_seed + idx.

    Returned keys:
      spec       : (C,) float - noisy spectrum (same as generator; may contain NaN/0 pads)
      spec_norm  : (C,) float32 - model input via shared prepare_spectrum_input (valid-only norm)
      valid_mask : (C,) float32 - 1.0 = real channel, 0.0 = NaN/zero pad
      spec_clean : (C,) float
      y_nlw      : (1,) float32 in {0., 1.} - 1 if any NLW component present
      component_v_kms     : (Kmax,) float32 - padded centers (NaN = unused slot)
      component_is_narrow : (Kmax,) float32 - 0/1, only meaningful where valid
      component_valid       : (Kmax,) float32 - 1.0 = real component, 0.0 = pad
    """

    def __init__(self, cfg: dict, n_samples: int, base_seed: int = 0):
        self.cfg = cfg
        self.v_axis = _make_v_axis(cfg)
        self.n_samples = int(n_samples)
        self.base_seed = int(base_seed)
        gen = deep_update(deepcopy(NLW_GEN_DEFAULT), cfg.get("nlw_gen", {}))
        ### Upper bound on injected Gaussians: wide (<= bg_count_max) + narrow (<= nlw_count_max).
        self.k_component_max = int(gen["bg_count_max"]) + int(gen["nlw_count_max"])

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx: int):
        rng = np.random.default_rng(self.base_seed + int(idx))
        ex = generate_nlw_spectrum(self.cfg, rng, v_axis=self.v_axis)
        ### Shared preprocessing contract (identical to real-cube inference).
        spec_norm, valid_mask = prepare_spectrum_input(ex["spec"])
        y = np.array([1.0 if ex["has_nlw"] else 0.0], dtype=np.float32)
        mus = np.asarray(ex["component_v_kms"], dtype=np.float32).reshape(-1)
        is_n = np.asarray(ex["component_is_narrow"], dtype=bool).astype(np.float32).reshape(-1)
        k_tot = int(mus.size)
        pad_v = np.full((self.k_component_max,), np.nan, dtype=np.float32)
        pad_n = np.zeros((self.k_component_max,), dtype=np.float32)
        pad_ok = np.zeros((self.k_component_max,), dtype=np.float32)
        if k_tot > 0:
            if k_tot > self.k_component_max:
                raise ValueError(
                    f"NLW dataset: k_tot={k_tot} exceeds k_component_max={self.k_component_max}; "
                    "increase bg_count_max/nlw_count_max in merged nlw_gen."
                )
            pad_v[:k_tot] = mus.astype(np.float32, copy=False)
            pad_n[:k_tot] = is_n.astype(np.float32, copy=False)
            pad_ok[:k_tot] = 1.0
        return {
            "spec": torch.from_numpy(ex["spec"]).float(),
            "spec_norm": torch.from_numpy(spec_norm).float(),
            "valid_mask": torch.from_numpy(valid_mask).float(),
            "spec_clean": torch.from_numpy(ex["spec_clean"]).float(),
            "y_nlw": torch.from_numpy(y).float(),
            "component_v_kms": torch.from_numpy(pad_v).float(),
            "component_is_narrow": torch.from_numpy(pad_n).float(),
            "component_valid": torch.from_numpy(pad_ok).float(),
        }


def make_nlw_loaders(
    cfg: dict,
    *,
    n_train: int = 10_000,
    n_val: int = 2_000,
    bs_train: int = 128,
    bs_val: int = 256,
    num_workers: int = 0,
    shuffle_seed: int | None = 42,
):
    """Train/val DataLoaders for NLW binary task. Non-overlapping index ranges via base_seed."""
    train_ds = NLWSpectraDataset(cfg, n_samples=n_train, base_seed=0)
    val_ds = NLWSpectraDataset(cfg, n_samples=n_val, base_seed=10_000_000)
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
