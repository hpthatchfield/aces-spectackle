### ACES-axis synthetic loaders (Scheme B/C + heatmap glance labels).
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from spectackle.data.aces_generator import generate_aces_spectrum
from spectackle.data.generator import _make_v_axis
from spectackle.data.preprocess import prepare_spectrum_input


class ACESSpectraDataset(Dataset):
    """
    Deterministic ACES synthetic dataset (generate_aces_spectrum).

    Same batch keys as SyntheticSpectraDataset / MOPRASpectraDataset so heatmap
    training can reuse batch_model_input + component_v_kms targets.
    """

    def __init__(self, cfg: dict, n_samples: int, base_seed: int = 0):
        self.cfg = cfg
        self.v_axis = _make_v_axis(cfg)
        self.n_samples = int(n_samples)
        self.base_seed = int(base_seed)
        self.k_component_max = int(cfg["max_components"])

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> dict:
        rng = np.random.default_rng(self.base_seed + int(idx))
        ex = generate_aces_spectrum(self.cfg, rng=rng, v_axis=self.v_axis)
        spec_norm, valid_mask = prepare_spectrum_input(ex["spec"])
        k = int(ex["k"])
        Kmax = self.k_component_max
        pad_ok = np.zeros((Kmax,), dtype=np.float32)
        if k > 0:
            pad_ok[:k] = 1.0
        return {
            "spec": torch.from_numpy(ex["spec"]).float(),
            "spec_norm": torch.from_numpy(spec_norm).float(),
            "valid_mask": torch.from_numpy(valid_mask).float(),
            "spec_clean": torch.from_numpy(ex["spec_clean"]).float(),
            "K_true": torch.from_numpy(np.array([k], dtype=np.int64)).long(),
            "component_amp": torch.from_numpy(ex["component_amp"].astype(np.float32)),
            "component_v_kms": torch.from_numpy(ex["component_v_kms"].astype(np.float32)),
            "component_sigma": torch.from_numpy(ex["component_sigma"].astype(np.float32)),
            "component_valid": torch.from_numpy(pad_ok),
        }


def make_aces_loaders(
    cfg: dict,
    *,
    n_train: int = 10_000,
    n_val: int = 2_000,
    bs_train: int = 128,
    bs_val: int = 256,
    num_workers: int = 0,
    shuffle_seed: int | None = 42,
):
    train_ds = ACESSpectraDataset(cfg, n_samples=n_train, base_seed=0)
    val_ds = ACESSpectraDataset(cfg, n_samples=n_val, base_seed=10_000_000)
    train_kw: dict = dict(batch_size=bs_train, shuffle=True, num_workers=num_workers, pin_memory=False)
    if shuffle_seed is not None:
        train_kw["generator"] = torch.Generator().manual_seed(int(shuffle_seed))
    train_loader = DataLoader(train_ds, **train_kw)
    val_loader = DataLoader(
        val_ds, batch_size=bs_val, shuffle=False, num_workers=num_workers, pin_memory=False
    )
    return train_loader, val_loader


__all__ = ["ACESSpectraDataset", "make_aces_loaders"]
