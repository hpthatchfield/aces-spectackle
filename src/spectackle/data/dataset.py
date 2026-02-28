### SyntheticSpectraDataset, make_loaders
from copy import deepcopy

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from spectackle.config import deep_update
from spectackle.data.generator import DEFAULT_GEN, _make_v_axis, generate_spectrum


class SyntheticSpectraDataset(Dataset):
    """
    Deterministic synthetic dataset indexed by base_seed + idx.

    Returned keys:
      spec       : (C,) float
      spec_clean : (C,) float
      K_true     : (1,) long, integer class in [0, Kmax]
    """

    def __init__(self, cfg: dict, n_samples: int, base_seed: int = 0):
        self.cfg = cfg
        self.v_axis = _make_v_axis(cfg)
        self.n_samples = int(n_samples)
        self.base_seed = int(base_seed)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx: int):
        rng = np.random.default_rng(self.base_seed + int(idx))
        ex = generate_spectrum(self.cfg, rng=rng, v_axis=self.v_axis)
        K_true = np.array([ex["k"]], dtype=np.int64)
        return {
            "spec": torch.from_numpy(ex["spec"]).float(),
            "spec_clean": torch.from_numpy(ex["spec_clean"]).float(),
            "K_true": torch.from_numpy(K_true).long(),
        }


BASE_CFG = dict(
    n_channels=256,
    min_components=0,
    max_components=10,
    vrange=(-200.0, 200.0),
    gen=deepcopy(DEFAULT_GEN),
)


def make_loaders(cfg: dict, *, n_train=50_000, n_val=5_000, bs_train=128, bs_val=256):
    train_ds = SyntheticSpectraDataset(cfg, n_samples=n_train, base_seed=0)
    val_ds = SyntheticSpectraDataset(cfg, n_samples=n_val, base_seed=10_000_000)
    train_loader = DataLoader(
        train_ds, batch_size=bs_train, shuffle=True, num_workers=0, pin_memory=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=bs_val, shuffle=False, num_workers=0, pin_memory=False
    )
    return train_loader, val_loader
