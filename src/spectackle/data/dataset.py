### SyntheticSpectraDataset, make_loaders
from copy import deepcopy

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from spectackle.data.generator import DEFAULT_GEN, _make_v_axis, generate_spectrum
from spectackle.data.preprocess import prepare_spectrum_input


class SyntheticSpectraDataset(Dataset):
    """
    Deterministic synthetic dataset indexed by base_seed + idx.

    Returned keys:
      spec       : (C,) float - noisy spectrum (may contain NaN/0 pads when gen.mask_prob > 0)
      spec_norm  : (C,) float32 - model input via prepare_spectrum_input (valid-only norm)
      valid_mask : (C,) float32 - 1.0 = real channel, 0.0 = NaN/zero pad
      spec_clean : (C,) float
      K_true     : (1,) long, integer class in [0, Kmax]
      component_amp     : (Kmax,) float32 - padded amplitudes; unused slots are 0
      component_v_kms   : (Kmax,) float32 - padded centers (km/s); unused slots are 0
      component_sigma   : (Kmax,) float32 - padded sigma (km/s); unused slots are 0
      component_valid   : (Kmax,) float32 - 1.0 = real component, 0.0 = pad
    """

    def __init__(self, cfg: dict, n_samples: int, base_seed: int = 0):
        self.cfg = cfg
        self.v_axis = _make_v_axis(cfg)
        self.n_samples = int(n_samples)
        self.base_seed = int(base_seed)
        self.k_component_max = int(cfg["max_components"])

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx: int):
        rng = np.random.default_rng(self.base_seed + int(idx))
        ex = generate_spectrum(self.cfg, rng=rng, v_axis=self.v_axis)
        spec_norm, valid_mask = prepare_spectrum_input(ex["spec"])
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


BASE_CFG = dict(
    n_channels=256,
    min_components=0,
    max_components=20,
    vrange=(-200.0, 200.0),
    gen=deepcopy(DEFAULT_GEN),
)


def make_loaders(
    cfg: dict,
    *,
    n_train=50_000,
    n_val=5_000,
    bs_train=128,
    bs_val=256,
    num_workers=0,
    shuffle_seed=None,
):
    """num_workers=0 on mac (semaphore leak). shuffle_seed for reproducible shuffle."""
    train_ds = SyntheticSpectraDataset(cfg, n_samples=n_train, base_seed=0)
    val_ds = SyntheticSpectraDataset(cfg, n_samples=n_val, base_seed=10_000_000)
    loader_kw = dict(batch_size=bs_train, shuffle=True, num_workers=num_workers, pin_memory=False)
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
