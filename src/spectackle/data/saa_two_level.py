### Synthetic SAA parent + pixel patch generation for two-level Scouse-style training.
from __future__ import annotations

from copy import deepcopy

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from spectackle.data.generator import _make_v_axis
from spectackle.data.mopra_generator import generate_mopra_spectrum, _apply_mopra_artifacts
from spectackle.data.mopra_preprocess import prepare_mopra_input_from_base


def average_noisy_copies(
    spec_clean: np.ndarray,
    noise_std: float,
    gen: dict,
    rng: np.random.Generator,
    *,
    n_copies: int,
) -> np.ndarray:
    """Mean of n independent noisy+artifact draws from the same clean profile (SAA average)."""
    acc = np.zeros_like(spec_clean, dtype=np.float64)
    for _ in range(int(n_copies)):
        noisy = spec_clean.astype(np.float64) + rng.normal(0.0, float(noise_std), size=spec_clean.shape)
        acc += _apply_mopra_artifacts(noisy.astype(np.float32), gen, rng).astype(np.float64)
    return (acc / max(1, int(n_copies))).astype(np.float32)


def pixel_draw_from_clean(
    spec_clean: np.ndarray,
    noise_std: float,
    gen: dict,
    rng: np.random.Generator,
) -> np.ndarray:
    noisy = spec_clean.astype(np.float64) + rng.normal(0.0, float(noise_std), size=spec_clean.shape)
    return _apply_mopra_artifacts(noisy.astype(np.float32), gen, rng)


def generate_saa_pixel_patch(
    cfg: dict,
    rng: np.random.Generator,
    v_axis: np.ndarray | None = None,
    *,
    n_avg: int = 81,
    n_pixels: int = 8,
) -> dict:
    """
    One synthetic SAA patch: shared component draw, parent = averaged noisy copies, pixels independent.

    Returns dict with parent_spec, pixel_specs (n_pixels, C), K_true, k_parent (same as K_true for synth).
    """
    ex = generate_mopra_spectrum(cfg, rng, v_axis=v_axis)
    gen = cfg.get("gen", {})
    k = int(ex["k"])
    clean = ex["spec_clean"]
    noise_std = float(ex["noise_std"][0])
    parent = average_noisy_copies(clean, noise_std, gen, rng, n_copies=n_avg)
    pixels = np.stack(
        [pixel_draw_from_clean(clean, noise_std, gen, rng) for _ in range(int(n_pixels))],
        axis=0,
    )
    return {
        "parent_spec": parent,
        "pixel_specs": pixels,
        "K_true": k,
        "k_parent": k,
    }


class SaaParentDataset(Dataset):
    """Stage 1: SAA-averaged spectra only."""

    def __init__(self, cfg: dict, n_samples: int, base_seed: int = 0, *, n_avg: int = 81):
        self.cfg = cfg
        self.v_axis = _make_v_axis(cfg)
        self.n_samples = int(n_samples)
        self.base_seed = int(base_seed)
        self.n_avg = int(n_avg)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> dict:
        rng = np.random.default_rng(self.base_seed + int(idx))
        patch = generate_saa_pixel_patch(self.cfg, rng, self.v_axis, n_avg=self.n_avg, n_pixels=1)
        spec_norm, valid_mask = prepare_mopra_input_from_base(patch["parent_spec"])
        return {
            "spec_norm": torch.from_numpy(spec_norm).float(),
            "valid_mask": torch.from_numpy(valid_mask).float(),
            "K_true": torch.tensor([patch["K_true"]], dtype=torch.int64),
        }


class SaaPixelCondDataset(Dataset):
    """Stage 2: pixel spectrum + parent SAA spectrum + K_parent conditioning."""

    def __init__(self, cfg: dict, n_samples: int, base_seed: int = 0, *, n_avg: int = 81, n_pixels: int = 8):
        self.cfg = cfg
        self.v_axis = _make_v_axis(cfg)
        self.n_samples = int(n_samples)
        self.base_seed = int(base_seed)
        self.n_avg = int(n_avg)
        self.n_pixels = int(n_pixels)

    def __len__(self) -> int:
        return self.n_samples * self.n_pixels

    def __getitem__(self, idx: int) -> dict:
        patch_idx = int(idx) // self.n_pixels
        pix_idx = int(idx) % self.n_pixels
        rng = np.random.default_rng(self.base_seed + patch_idx)
        patch = generate_saa_pixel_patch(
            self.cfg, rng, self.v_axis, n_avg=self.n_avg, n_pixels=self.n_pixels,
        )
        pixel = patch["pixel_specs"][pix_idx]
        parent = patch["parent_spec"]
        spec_norm, valid_mask = prepare_mopra_input_from_base(pixel)
        parent_norm, _ = prepare_mopra_input_from_base(parent)
        return {
            "spec_norm": torch.from_numpy(spec_norm).float(),
            "parent_spec_norm": torch.from_numpy(parent_norm).float(),
            "valid_mask": torch.from_numpy(valid_mask).float(),
            "K_true": torch.tensor([patch["K_true"]], dtype=torch.int64),
            "K_parent": torch.tensor([patch["k_parent"]], dtype=torch.int64),
        }


def make_saa_parent_loaders(
    cfg: dict,
    *,
    n_train: int = 10_000,
    n_val: int = 2_000,
    bs_train: int = 128,
    bs_val: int = 256,
    shuffle_seed: int = 0,
    n_avg: int = 81,
) -> tuple[DataLoader, DataLoader]:
    train_ds = SaaParentDataset(cfg, n_train, base_seed=shuffle_seed, n_avg=n_avg)
    val_ds = SaaParentDataset(cfg, n_val, base_seed=shuffle_seed + 1_000_000, n_avg=n_avg)
    train_loader = DataLoader(train_ds, batch_size=bs_train, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=bs_val, shuffle=False)
    return train_loader, val_loader


def make_saa_pixel_loaders(
    cfg: dict,
    *,
    n_train: int = 2_000,
    n_val: int = 400,
    bs_train: int = 128,
    bs_val: int = 256,
    shuffle_seed: int = 0,
    n_avg: int = 81,
    n_pixels: int = 8,
) -> tuple[DataLoader, DataLoader]:
    ### n_train/n_val are patch counts; loader length is x n_pixels.
    train_ds = SaaPixelCondDataset(
        cfg, n_train, base_seed=shuffle_seed, n_avg=n_avg, n_pixels=n_pixels,
    )
    val_ds = SaaPixelCondDataset(
        cfg, n_val, base_seed=shuffle_seed + 1_000_000, n_avg=n_avg, n_pixels=n_pixels,
    )
    train_loader = DataLoader(train_ds, batch_size=bs_train, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=bs_val, shuffle=False)
    return train_loader, val_loader


__all__ = [
    "SaaParentDataset",
    "SaaPixelCondDataset",
    "average_noisy_copies",
    "generate_saa_pixel_patch",
    "make_saa_parent_loaders",
    "make_saa_pixel_loaders",
    "pixel_draw_from_clean",
]
