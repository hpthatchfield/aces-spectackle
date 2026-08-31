### Mixed synthetic + Scouse-labeled real datasets for MOPRA fine-tuning.
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from spectackle.data.mopra_dataset import MOPRASpectraDataset
from spectackle.data.mopra_scouse_labels import load_scouse_labeled_cache


class ScouseLabeledDataset(Dataset):
    """Real spectra from build_scouse_labeled_cache NPZ (spec_norm, valid_mask, K_true)."""

    def __init__(self, cache_path, *, split: str = "train"):
        loaded = load_scouse_labeled_cache(cache_path)
        ar = loaded["arrays"]
        split_code = 0 if split == "train" else 1
        mask = ar["split"] == split_code
        if not np.any(mask):
            raise ValueError(f"No samples for split={split!r} in {cache_path}")
        self.spec_norm = ar["spec_norm"][mask]
        self.valid_mask = ar["valid_mask"][mask]
        self.K_true = ar["K_true"][mask]
        self.l = ar["l"][mask]
        self.b = ar["b"][mask]
        self.n_channels = int(self.spec_norm.shape[1])

    def __len__(self) -> int:
        return int(self.spec_norm.shape[0])

    def __getitem__(self, idx: int) -> dict:
        k = int(self.K_true[idx, 0])
        return {
            "spec_norm": torch.from_numpy(self.spec_norm[idx]).float(),
            "valid_mask": torch.from_numpy(self.valid_mask[idx]).float(),
            "K_true": torch.tensor([k], dtype=torch.long),
            "source": "real",
        }


class MixedMOPRAFinetuneDataset(Dataset):
    """
    Each index draws real or synthetic with probability real_frac (deterministic per idx+seed).
    """

    def __init__(
        self,
        real_ds: ScouseLabeledDataset,
        synth_ds: MOPRASpectraDataset,
        *,
        real_frac: float = 0.5,
        base_seed: int = 0,
        length: int | None = None,
    ):
        self.real = real_ds
        self.synth = synth_ds
        self.real_frac = float(real_frac)
        self.base_seed = int(base_seed)
        self._length = int(length) if length is not None else max(len(real_ds), len(synth_ds)) * 2

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> dict:
        rng = np.random.default_rng(self.base_seed + int(idx))
        if rng.random() < self.real_frac:
            j = int(rng.integers(0, len(self.real)))
            batch = self.real[j]
        else:
            j = int(rng.integers(0, len(self.synth)))
            batch = self.synth[j]
            batch = {k: v for k, v in batch.items() if k in ("spec_norm", "valid_mask", "K_true")}
            batch["source"] = "synth"
        return batch


def make_mopra_finetune_loaders(
    cfg: dict,
    cache_path,
    *,
    real_frac: float = 0.5,
    n_synth_train: int = 10_000,
    bs_train: int = 128,
    bs_val: int = 256,
    num_workers: int = 0,
    shuffle_seed: int | None = 42,
    mixed_length: int | None = None,
):
    """
    Train: mixed real (spatial train split) + synthetic.
    Val: real only (spatial val split).
    """
    real_train = ScouseLabeledDataset(cache_path, split="train")
    real_val = ScouseLabeledDataset(cache_path, split="val")
    synth_train = MOPRASpectraDataset(cfg, n_samples=n_synth_train, base_seed=100_000)

    mixed = MixedMOPRAFinetuneDataset(
        real_train,
        synth_train,
        real_frac=real_frac,
        base_seed=shuffle_seed or 0,
        length=mixed_length,
    )

    loader_kw: dict = dict(batch_size=bs_train, shuffle=True, num_workers=num_workers, pin_memory=False)
    if shuffle_seed is not None:
        loader_kw["generator"] = torch.Generator().manual_seed(int(shuffle_seed))
    train_loader = DataLoader(mixed, **loader_kw)
    val_loader = DataLoader(
        real_val,
        batch_size=bs_val,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
    return train_loader, val_loader, {"n_real_train": len(real_train), "n_real_val": len(real_val)}


__all__ = [
    "ScouseLabeledDataset",
    "MixedMOPRAFinetuneDataset",
    "make_mopra_finetune_loaders",
]
