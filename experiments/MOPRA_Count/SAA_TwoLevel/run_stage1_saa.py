#!/usr/bin/env python
"""
Stage 1: train Scheme B on synthetic SAA-averaged spectra (scouse_dat axis).

No human labels - parent spectra are noise-averaged copies of the same synthetic draw.

Example (repo root):
  python experiments/MOPRA_Count/SAA_TwoLevel/run_stage1_saa.py --epochs 8 --tag saa2_stage1
"""
from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

_SCRIPT = Path(__file__).resolve()
_EXP = _SCRIPT.parent
_MOPRA = _EXP.parent
_REPO = _MOPRA.parents[1]
sys.path.insert(0, str(_REPO / "src"))

from spectackle.config import set_cpu_safety  ### noqa: E402
from spectackle.data.mopra_generator import build_mopra_synth_cfg  ### noqa: E402
from spectackle.data.saa_two_level import make_saa_parent_loaders  ### noqa: E402
from spectackle.models import CountNet1DDeep  ### noqa: E402
from spectackle.training import train_scheme_b  ### noqa: E402


def _default_run_dir() -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    return _EXP / "runs" / f"saa2_stage1_{ts}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Two-level SAA stage 1: parent SAA K model.")
    parser.add_argument("--n-train", type=int, default=10_000)
    parser.add_argument("--n-val", type=int, default=2_000)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--bs-train", type=int, default=128)
    parser.add_argument("--bs-val", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--n-blocks", type=int, default=6)
    parser.add_argument("--Kmax", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--n-avg", type=int, default=81, help="Noise copies averaged for parent SAA (~9x9).")
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--gen-preset", type=str, default="scouse_dat")
    args = parser.parse_args()

    set_cpu_safety(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = build_mopra_synth_cfg(
        repo_root=_REPO,
        gen_preset=args.gen_preset,
        max_components=args.Kmax,
        noise_calibration_cube=_REPO / "data" / "CMZ_3mm_HNCO_60.fits",
    )
    run_dir = args.run_dir if args.run_dir is not None else _default_run_dir()
    if args.tag:
        run_dir = run_dir.parent / f"{run_dir.name}_{args.tag}"
    run_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader = make_saa_parent_loaders(
        cfg,
        n_train=args.n_train,
        n_val=args.n_val,
        bs_train=args.bs_train,
        bs_val=args.bs_val,
        shuffle_seed=args.seed,
        n_avg=args.n_avg,
    )
    model = CountNet1DDeep(width=args.width, n_blocks=args.n_blocks)
    history: dict = {}
    train_scheme_b(
        model,
        train_loader,
        val_loader,
        device=args.device,
        lr=args.lr,
        epochs=args.epochs,
        log_every=args.log_every,
        Kmax=args.Kmax,
        history=history,
    )
    manifest = {
        "script": str(_SCRIPT),
        "variant": "saa2_stage1",
        "cfg": cfg,
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "final_val_K_MAE": history["val_K_MAE"][-1] if history.get("val_K_MAE") else None,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    torch.save(model.state_dict(), run_dir / "count_net.pt")
    if history.get("epoch"):
        fig, ax = plt.subplots(figsize=(6, 3.2))
        ax.plot(history["epoch"], history["train_loss_epoch"], "o-", label="train")
        ax.plot(history["epoch"], history["val_K_MAE"], "s-", label="val K MAE")
        ax.legend(fontsize=8)
        ax.set_xlabel("epoch")
        fig.tight_layout()
        fig.savefig(run_dir / "curves.png", dpi=120)
        plt.close(fig)
    print(f"Wrote {run_dir}", flush=True)


if __name__ == "__main__":
    main()
