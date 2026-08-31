#!/usr/bin/env python
"""
Train Scheme B (CountNet1DDeep) on MOPRA-axis synthetic spectra.

Run from repo root (cap BLAS threads on macOS if needed):
  python experiments/MOPRA_Count/run_baseline.py

Writes experiments/MOPRA_Count/runs/<tag>/ with manifest.json, history.json, count_net.pt, curves.png
"""
from __future__ import annotations

import os

### Cap BLAS/OpenMP before numpy/torch import (macOS teardown segfaults otherwise).
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
_REPO = _SCRIPT.parents[2]
sys.path.insert(0, str(_REPO / "src"))

from spectackle.data.mopra_dataset import make_mopra_loaders  ### noqa: E402
from spectackle.data.mopra_generator import build_mopra_synth_cfg  ### noqa: E402
from spectackle.data.mopra_preprocess import NORM_MODES  ### noqa: E402
from spectackle.config import set_cpu_safety  ### noqa: E402
from spectackle.models import CountNet1DDeep  ### noqa: E402
from spectackle.plotting import collect_count_predictions_b, mae_by_true_k  ### noqa: E402
from spectackle.training import train_scheme_b  ### noqa: E402


def _default_run_dir() -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    return _SCRIPT.parent / "runs" / f"mopra_count_{ts}"


def main() -> None:
    parser = argparse.ArgumentParser(description="MOPRA-axis Scheme B baseline training.")
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
    parser.add_argument("--scheduler", action="store_true")
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument(
        "--cube",
        type=Path,
        default=_REPO / "data" / "CMZ_3mm_HNCO.fits",
        help="MOPRA FITS cube (axis metadata; not used for training data).",
    )
    parser.add_argument(
        "--gen-preset",
        choices=(
            "default",
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
            "legacy",
        ),
        default="scouse_dat",
        help="Synthetic generator preset. simple_matched = simple draws + matched-filter "
        "glance credit (SNR floor + 4 km/s merge). simple_residual = residual-flux cap. "
        "simple_realamp_snrk = realamp morphology, K = SNR-pass count.",
    )
    parser.add_argument(
        "--noise-calibration-cube",
        type=Path,
        default=_REPO / "data" / "CMZ_3mm_HNCO_60.fits",
        help="Cube to re-estimate gen.noise_std_range (default: smooth60).",
    )
    parser.add_argument(
        "--k-train-weights",
        type=str,
        default="",
        help="Optional comma K weights for SmoothL1, length Kmax+1 (e.g. '1,1,1.5,3,5' for K=0..4). "
        "Up-weights high-K loss without changing the generator prior.",
    )
    parser.add_argument(
        "--norm-mode",
        choices=NORM_MODES,
        default="zscore",
        help="Input normalization: zscore=(T-mean)/std (legacy); rms=(T-median)/sigma_rms (SNR-preserving).",
    )
    args = parser.parse_args()

    set_cpu_safety(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = build_mopra_synth_cfg(
        repo_root=_REPO,
        cube_path=args.cube,
        max_components=args.Kmax,
        gen_preset=args.gen_preset,
        noise_calibration_cube=args.noise_calibration_cube,
    )
    cfg["norm_mode"] = str(args.norm_mode)
    run_dir = args.run_dir if args.run_dir is not None else _default_run_dir()
    if args.tag:
        run_dir = run_dir.parent / f"{run_dir.name}_{args.tag}"
    run_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader = make_mopra_loaders(
        cfg,
        n_train=args.n_train,
        n_val=args.n_val,
        bs_train=args.bs_train,
        bs_val=args.bs_val,
        shuffle_seed=args.seed,
        norm_mode=args.norm_mode,
    )

    k_loss_weights = None
    if args.k_train_weights.strip():
        parts = [float(x.strip()) for x in args.k_train_weights.split(",")]
        if len(parts) != int(args.Kmax) + 1:
            raise ValueError(
                f"--k-train-weights needs {int(args.Kmax) + 1} values (K=0..{args.Kmax}), got {len(parts)}"
            )
        k_loss_weights = np.asarray(parts, dtype=np.float32)

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
        use_scheduler=args.scheduler,
        k_loss_weights=k_loss_weights,
        history=history,
    )

    y_true, y_pred = collect_count_predictions_b(model, val_loader, device=args.device, Kmax=args.Kmax)
    final_mae = float(np.abs(y_pred - y_true).mean())
    mae_k = mae_by_true_k(y_true, y_pred, args.Kmax)

    manifest = {
        "script": str(_SCRIPT),
        "cfg": cfg,
        "norm_mode": str(args.norm_mode),
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "final_val_K_MAE": final_mae,
        "mae_by_K": {str(k): v for k, v in mae_k.items()},
        "k_train_weights": k_loss_weights.tolist() if k_loss_weights is not None else None,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    torch.save(model.state_dict(), run_dir / "count_net.pt")

    if history.get("epoch"):
        fig, ax = plt.subplots(figsize=(6, 3.2))
        ep = history["epoch"]
        ax.plot(ep, history["train_loss_epoch"], "o-", label="train")
        ax.plot(ep, history["val_K_MAE"], "s-", label="val K MAE")
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss / MAE")
        ax.set_title(f"MOPRA synthetic Scheme B (Kmax={args.Kmax}, n_ch={cfg['n_channels']})")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(run_dir / "curves.png", dpi=120)
        plt.close(fig)

    print(f"final val_K_MAE {final_mae:.4f}")
    print(f"Wrote run artifacts to {run_dir}")


if __name__ == "__main__":
    main()
