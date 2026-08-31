#!/usr/bin/env python
"""
Train Scheme D-lite (CenterNet1DDeep) on MOPRA-axis synthetic spectra.

Two heads on one encoder: Scheme C K logits + Kmax velocity-center slots (km/s).
Labels come from the same MOPRA generator as Scheme B, so K is directly comparable
to run_baseline.py; the extra v head gives per-component velocity centers for the
Henshaw comparison and for debugging where peaks are missed.

Run from repo root (cap BLAS threads on macOS if needed):
  python experiments/MOPRA_Count/run_dlite.py --gen-preset scouse_dat --tag dlite_v1

Writes experiments/MOPRA_Count/runs/<tag>/ with manifest.json, history.json,
center_net.pt, curves.png
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

from spectackle.config import set_cpu_safety  ### noqa: E402
from spectackle.data.generator import _make_v_axis  ### noqa: E402
from spectackle.data.mopra_dataset import make_mopra_loaders  ### noqa: E402
from spectackle.data.mopra_generator import build_mopra_synth_cfg  ### noqa: E402
from spectackle.models import CenterNet1DDeep  ### noqa: E402
from spectackle.training import eval_scheme_d_lite_metrics, train_scheme_d_lite  ### noqa: E402


def _default_run_dir() -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    return _SCRIPT.parent / "runs" / f"mopra_dlite_{ts}"


def main() -> None:
    parser = argparse.ArgumentParser(description="MOPRA-axis Scheme D-lite (K + velocity centers).")
    parser.add_argument("--n-train", type=int, default=10_000)
    parser.add_argument("--n-val", type=int, default=2_000)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--bs-train", type=int, default=128)
    parser.add_argument("--bs-val", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--n-blocks", type=int, default=6)
    parser.add_argument(
        "--Kmax",
        type=int,
        default=6,
        help="K classes (0..Kmax) and velocity slots. Must be >= generator k_low_max.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--scheduler", action="store_true")
    parser.add_argument("--w-v", type=float, default=1.0, help="Weight on v slot loss vs K loss.")
    parser.add_argument(
        "--k-mode",
        choices=("ce", "reg"),
        default="ce",
        help="K head: ce = Scheme C logits; reg = Scheme B SmoothL1 scalar (better Scouse MAE).",
    )
    parser.add_argument(
        "--coord",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Feed km/s-per-channel as a 2nd input channel so the v head can localize "
        "on the wide MOPRA axis (CoordConv). Default on.",
    )
    parser.add_argument(
        "--v-scale-kms",
        type=float,
        default=100.0,
        help="Fixed physical scale for the coord channel (v/scale). Shared across datasets so "
        "absolute velocity means the same thing at any resolution.",
    )
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
        help="Synthetic generator preset (simple / simple_matched / simple_residual / ...).",
    )
    parser.add_argument(
        "--noise-calibration-cube",
        type=Path,
        default=None,
        help="Optional cube to re-estimate gen.noise_std_range (e.g. CMZ_3mm_HNCO_60.fits).",
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
    ### Guard: K classes must cover the generator's max draw so labels stay in range.
    k_low_max = int(cfg["gen"].get("k_low_max", args.Kmax))
    if k_low_max > args.Kmax:
        raise ValueError(
            f"--Kmax={args.Kmax} < generator k_low_max={k_low_max}; raise --Kmax."
        )

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
    )

    coord = None
    if args.coord:
        v_axis = _make_v_axis(cfg).astype(np.float32)
        coord = v_axis / float(args.v_scale_kms)
    model = CenterNet1DDeep(
        Kmax=args.Kmax,
        width=args.width,
        n_blocks=args.n_blocks,
        coord=coord,
        k_mode=args.k_mode,
    )
    history: dict = {}
    train_scheme_d_lite(
        model,
        train_loader,
        val_loader,
        device=args.device,
        lr=args.lr,
        epochs=args.epochs,
        log_every=args.log_every,
        w_v=args.w_v,
        use_scheduler=args.scheduler,
        history=history,
    )

    metrics = eval_scheme_d_lite_metrics(model, val_loader, device=args.device)

    cw = float(cfg["vrange"][1] - cfg["vrange"][0]) / max(1, int(cfg["n_channels"]) - 1)
    manifest = {
        "script": str(_SCRIPT),
        "variant": "d_lite",
        "oracle": False,
        "cfg": cfg,
        "vel_window": {
            "n_channels": int(cfg["n_channels"]),
            "channel_width_kms": cw,
            "vrange": list(cfg["vrange"]),
        },
        "coord": {
            "enabled": bool(args.coord),
            "v_scale_kms": float(args.v_scale_kms),
        },
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "final_val": metrics,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    torch.save(model.state_dict(), run_dir / "center_net.pt")

    if history.get("epoch"):
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
        ep = history["epoch"]
        axes[0].plot(ep, history["train_loss_epoch"], "o-", label="train")
        axes[0].set_xlabel("epoch")
        axes[0].set_ylabel("loss")
        axes[0].set_title(f"MOPRA D-lite ({args.gen_preset})")
        axes[0].legend(fontsize=8)
        axes[1].plot(ep, history["val_mae_k"], "s-", label="K MAE")
        axes[1].plot(ep, history["val_k_acc"], "^-", label="K acc")
        axes[1].set_xlabel("epoch")
        axes[1].legend(fontsize=8)
        axes[2].plot(ep, history["val_mae_v_oracle"], "s-", label="v oracle-mask")
        axes[2].plot(ep, history["val_mae_v_pred_k"], "^-", label="v pred-K gate")
        axes[2].set_xlabel("epoch")
        axes[2].set_ylabel("km/s")
        axes[2].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(run_dir / "curves.png", dpi=120)
        plt.close(fig)

    print(
        f"final val: K_MAE={metrics['mae_k']:.3f}  K_acc={metrics['k_acc']:.3f}  "
        f"mae_v_oracle={metrics['mae_v_oracle']:.3f} km/s  "
        f"mae_v_predK={metrics['mae_v_pred_k']:.3f} km/s"
    )
    print(f"Wrote run artifacts to {run_dir}")


if __name__ == "__main__":
    main()
