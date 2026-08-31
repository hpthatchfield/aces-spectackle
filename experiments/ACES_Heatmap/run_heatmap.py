#!/usr/bin/env python
"""
Train ACES-axis center heatmap (Stage 1), default labels: simple_snr.

Default: +/-80 km/s window, kernel_size=25 (wider RF for fine dv), label_sigma=1 km/s.

  python experiments/ACES_Heatmap/run_heatmap.py \\
    --tag aces_hm_simple_k6_pm80 --Kmax 6 --scheduler
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
_REPO = _SCRIPT.parents[2]
sys.path.insert(0, str(_REPO / "src"))

from spectackle.config import set_cpu_safety  ### noqa: E402
from spectackle.data.aces_dataset import make_aces_loaders  ### noqa: E402
from spectackle.data.aces_generator import build_aces_synth_cfg  ### noqa: E402
from spectackle.data.generator import _make_v_axis, channel_width_kms  ### noqa: E402
from spectackle.models import CenterHeatmapNet1DDeep  ### noqa: E402
from spectackle.models.center_heatmap_decode import (  ### noqa: E402
    eval_center_heatmap_k_decode,
    tune_heatmap_decode_thresholds,
)
from spectackle.training import eval_center_heatmap_metrics, train_center_heatmap  ### noqa: E402


def _default_run_dir() -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    return _SCRIPT.parent / "runs" / f"aces_heatmap_{ts}"


@torch.no_grad()
def _plot_examples(model, val_loader, v_axis, out_path, *, device, n_examples=6):
    model.eval()
    batch = next(iter(val_loader))
    x = batch["spec_norm"].to(device)
    mask = batch["valid_mask"].to(device)
    prob = torch.sigmoid(model(x, mask)).cpu().numpy()
    spec = batch["spec_norm"].cpu().numpy()
    valid = batch["valid_mask"].cpu().numpy() > 0.5
    v_true = batch["component_v_kms"].cpu().numpy()
    v_ok = batch["component_valid"].cpu().numpy() > 0.5
    n = min(n_examples, spec.shape[0])
    fig, axes = plt.subplots(n, 1, figsize=(9, 1.7 * n), sharex=True)
    if n == 1:
        axes = [axes]
    for i in range(n):
        ax = axes[i]
        m = valid[i]
        ax.plot(v_axis[m], spec[i][m], color="0.4", lw=0.8)
        ax2 = ax.twinx()
        ax2.plot(v_axis[m], prob[i][m], color="tab:red", lw=1.0)
        ax2.set_ylim(-0.05, 1.05)
        for vc in v_true[i][v_ok[i]]:
            ax.axvline(float(vc), color="tab:blue", ls="--", lw=0.8)
        ax.set_ylabel("T (norm)", fontsize=8)
        ax2.set_ylabel("P", fontsize=8, color="tab:red")
    axes[-1].set_xlabel("velocity (km/s)")
    axes[0].set_title("ACES heatmap: red=P(center), blue dashed=true centers")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="ACES center heatmap training.")
    parser.add_argument("--n-train", type=int, default=10_000)
    parser.add_argument("--n-val", type=int, default=2_000)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--bs-train", type=int, default=64)
    parser.add_argument("--bs-val", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--n-blocks", type=int, default=6)
    parser.add_argument("--kernel-size", type=int, default=25, help="ACES default 25 for RF; MOPRA used 9.")
    parser.add_argument("--Kmax", type=int, default=6)
    parser.add_argument("--label-sigma-kms", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--scheduler", action="store_true")
    parser.add_argument("--coord", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--v-scale-kms", type=float, default=100.0)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--gen-preset", choices=("simple_snr", "simple_glance", "default"), default="simple_snr")
    parser.add_argument("--v-half-kms", type=float, default=80.0)
    parser.add_argument("--v-center-kms", type=float, default=0.0)
    parser.add_argument("--full-axis", action="store_true")
    parser.add_argument("--decode-min-sep-kms", type=float, default=4.0)
    args = parser.parse_args()

    set_cpu_safety(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = build_aces_synth_cfg(
        Kmax=args.Kmax,
        v_center_kms=args.v_center_kms,
        v_half_width_kms=None if args.full_axis else float(args.v_half_kms),
        full_axis=args.full_axis,
        gen_preset=args.gen_preset,
    )

    run_dir = args.run_dir if args.run_dir is not None else _default_run_dir()
    if args.tag:
        run_dir = run_dir.parent / f"{run_dir.name}_{args.tag}"
    run_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader = make_aces_loaders(
        cfg,
        n_train=args.n_train,
        n_val=args.n_val,
        bs_train=args.bs_train,
        bs_val=args.bs_val,
        shuffle_seed=args.seed,
    )

    v_axis = _make_v_axis(cfg).astype(np.float32)
    cw = float(channel_width_kms(cfg))
    coord = (v_axis / float(args.v_scale_kms)) if args.coord else None
    model = CenterHeatmapNet1DDeep(
        width=args.width,
        n_blocks=args.n_blocks,
        coord=coord,
        kernel_size=args.kernel_size,
    )

    history: dict = {}
    train_center_heatmap(
        model,
        train_loader,
        val_loader,
        v_axis,
        device=args.device,
        lr=args.lr,
        epochs=args.epochs,
        log_every=args.log_every,
        label_sigma_kms=args.label_sigma_kms,
        use_scheduler=args.scheduler,
        history=history,
    )

    metrics = eval_center_heatmap_metrics(
        model, val_loader, v_axis, device=args.device, label_sigma_kms=args.label_sigma_kms
    )
    decode_tune = tune_heatmap_decode_thresholds(
        model, val_loader, v_axis, device=args.device, Kmax=int(args.Kmax),
        min_sep_kms=float(args.decode_min_sep_kms),
    )
    decode_height = float(decode_tune["height"])
    decode_prom = float(decode_tune["prominence"])
    k_decode = eval_center_heatmap_k_decode(
        model, val_loader, v_axis, device=args.device, label_sigma_kms=args.label_sigma_kms,
        height=decode_height, prominence=decode_prom,
        min_sep_kms=float(args.decode_min_sep_kms), Kmax=int(args.Kmax),
    )
    print(
        f"decode tune: height={decode_height:.2f} prom={decode_prom:.2f}  "
        f"val K_MAE={k_decode['k_mae']:.4f} exact={k_decode['k_exact_frac']:.3f}",
        flush=True,
    )

    rf_ch = 1 + int(args.n_blocks) * (int(args.kernel_size) - 1)
    manifest = {
        "script": str(_SCRIPT),
        "variant": "center_heatmap",
        "cfg": cfg,
        "vel_window": {
            "n_channels": int(cfg["n_channels"]),
            "channel_width_kms": cw,
            "vrange": list(cfg["vrange"]),
            "v_center_kms": float(args.v_center_kms),
            "v_half_width_kms": None if args.full_axis else float(args.v_half_kms),
        },
        "coord": {"enabled": bool(args.coord), "v_scale_kms": float(args.v_scale_kms)},
        "label_sigma_kms": float(args.label_sigma_kms),
        "kernel_size": int(args.kernel_size),
        "receptive_field_channels": int(rf_ch),
        "receptive_field_kms": float(rf_ch * cw),
        "decode": {
            "height": decode_height,
            "prominence": decode_prom,
            "min_sep_kms": float(args.decode_min_sep_kms),
            "Kmax": int(args.Kmax),
        },
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "final_val": metrics,
        "final_val_k_decode": k_decode,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    torch.save(model.state_dict(), run_dir / "center_heatmap_net.pt")

    if history.get("epoch"):
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
        ep = history["epoch"]
        axes[0].plot(ep, history["train_loss_epoch"], "o-", label="train")
        axes[0].plot(ep, history["val_loss"], "s-", label="val")
        axes[0].set_xlabel("epoch")
        axes[0].set_ylabel("focal loss")
        axes[0].set_title(f"ACES heatmap ({args.gen_preset}, k={args.kernel_size})")
        axes[0].legend(fontsize=8)
        axes[1].plot(ep, history["val_pos_prob"], "s-", label="peak prob")
        axes[1].plot(ep, history["val_neg_prob"], "^-", label="background")
        axes[1].plot(ep, history["val_auc"], "d-", label="channel AUC")
        axes[1].set_xlabel("epoch")
        axes[1].set_ylim(-0.05, 1.05)
        axes[1].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(run_dir / "curves.png", dpi=120)
        plt.close(fig)

    _plot_examples(model, val_loader, v_axis, run_dir / "example_heatmaps.png", device=args.device)
    print(
        f"final val: loss={metrics['loss']:.4f}  peak_prob={metrics['pos_prob_mean']:.3f}  "
        f"RF~{rf_ch} ch ({rf_ch * cw:.1f} km/s)"
    )
    print(f"Wrote run artifacts to {run_dir}")


if __name__ == "__main__":
    main()
