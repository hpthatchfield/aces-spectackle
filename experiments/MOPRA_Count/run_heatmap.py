#!/usr/bin/env python
"""
Train a per-channel center heatmap on MOPRA-axis synthetic spectra.

Emits P(center) per velocity channel. K/centers come later from peak decode or a K head.

  python experiments/MOPRA_Count/run_heatmap.py --gen-preset simple --tag heatmap_v1
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
from spectackle.data.generator import _make_v_axis  ### noqa: E402
from spectackle.data.mopra_dataset import make_mopra_loaders  ### noqa: E402
from spectackle.data.mopra_generator import build_mopra_synth_cfg  ### noqa: E402
from spectackle.data.mopra_preprocess import NORM_MODES  ### noqa: E402
from spectackle.models import CenterHeatmapNet1DDeep  ### noqa: E402
from spectackle.models.center_heatmap_decode import (  ### noqa: E402
    eval_center_heatmap_k_decode,
    tune_heatmap_decode_thresholds,
)
from spectackle.training import (  ### noqa: E402
    eval_center_heatmap_metrics,
    train_center_heatmap,
)


def _default_run_dir() -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    return _SCRIPT.parent / "runs" / f"mopra_heatmap_{ts}"


@torch.no_grad()
def _plot_examples(model, val_loader, v_axis, out_path, *, device, n_examples=6):
    """Overlay predicted center probability + true centers on a few val spectra."""
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
        ax.plot(v_axis[m], spec[i][m], color="0.4", lw=0.8, label="spectrum (norm)")
        ax2 = ax.twinx()
        ax2.plot(v_axis[m], prob[i][m], color="tab:red", lw=1.0, label="P(center)")
        ax2.set_ylim(-0.05, 1.05)
        for vc in v_true[i][v_ok[i]]:
            ax.axvline(float(vc), color="tab:blue", ls="--", lw=0.8)
        ax.set_ylabel("T (norm)", fontsize=8)
        ax2.set_ylabel("P", fontsize=8, color="tab:red")
    axes[-1].set_xlabel("velocity (km/s)")
    axes[0].set_title("Center heatmap: red=P(center), blue dashed=true centers")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="MOPRA-axis per-channel center heatmap training.")
    parser.add_argument("--n-train", type=int, default=10_000)
    parser.add_argument("--n-val", type=int, default=2_000)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--bs-train", type=int, default=128)
    parser.add_argument("--bs-val", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--n-blocks", type=int, default=6)
    parser.add_argument("--Kmax", type=int, default=10, help="Generator max components + decode cap.")
    parser.add_argument(
        "--label-sigma-kms",
        type=float,
        default=4.0,
        help="Gaussian splat width of the center target (km/s). ~2 channels at dv=2.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--scheduler", action="store_true")
    parser.add_argument(
        "--coord",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also feed km/s-per-channel (absolute-velocity prior). Off by default; the heatmap "
        "head is already positional.",
    )
    parser.add_argument("--v-scale-kms", type=float, default=100.0)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--cube", type=Path, default=_REPO / "data" / "CMZ_3mm_HNCO.fits")
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
            "heatmap_realamp",
            "heatmap_realamp_snr5",
            "legacy",
        ),
        default="scouse_dat",
        help="Synthetic generator preset. heatmap_realamp = realamp+cluster, planted "
        "centers (no glance), Scouse-like K prior (heatmap benchmark). "
        "heatmap_realamp_snr5 = same but only amp/sigma>=5 centers enter the heatmap target.",
    )
    parser.add_argument(
        "--noise-calibration-cube",
        type=Path,
        default=_REPO / "data" / "CMZ_3mm_HNCO_60.fits",
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

    v_axis = _make_v_axis(cfg).astype(np.float32)
    coord = (v_axis / float(args.v_scale_kms)) if args.coord else None
    model = CenterHeatmapNet1DDeep(width=args.width, n_blocks=args.n_blocks, coord=coord)

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
    )
    decode_height = float(decode_tune["height"])
    decode_prom = float(decode_tune["prominence"])
    k_decode = eval_center_heatmap_k_decode(
        model, val_loader, v_axis, device=args.device, label_sigma_kms=args.label_sigma_kms,
        height=decode_height, prominence=decode_prom, Kmax=int(args.Kmax),
    )
    print(
        f"decode tune: height={decode_height:.2f} prom={decode_prom:.2f}  "
        f"val K_MAE={k_decode['k_mae']:.4f}  exact={k_decode['k_exact_frac']:.3f}",
        flush=True,
    )

    cw = float(cfg["vrange"][1] - cfg["vrange"][0]) / max(1, int(cfg["n_channels"]) - 1)
    manifest = {
        "script": str(_SCRIPT),
        "variant": "center_heatmap",
        "cfg": cfg,
        "norm_mode": str(args.norm_mode),
        "vel_window": {
            "n_channels": int(cfg["n_channels"]),
            "channel_width_kms": cw,
            "vrange": list(cfg["vrange"]),
        },
        "coord": {"enabled": bool(args.coord), "v_scale_kms": float(args.v_scale_kms)},
        "label_sigma_kms": float(args.label_sigma_kms),
        "decode": {
            "height": decode_height,
            "prominence": decode_prom,
            "min_sep_kms": 4.0,
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
        axes[0].set_title(f"Center heatmap ({args.gen_preset})")
        axes[0].legend(fontsize=8)
        axes[1].plot(ep, history["val_pos_prob"], "s-", label="peak prob")
        axes[1].plot(ep, history["val_neg_prob"], "^-", label="background prob")
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
        f"bg_prob={metrics['neg_prob_mean']:.3f}  channel_AUC={metrics['channel_auc']:.3f}"
    )
    print(f"Wrote run artifacts to {run_dir}")


if __name__ == "__main__":
    main()
