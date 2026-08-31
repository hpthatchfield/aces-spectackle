#!/usr/bin/env python
"""
Stage-2 K head on a pretrained ACES center heatmap (frozen Stage 1 by default).

  python experiments/ACES_Heatmap/run_heatmap_k.py \\
    --heatmap-run-dir experiments/ACES_Heatmap/runs/aces_heatmap_<ts>_aces_hm_simple_k6_pm80 \\
    --tag aces_hm_k_simple_k6_pm80 --scheduler
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
from spectackle.models import CenterHeatmapNet1DDeep, HeatmapCountNet  ### noqa: E402
from spectackle.models.center_heatmap_decode import eval_center_heatmap_k_decode  ### noqa: E402
from spectackle.training import eval_heatmap_count_k, train_heatmap_count  ### noqa: E402


def _path_rel_to_repo(path: Path, repo: Path) -> str:
    """Prefer repo-relative paths in manifests so runs move across machines."""
    try:
        return str(path.resolve().relative_to(repo.resolve()))
    except ValueError:
        return str(path.resolve())


def _resolve_run_path(path_str: str, repo: Path) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    return (repo / p).resolve()


def _default_run_dir() -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    return _SCRIPT.parent / "runs" / f"aces_heatmap_k_{ts}"


def _load_heatmap_stage1(run_dir: Path, *, device: str) -> tuple[CenterHeatmapNet1DDeep, dict, np.ndarray]:
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    args_m = dict(manifest.get("args", {}))
    cfg = manifest["cfg"]
    width = int(args_m.get("width", 96))
    n_blocks = int(args_m.get("n_blocks", 6))
    kernel_size = int(manifest.get("kernel_size", args_m.get("kernel_size", 9)))
    coord = None
    if manifest.get("coord", {}).get("enabled"):
        v_scale = float(manifest["coord"].get("v_scale_kms", 100.0))
        coord = _make_v_axis(cfg).astype(np.float32) / v_scale
    heatmap = CenterHeatmapNet1DDeep(
        width=width, n_blocks=n_blocks, coord=coord, kernel_size=kernel_size
    )
    state = torch.load(run_dir / "center_heatmap_net.pt", map_location="cpu")
    heatmap.load_state_dict(state)
    heatmap.to(device)
    heatmap.eval()
    v_axis = _make_v_axis(cfg).astype(np.float32)
    return heatmap, manifest, v_axis


def main() -> None:
    parser = argparse.ArgumentParser(description="ACES Stage-2 K head on frozen heatmap.")
    parser.add_argument("--heatmap-run-dir", type=Path, required=True)
    parser.add_argument("--n-train", type=int, default=10_000)
    parser.add_argument("--n-val", type=int, default=2_000)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--bs-train", type=int, default=64)
    parser.add_argument("--bs-val", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--n-blocks", type=int, default=6)
    parser.add_argument("--kernel-size", type=int, default=None, help="Default: inherit Stage-1.")
    parser.add_argument("--k-input", choices=("p", "spec_p"), default="spec_p")
    parser.add_argument(
        "--freeze-heatmap",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--scheduler", action="store_true")
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--gen-preset", choices=("simple_snr", "simple_glance", "default"), default=None)
    parser.add_argument("--Kmax", type=int, default=None)
    parser.add_argument("--v-half-kms", type=float, default=None)
    parser.add_argument("--v-center-kms", type=float, default=None)
    args = parser.parse_args()

    set_cpu_safety(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    stage1_dir = args.heatmap_run_dir.resolve()
    heatmap, stage1_manifest, v_axis = _load_heatmap_stage1(stage1_dir, device=args.device)
    stage1_args = dict(stage1_manifest.get("args", {}))
    vw = dict(stage1_manifest.get("vel_window", {}))

    gen_preset = args.gen_preset or str(stage1_args.get("gen_preset", "simple_snr"))
    Kmax = int(args.Kmax if args.Kmax is not None else stage1_args.get("Kmax", 6))
    v_half = (
        float(args.v_half_kms)
        if args.v_half_kms is not None
        else vw.get("v_half_width_kms", stage1_args.get("v_half_kms", 80.0))
    )
    v_center = float(
        args.v_center_kms
        if args.v_center_kms is not None
        else vw.get("v_center_kms", stage1_args.get("v_center_kms", 0.0))
    )
    kernel_size = int(
        args.kernel_size
        if args.kernel_size is not None
        else stage1_manifest.get("kernel_size", stage1_args.get("kernel_size", 25))
    )

    cfg = build_aces_synth_cfg(
        Kmax=Kmax,
        v_center_kms=v_center,
        v_half_width_kms=None if v_half is None else float(v_half),
        full_axis=v_half is None and not vw,
        gen_preset=gen_preset,
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

    model = HeatmapCountNet(
        heatmap,
        width=args.width,
        n_blocks=args.n_blocks,
        k_input=args.k_input,
        freeze_heatmap=args.freeze_heatmap,
        kernel_size=kernel_size,
    )

    history: dict = {}
    train_heatmap_count(
        model,
        train_loader,
        val_loader,
        device=args.device,
        lr=args.lr,
        epochs=args.epochs,
        log_every=args.log_every,
        Kmax=Kmax,
        use_scheduler=args.scheduler,
        history=history,
    )

    k_head = eval_heatmap_count_k(model, val_loader, device=args.device, Kmax=Kmax)
    decode = dict(stage1_manifest.get("decode", {}))
    height = float(decode.get("height", 0.35))
    prominence = float(decode.get("prominence", 0.15))
    min_sep = float(decode.get("min_sep_kms", 4.0))
    label_sigma = float(stage1_manifest.get("label_sigma_kms", 1.0))
    k_decode = eval_center_heatmap_k_decode(
        model.heatmap,
        val_loader,
        v_axis,
        device=args.device,
        label_sigma_kms=label_sigma,
        height=height,
        prominence=prominence,
        min_sep_kms=min_sep,
        Kmax=Kmax,
    )
    print(
        f"final val: K_head MAE={k_head['k_mae']:.4f} exact={k_head['k_exact_frac']:.3f}  |  "
        f"peak_decode MAE={k_decode['k_mae']:.4f} exact={k_decode['k_exact_frac']:.3f}",
        flush=True,
    )

    cw = float(channel_width_kms(cfg))
    manifest = {
        "script": str(_SCRIPT),
        "variant": "heatmap_count",
        "stage1_run_dir": _path_rel_to_repo(stage1_dir, _REPO),
        "cfg": cfg,
        "vel_window": {
            "n_channels": int(cfg["n_channels"]),
            "channel_width_kms": cw,
            "vrange": list(cfg["vrange"]),
            "v_center_kms": float(v_center),
            "v_half_width_kms": None if v_half is None else float(v_half),
        },
        "k_input": args.k_input,
        "freeze_heatmap": bool(args.freeze_heatmap),
        "kernel_size": int(kernel_size),
        "decode_stage1": decode,
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "final_val_k_head": k_head,
        "final_val_k_decode": k_decode,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    torch.save(model.state_dict(), run_dir / "heatmap_count_net.pt")

    if history.get("epoch"):
        fig, ax = plt.subplots(figsize=(5.5, 3.2))
        ep = history["epoch"]
        ax.plot(ep, history["train_loss_epoch"], "o-", label="train SmoothL1")
        ax.plot(ep, history["val_K_MAE"], "s-", label="val K MAE (head)")
        ax.set_xlabel("epoch")
        ax.set_title(f"ACES Heatmap->K ({args.k_input}, freeze={args.freeze_heatmap})")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(run_dir / "curves.png", dpi=120)
        plt.close(fig)

    print(f"Wrote run artifacts to {run_dir}")


if __name__ == "__main__":
    main()
