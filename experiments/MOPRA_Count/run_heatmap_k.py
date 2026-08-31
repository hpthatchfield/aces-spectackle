#!/usr/bin/env python
"""
Stage-2 K head on a pretrained MOPRA center heatmap (frozen Stage 1 by default).

Loads an existing heatmap run, trains a Scheme-B-style scalar K head on P(center) or
[spectrum ; P], and reports learned K MAE beside peak-decode K MAE from the same Stage 1.

Run from repo root:
  python experiments/MOPRA_Count/run_heatmap_k.py \\
    --heatmap-run-dir experiments/MOPRA_Count/runs/mopra_heatmap_<ts>_heatmap_simple_k6_20k \\
    --tag hm_k_simple_quick --n-train 2000 --n-val 500 --epochs 2

Full run (match Stage-1 simple_k6 dials):
  python experiments/MOPRA_Count/run_heatmap_k.py \\
    --heatmap-run-dir experiments/MOPRA_Count/runs/mopra_heatmap_<ts>_heatmap_simple_k6_20k \\
    --tag hm_k_simple_k6_20k --n-train 20000 --n-val 4000 --epochs 8 --scheduler
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
from spectackle.models import CenterHeatmapNet1DDeep, HeatmapCountNet  ### noqa: E402
from spectackle.models.center_heatmap_decode import eval_center_heatmap_k_decode  ### noqa: E402
from spectackle.training import eval_heatmap_count_k, train_heatmap_count  ### noqa: E402


def _default_run_dir() -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    return _SCRIPT.parent / "runs" / f"mopra_heatmap_k_{ts}"


def _load_heatmap_stage1(run_dir: Path, *, device: str) -> tuple[CenterHeatmapNet1DDeep, dict, np.ndarray]:
    """Rebuild Stage-1 net from manifest + center_heatmap_net.pt."""
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    args_m = dict(manifest.get("args", {}))
    cfg = manifest["cfg"]
    width = int(args_m.get("width", 96))
    n_blocks = int(args_m.get("n_blocks", 6))
    coord = None
    if manifest.get("coord", {}).get("enabled"):
        v_scale = float(manifest["coord"].get("v_scale_kms", 100.0))
        coord = _make_v_axis(cfg).astype(np.float32) / v_scale
    heatmap = CenterHeatmapNet1DDeep(
        width=width,
        n_blocks=n_blocks,
        coord=coord,
        kernel_size=int(manifest.get("kernel_size", args_m.get("kernel_size", 9))),
    )
    state = torch.load(run_dir / "center_heatmap_net.pt", map_location="cpu")
    heatmap.load_state_dict(state)
    heatmap.to(device)
    heatmap.eval()
    v_axis = _make_v_axis(cfg).astype(np.float32)
    return heatmap, manifest, v_axis


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Stage-2 K head on a frozen center heatmap.")
    parser.add_argument(
        "--heatmap-run-dir",
        type=Path,
        required=True,
        help="Existing mopra_heatmap_* run with center_heatmap_net.pt + manifest.json.",
    )
    parser.add_argument("--n-train", type=int, default=10_000)
    parser.add_argument("--n-val", type=int, default=2_000)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--bs-train", type=int, default=128)
    parser.add_argument("--bs-val", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--width", type=int, default=96, help="Stage-2 count trunk width.")
    parser.add_argument("--n-blocks", type=int, default=6, help="Stage-2 count trunk depth.")
    parser.add_argument(
        "--k-input",
        choices=("p", "spec_p"),
        default="spec_p",
        help="K-head input: P(center) alone, or [spectrum ; P].",
    )
    parser.add_argument(
        "--freeze-heatmap",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Freeze Stage-1 weights (default). Use --no-freeze-heatmap for joint fine-tune.",
    )
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
        help="Axis metadata cube (defaults match Stage-1 unless overridden).",
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
            "heatmap_realamp",
            "heatmap_realamp_snr5",
            "legacy",
        ),
        default=None,
        help="Generator preset. Default: inherit Stage-1 manifest args.gen_preset.",
    )
    parser.add_argument(
        "--noise-calibration-cube",
        type=Path,
        default=_REPO / "data" / "CMZ_3mm_HNCO_60.fits",
    )
    parser.add_argument(
        "--norm-mode",
        choices=NORM_MODES,
        default=None,
        help="Default: inherit Stage-1 norm_mode / args.",
    )
    parser.add_argument(
        "--Kmax",
        type=int,
        default=None,
        help="Default: inherit Stage-1 Kmax / decode.Kmax.",
    )
    args = parser.parse_args()

    set_cpu_safety(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    stage1_dir = args.heatmap_run_dir.resolve()
    if not (stage1_dir / "center_heatmap_net.pt").is_file():
        raise FileNotFoundError(f"Missing center_heatmap_net.pt in {stage1_dir}")

    heatmap, stage1_manifest, v_axis = _load_heatmap_stage1(stage1_dir, device=args.device)
    stage1_args = dict(stage1_manifest.get("args", {}))

    gen_preset = args.gen_preset or str(stage1_args.get("gen_preset", "simple"))
    Kmax = int(args.Kmax if args.Kmax is not None else stage1_args.get("Kmax", stage1_manifest.get("decode", {}).get("Kmax", 6)))
    norm_mode = str(
        args.norm_mode
        if args.norm_mode is not None
        else stage1_manifest.get("norm_mode", stage1_args.get("norm_mode", "zscore"))
    )

    cfg = build_mopra_synth_cfg(
        repo_root=_REPO,
        cube_path=args.cube,
        max_components=Kmax,
        gen_preset=gen_preset,
        noise_calibration_cube=args.noise_calibration_cube,
    )
    cfg["norm_mode"] = norm_mode

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
        norm_mode=norm_mode,
    )

    model = HeatmapCountNet(
        heatmap,
        width=args.width,
        n_blocks=args.n_blocks,
        k_input=args.k_input,
        freeze_heatmap=args.freeze_heatmap,
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
    label_sigma = float(stage1_manifest.get("label_sigma_kms", 4.0))
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

    cw = float(cfg["vrange"][1] - cfg["vrange"][0]) / max(1, int(cfg["n_channels"]) - 1)
    manifest = {
        "script": str(_SCRIPT),
        "variant": "heatmap_count",
        "stage1_run_dir": str(stage1_dir),
        "cfg": cfg,
        "norm_mode": norm_mode,
        "vel_window": {
            "n_channels": int(cfg["n_channels"]),
            "channel_width_kms": cw,
            "vrange": list(cfg["vrange"]),
        },
        "k_input": args.k_input,
        "freeze_heatmap": bool(args.freeze_heatmap),
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
        ax.set_title(f"Heatmap->K ({args.k_input}, freeze={args.freeze_heatmap})")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(run_dir / "curves.png", dpi=120)
        plt.close(fig)

    print(f"Wrote run artifacts to {run_dir}")


if __name__ == "__main__":
    main()
