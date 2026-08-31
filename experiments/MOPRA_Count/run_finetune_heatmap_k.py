#!/usr/bin/env python
"""
Fine-tune HeatmapCountNet (Stage-2 K head) on mixed real Scouse labels + synth.

Mirrors run_finetune_scouse.py for the heatmap->K path. Default: freeze Stage-1 heatmap,
adapt only the K head to the spatial train split of labeled pixels (all K values present
in the Scouse .dat via the existing sky-cell split).

Run from repo root:
  python experiments/MOPRA_Count/build_scouse_cache.py

  python experiments/MOPRA_Count/run_finetune_heatmap_k.py \\
    --init-run-dir experiments/MOPRA_Count/runs/mopra_heatmap_k_<ts>_hm_k_simple_k6_20k \\
    --tag hm_k_scouse_ft_v1 --epochs 10 --real-frac 0.5 --scheduler
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
from spectackle.data.mopra_finetune_dataset import make_mopra_finetune_loaders  ### noqa: E402
from spectackle.data.mopra_generator import MOPRA_CUBE_SMOOTH60, build_mopra_synth_cfg  ### noqa: E402
from spectackle.data.mopra_scouse_labels import build_scouse_labeled_cache, load_scouse_labeled_cache  ### noqa: E402
from spectackle.models import CenterHeatmapNet1DDeep, HeatmapCountNet  ### noqa: E402
from spectackle.training import eval_heatmap_count_k, train_heatmap_count  ### noqa: E402


def _default_run_dir() -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    return _SCRIPT.parent / "runs" / f"mopra_heatmap_k_{ts}"


def _ensure_cache(
    cache_path: Path,
    *,
    dat: Path,
    cube: Path,
    val_frac: float,
    cell_deg: float,
    seed: int,
) -> None:
    if cache_path.is_file() and cache_path.with_suffix(".json").is_file():
        return
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Building labeled cache -> {cache_path}", flush=True)
    build_scouse_labeled_cache(
        dat_path=dat,
        cube_path=cube,
        out_path=cache_path,
        val_frac=val_frac,
        cell_deg=cell_deg,
        seed=seed,
    )


def _load_heatmap_count(init_run_dir: Path, *, device: str, freeze_heatmap: bool) -> tuple[HeatmapCountNet, dict]:
    """Rebuild HeatmapCountNet from a heatmap_count run (or stage-2) checkpoint."""
    manifest = json.loads((init_run_dir / "manifest.json").read_text(encoding="utf-8"))
    variant = str(manifest.get("variant", ""))
    if variant != "heatmap_count":
        raise ValueError(
            f"Expected heatmap_count init run, got variant={variant!r} in {init_run_dir}"
        )

    stage1_dir = Path(manifest["stage1_run_dir"])
    stage1_manifest = json.loads((stage1_dir / "manifest.json").read_text(encoding="utf-8"))
    s1_args = dict(stage1_manifest.get("args", {}))
    cfg = stage1_manifest["cfg"]
    coord = None
    if stage1_manifest.get("coord", {}).get("enabled"):
        v_scale = float(stage1_manifest["coord"].get("v_scale_kms", 100.0))
        coord = _make_v_axis(cfg).astype(np.float32) / v_scale
    heatmap = CenterHeatmapNet1DDeep(
        width=int(s1_args.get("width", 96)),
        n_blocks=int(s1_args.get("n_blocks", 6)),
        coord=coord,
    )
    ### Prefer joint FT weights; fall back to stage-1 file if somehow missing from state.
    args_m = dict(manifest.get("args", {}))
    model = HeatmapCountNet(
        heatmap,
        width=int(args_m.get("width", 96)),
        n_blocks=int(args_m.get("n_blocks", 6)),
        k_input=str(manifest.get("k_input", args_m.get("k_input", "spec_p"))),
        freeze_heatmap=freeze_heatmap,
    )
    ckpt = init_run_dir / "heatmap_count_net.pt"
    if not ckpt.is_file():
        raise FileNotFoundError(f"Missing {ckpt}")
    state = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(state)
    model.set_freeze_heatmap(freeze_heatmap)
    model.to(device)
    return model, manifest


def _k_hist_from_cache(cache_path: Path) -> dict:
    ar = load_scouse_labeled_cache(cache_path)["arrays"]
    out = {}
    for split_name, code in (("train", 0), ("val", 1)):
        mask = ar["split"] == code
        k = ar["K_true"][mask].reshape(-1).astype(int)
        hist = {str(int(v)): int(c) for v, c in zip(*np.unique(k, return_counts=True))}
        out[split_name] = {"n": int(mask.sum()), "K_hist": hist}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune heatmap->K head on Scouse + synth mix.")
    parser.add_argument(
        "--init-run-dir",
        type=Path,
        required=True,
        help="Synth-trained heatmap_count run (heatmap_count_net.pt).",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=_SCRIPT.parent / "cache" / "scouse_labeled_smooth60.npz",
    )
    parser.add_argument("--dat", type=Path, default=_REPO / "data" / "final_fits_updated.dat")
    parser.add_argument("--cube", type=Path, default=_REPO / "data" / MOPRA_CUBE_SMOOTH60)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--real-frac", type=float, default=0.5)
    parser.add_argument("--n-synth-train", type=int, default=10_000)
    parser.add_argument("--bs-train", type=int, default=128)
    parser.add_argument("--bs-val", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--scheduler", action="store_true")
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--cell-deg", type=float, default=0.08)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--tag", type=str, default="hm_k_scouse_ft")
    parser.add_argument(
        "--freeze-heatmap",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Freeze Stage-1 heatmap during FT (default). Use --no-freeze-heatmap to adapt both.",
    )
    parser.add_argument(
        "--gen-preset",
        default=None,
        help="Synth mix preset. Default: inherit Stage-1 / init gen_preset (usually simple).",
    )
    parser.add_argument("--Kmax", type=int, default=None)
    args = parser.parse_args()

    set_cpu_safety(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    init_dir = args.init_run_dir.resolve()
    model, init_manifest = _load_heatmap_count(
        init_dir, device=args.device, freeze_heatmap=args.freeze_heatmap
    )
    init_args = dict(init_manifest.get("args", {}))
    stage1_dir = Path(init_manifest["stage1_run_dir"])
    stage1_manifest = json.loads((stage1_dir / "manifest.json").read_text(encoding="utf-8"))
    stage1_args = dict(stage1_manifest.get("args", {}))

    gen_preset = args.gen_preset or str(
        init_args.get("gen_preset") or stage1_args.get("gen_preset") or "simple"
    )
    Kmax = int(
        args.Kmax
        if args.Kmax is not None
        else init_args.get("Kmax")
        or stage1_args.get("Kmax")
        or init_manifest.get("decode_stage1", {}).get("Kmax", 6)
    )
    norm_mode = str(
        init_manifest.get("norm_mode")
        or stage1_manifest.get("norm_mode")
        or "zscore"
    )

    _ensure_cache(
        args.cache.resolve(),
        dat=args.dat,
        cube=args.cube,
        val_frac=args.val_frac,
        cell_deg=args.cell_deg,
        seed=args.seed,
    )
    cache_path = args.cache.resolve()
    k_split = _k_hist_from_cache(cache_path)
    print(
        f"Cache K coverage: train={k_split['train']['K_hist']}  val={k_split['val']['K_hist']}",
        flush=True,
    )

    cfg = build_mopra_synth_cfg(
        repo_root=_REPO,
        cube_path=args.cube,
        max_components=Kmax,
        gen_preset=gen_preset,
        noise_calibration_cube=args.cube,
    )
    cfg["norm_mode"] = norm_mode

    n_ch_cache = int(load_scouse_labeled_cache(cache_path)["arrays"]["spec_norm"].shape[1])
    if n_ch_cache != int(cfg["n_channels"]):
        raise ValueError(
            f"Cache n_ch={n_ch_cache} != cfg n_ch={cfg['n_channels']}. "
            "Heatmap->K FT expects smooth60 axis matching the Stage-1 run."
        )

    run_dir = args.run_dir if args.run_dir is not None else _default_run_dir()
    if args.tag:
        run_dir = run_dir.parent / f"{run_dir.name}_{args.tag}"
    run_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, split_info = make_mopra_finetune_loaders(
        cfg,
        cache_path,
        real_frac=args.real_frac,
        n_synth_train=args.n_synth_train,
        bs_train=args.bs_train,
        bs_val=args.bs_val,
        shuffle_seed=args.seed,
    )
    print(
        f"Fine-tune heatmap->K: real_train={split_info['n_real_train']} "
        f"real_val={split_info['n_real_val']} real_frac={args.real_frac} "
        f"synth={args.n_synth_train} gen={gen_preset} freeze_heatmap={args.freeze_heatmap}",
        flush=True,
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
    print(
        f"Scouse spatial-val K_head MAE={k_head['k_mae']:.4f} exact={k_head['k_exact_frac']:.3f}",
        flush=True,
    )

    cw = float(cfg["vrange"][1] - cfg["vrange"][0]) / max(1, int(cfg["n_channels"]) - 1)
    manifest = {
        "script": str(_SCRIPT),
        "variant": "heatmap_count",
        "stage1_run_dir": str(stage1_dir),
        "init_run_dir": str(init_dir),
        "cache_path": str(cache_path),
        "cfg": cfg,
        "norm_mode": norm_mode,
        "vel_window": {
            "n_channels": int(cfg["n_channels"]),
            "channel_width_kms": cw,
            "vrange": list(cfg["vrange"]),
        },
        "k_input": model.k_input,
        "freeze_heatmap": bool(args.freeze_heatmap),
        "decode_stage1": dict(
            init_manifest.get("decode_stage1") or stage1_manifest.get("decode") or {}
        ),
        "split_info": {**split_info, "K_hist_by_split": k_split},
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "final_val_k_head": k_head,
        "scouse_val_K_MAE": float(k_head["k_mae"]),
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    torch.save(model.state_dict(), run_dir / "heatmap_count_net.pt")

    if history.get("epoch"):
        fig, ax = plt.subplots(figsize=(6, 3.2))
        ep = history["epoch"]
        ax.plot(ep, history["train_loss_epoch"], "o-", label="train SmoothL1")
        ax.plot(ep, history["val_K_MAE"], "s-", label="Scouse spatial-val MAE")
        ax.set_xlabel("epoch")
        ax.set_title(
            f"Heatmap->K FT (val MAE={k_head['k_mae']:.3f}, freeze_hm={args.freeze_heatmap})"
        )
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(run_dir / "curves.png", dpi=120)
        plt.close(fig)

    print(f"Wrote run artifacts to {run_dir}")


if __name__ == "__main__":
    main()
