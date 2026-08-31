#!/usr/bin/env python
"""
Fine-tune CountNet1DDeep on mixed synthetic (Scouse-aligned) + real Scouse labels.

Run from repo root:
  python experiments/MOPRA_Count/build_scouse_cache.py

  python experiments/MOPRA_Count/run_finetune_scouse.py \\
    --init-run-dir experiments/MOPRA_Count/runs/mopra_count_<ts>_scouse_smooth60 \\
    --tag scouse_ft_v1

If init checkpoint n_channels differs from scouse_dat axis (250), real spectra are
velocity-resampled to match the init model axis.
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

from spectackle.data.mopra_finetune_dataset import make_mopra_finetune_loaders  ### noqa: E402
from spectackle.config import set_cpu_safety  ### noqa: E402
from spectackle.data.mopra_generator import (  ### noqa: E402
    MOPRA_CUBE_SMOOTH60,
    build_mopra_synth_cfg,
)
from spectackle.data.mopra_resample import resample_spec_batch, velocity_axis_kms  ### noqa: E402
from spectackle.data.mopra_scouse_labels import build_scouse_labeled_cache, load_scouse_labeled_cache  ### noqa: E402
from spectackle.models import CountNet1DDeep  ### noqa: E402
from spectackle.plotting import collect_count_predictions_b, mae_by_true_k  ### noqa: E402
from spectackle.training import train_scheme_b  ### noqa: E402


def _default_run_dir() -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    return _SCRIPT.parent / "runs" / f"mopra_count_{ts}"


def _load_init_model(init_run_dir: Path, *, device: str) -> tuple[CountNet1DDeep, dict]:
    manifest = json.loads((init_run_dir / "manifest.json").read_text(encoding="utf-8"))
    args = manifest["args"]
    model = CountNet1DDeep(width=int(args["width"]), n_blocks=int(args["n_blocks"]))
    state = torch.load(init_run_dir / "count_net.pt", map_location="cpu")
    model.load_state_dict(state)
    model.to(device)
    return model, manifest


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


def _resample_cache_to_axis(cache_path: Path, target_n_channels: int, ref_cube: Path) -> Path:
    """Write resampled NPZ when init model axis != smooth60 cache."""
    loaded = load_scouse_labeled_cache(cache_path)
    ar = loaded["arrays"]
    n_ch_cache = int(ar["spec_norm"].shape[1])
    if n_ch_cache == target_n_channels:
        return cache_path

    out = cache_path.with_name(f"{cache_path.stem}_to{target_n_channels}ch.npz")
    if out.is_file() and out.with_suffix(".json").is_file():
        print(f"Using resampled cache {out} ({target_n_channels} ch)", flush=True)
        return out

    smooth60 = cache_path  ### meta has cube_path
    meta = loaded["meta"]
    cube_smooth = Path(meta.get("cube_path", str(_REPO / "data" / MOPRA_CUBE_SMOOTH60)))
    v_src = velocity_axis_kms(cube_smooth)
    v_tgt = velocity_axis_kms(ref_cube)

    ### Reconstruct raw-ish spectra from norm is lossy; re-read cube for resample.
    from spectral_cube import SpectralCube

    cube = SpectralCube.read(str(cube_smooth.resolve()), use_dask=False)
    arr = np.asarray(cube.filled(np.nan), dtype=np.float32)
    if arr.ndim == 4:
        arr = arr[0]
    nv, _, _ = arr.shape
    n = ar["spec_norm"].shape[0]
    spec_tgt = np.zeros((n, target_n_channels), dtype=np.float32)
    vm_tgt = np.zeros((n, target_n_channels), dtype=np.float32)

    from spectackle.data.mopra_preprocess import prepare_mopra_input

    for i in range(n):
        yi, xi = int(ar["yi"][i]), int(ar["xi"][i])
        raw = arr[:, yi, xi].astype(np.float64)
        raw_rs = resample_spec_batch(raw, v_src, v_tgt)
        sn, vm = prepare_mopra_input(raw_rs)
        spec_tgt[i] = sn
        vm_tgt[i] = vm

    np.savez_compressed(
        out,
        spec_norm=spec_tgt,
        valid_mask=vm_tgt,
        K_true=ar["K_true"],
        l=ar["l"],
        b=ar["b"],
        yi=ar["yi"],
        xi=ar["xi"],
        split=ar["split"],
        n_channels=np.array([target_n_channels], dtype=np.int32),
    )
    meta_out = dict(meta)
    meta_out["resampled_from"] = str(cache_path)
    meta_out["n_channels"] = target_n_channels
    meta_out["ref_cube"] = str(ref_cube.resolve())
    out.with_suffix(".json").write_text(json.dumps(meta_out, indent=2), encoding="utf-8")
    print(f"Wrote resampled cache {out} ({n_ch_cache}->{target_n_channels} ch)", flush=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune MOPRA K model on Scouse labels + synth mix.")
    parser.add_argument("--init-run-dir", type=Path, required=True, help="Pretrained run (count_net.pt).")
    parser.add_argument("--cache", type=Path, default=_SCRIPT.parent / "cache" / "scouse_labeled_smooth60.npz")
    parser.add_argument("--dat", type=Path, default=_REPO / "data" / "final_fits_updated.dat")
    parser.add_argument("--cube", type=Path, default=_REPO / "data" / MOPRA_CUBE_SMOOTH60)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--real-frac", type=float, default=0.5, help="Fraction of train batches from real labels.")
    parser.add_argument("--n-synth-train", type=int, default=10_000)
    parser.add_argument("--bs-train", type=int, default=128)
    parser.add_argument("--bs-val", type=int, default=256)
    parser.add_argument("--Kmax", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--scheduler", action="store_true")
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--cell-deg", type=float, default=0.08)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--tag", type=str, default="scouse_ft")
    parser.add_argument(
        "--gen-preset",
        choices=("scouse_dat", "scouse_smooth60", "default"),
        default="scouse_dat",
        help="Synthetic mix preset (scouse_dat: smooth60 axis + Scouse K acceptance).",
    )
    parser.add_argument(
        "--match-init-axis",
        action="store_true",
        default=True,
        help="Resample real cache to init model n_channels when they differ (default: on).",
    )
    parser.add_argument(
        "--no-match-init-axis",
        action="store_false",
        dest="match_init_axis",
        help="Require cache n_channels == cfg (train fresh on smooth60 axis).",
    )
    args = parser.parse_args()

    set_cpu_safety(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    init_dir = args.init_run_dir.resolve()
    model, init_manifest = _load_init_model(init_dir, device=args.device)
    init_n_ch = int(init_manifest["cfg"]["n_channels"])
    init_cube = Path(init_manifest["cfg"].get("mopra_meta", {}).get("cube_path", _REPO / "data" / "CMZ_3mm_HNCO.fits"))

    _ensure_cache(
        args.cache.resolve(),
        dat=args.dat,
        cube=args.cube,
        val_frac=args.val_frac,
        cell_deg=args.cell_deg,
        seed=args.seed,
    )
    cache_path = args.cache.resolve()
    if args.match_init_axis:
        cache_path = _resample_cache_to_axis(cache_path, init_n_ch, init_cube).resolve()

    cfg = build_mopra_synth_cfg(
        repo_root=_REPO,
        max_components=args.Kmax,
        gen_preset=args.gen_preset,
        noise_calibration_cube=args.cube,
    )
    if args.match_init_axis and init_n_ch != cfg["n_channels"]:
        cache_path = _resample_cache_to_axis(cache_path, init_n_ch, init_cube).resolve()
        cfg = build_mopra_synth_cfg(
            repo_root=_REPO,
            axis_cube=init_cube,
            max_components=args.Kmax,
            gen_preset=args.gen_preset,
            noise_calibration_cube=args.cube,
        )
        cfg.setdefault("mopra_meta", {})["finetune_resampled_cache"] = str(cache_path)
    elif cfg["n_channels"] != load_scouse_labeled_cache(cache_path)["arrays"]["spec_norm"].shape[1]:
        raise ValueError(
            f"Cache n_ch mismatch cfg={cfg['n_channels']}; use --match-init-axis or align init run."
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
        f"Fine-tune: real_train={split_info['n_real_train']} real_val={split_info['n_real_val']} "
        f"real_frac={args.real_frac} synth={args.n_synth_train} n_ch={cfg['n_channels']}",
        flush=True,
    )

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
        history=history,
    )

    y_true, y_pred = collect_count_predictions_b(model, val_loader, device=args.device, Kmax=args.Kmax)
    scouse_mae = float(np.abs(y_pred - y_true).mean())
    mae_k = mae_by_true_k(y_true, y_pred, args.Kmax)

    init_args = json.loads((init_dir / "manifest.json").read_text(encoding="utf-8"))["args"]
    manifest = {
        "script": str(_SCRIPT),
        "init_run_dir": str(init_dir),
        "cache_path": str(cache_path),
        "cfg": cfg,
        "split_info": split_info,
        "args": {
            **{k: init_args[k] for k in ("width", "n_blocks", "Kmax") if k in init_args},
            **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        },
        "scouse_val_K_MAE": scouse_mae,
        "mae_by_K": {str(k): v for k, v in mae_k.items()},
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    torch.save(model.state_dict(), run_dir / "count_net.pt")

    if history.get("epoch"):
        fig, ax = plt.subplots(figsize=(6, 3.2))
        ep = history["epoch"]
        ax.plot(ep, history["train_loss_epoch"], "o-", label="train")
        ax.plot(ep, history["val_K_MAE"], "s-", label="Scouse val MAE")
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss / MAE")
        ax.set_title(f"Fine-tune (Scouse val MAE={scouse_mae:.3f})")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(run_dir / "curves.png", dpi=120)
        plt.close(fig)

    print(f"Scouse spatial-val K_MAE {scouse_mae:.4f}")
    print(f"Wrote run artifacts to {run_dir}")


if __name__ == "__main__":
    main()
