#!/usr/bin/env python
"""
Gallery of synthetic validation spectra for a trained center-heatmap run.

Shows spectrum (norm), predicted P(center), true planted centers, and decoded peaks.
Samples are drawn from the same val seed space as training (base_seed=10_000_000).

Example:
  python experiments/MOPRA_Count/plots/plot_heatmap_val_gallery.py \\
    --run-dir experiments/MOPRA_Count/runs/mopra_heatmap_<ts>_<tag>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPT = Path(__file__).resolve()
_MOPRA = _SCRIPT.parent.parent
_REPO = _MOPRA.parents[1]
sys.path.insert(0, str(_REPO / "src"))

import matplotlib.pyplot as plt
import numpy as np
import torch

from spectackle.data.generator import _make_v_axis
from spectackle.data.mopra_dataset import MOPRASpectraDataset
from spectackle.models import CenterHeatmapNet1DDeep
from spectackle.models.center_heatmap_decode import decode_centers_from_heatmap


def _load_model(run_dir: Path, manifest: dict, *, device: str) -> CenterHeatmapNet1DDeep:
    args = dict(manifest.get("args", {}))
    coord = None
    if manifest.get("coord", {}).get("enabled"):
        v_scale = float(manifest["coord"].get("v_scale_kms", 100.0))
        v_axis = _make_v_axis(manifest["cfg"]).astype(np.float32)
        coord = v_axis / v_scale
    model = CenterHeatmapNet1DDeep(
        width=int(args.get("width", 96)),
        n_blocks=int(args.get("n_blocks", 6)),
        coord=coord,
    )
    state = torch.load(run_dir / "center_heatmap_net.pt", map_location="cpu")
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def _pick_indices(ks: np.ndarray, *, n_each: int, seed: int, k_values: list[int]) -> list[int]:
    rng = np.random.default_rng(seed)
    picked: list[int] = []
    for k in k_values:
        pool = np.flatnonzero(ks == k)
        if pool.size == 0:
            continue
        take = min(int(n_each), int(pool.size))
        picked.extend(rng.choice(pool, size=take, replace=False).tolist())
    return picked


@torch.no_grad()
def plot_val_gallery(
    *,
    run_dir: Path,
    out: Path,
    n_each: int = 3,
    k_values: list[int] | None = None,
    seed: int = 0,
    device: str = "cpu",
    n_pool: int = 2000,
    vel_range: tuple[float, float] | None = None,
) -> Path:
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    cfg = manifest["cfg"]
    args = dict(manifest.get("args", {}))
    dec = dict(manifest.get("decode", {}))
    height = float(dec.get("height", 0.35))
    prom = float(dec.get("prominence", 0.15))
    min_sep = float(dec.get("min_sep_kms", 4.0))
    Kmax = int(dec.get("Kmax", args.get("Kmax", 6)))
    if k_values is None:
        k_values = [0, 1, 2, 3, 4]

    ### Match make_mopra_loaders val split.
    n_val = int(args.get("n_val", n_pool))
    ds = MOPRASpectraDataset(cfg, n_samples=min(n_pool, n_val), base_seed=10_000_000)
    v_axis = _make_v_axis(cfg).astype(np.float64)
    model = _load_model(run_dir, manifest, device=device)

    ks = np.zeros(len(ds), dtype=np.int64)
    for i in range(len(ds)):
        ks[i] = int(ds[i]["K_true"].item())
    idxs = _pick_indices(ks, n_each=n_each, seed=seed, k_values=k_values)
    if not idxs:
        raise RuntimeError("No validation samples selected for gallery.")

    n = len(idxs)
    n_col = min(3, n)
    n_row = int(np.ceil(n / n_col))
    fig, axes = plt.subplots(n_row, n_col, figsize=(4.2 * n_col, 2.4 * n_row), sharex=True)
    axes = np.atleast_1d(axes).ravel()

    for panel_i, idx in enumerate(idxs):
        ax = axes[panel_i]
        batch = ds[idx]
        x = batch["spec_norm"].unsqueeze(0).to(device)
        mask = batch["valid_mask"].unsqueeze(0).to(device)
        prob = torch.sigmoid(model(x, mask)).cpu().numpy()[0]
        spec = batch["spec_norm"].numpy()
        valid = batch["valid_mask"].numpy() > 0.5
        v_true = batch["component_v_kms"].numpy()
        v_ok = batch["component_valid"].numpy() > 0.5
        k_true = int(batch["K_true"].item())

        k_hat, peak_idx = decode_centers_from_heatmap(
            prob,
            v_axis,
            valid_mask=valid,
            height=height,
            prominence=prom,
            min_sep_kms=min_sep,
            Kmax=Kmax,
        )

        m = valid
        ax.plot(v_axis[m], spec[m], color="0.35", lw=0.9, label="spectrum")
        ax2 = ax.twinx()
        ax2.plot(v_axis[m], prob[m], color="tab:red", lw=1.0, alpha=0.85, label="P(center)")
        ax2.set_ylim(-0.05, 1.05)
        for vc in v_true[v_ok]:
            ax.axvline(float(vc), color="tab:blue", ls="--", lw=0.9, alpha=0.9)
        for pi in peak_idx:
            ax.axvline(float(v_axis[int(pi)]), color="tab:orange", ls=":", lw=1.0, alpha=0.95)
        if vel_range is not None:
            ax.set_xlim(float(vel_range[0]), float(vel_range[1]))
        ax.set_title(f"idx={idx}  K_true={k_true}  K_dec={k_hat}", fontsize=9)
        ax.tick_params(labelsize=8)
        ax2.tick_params(labelsize=8, colors="tab:red")

    for j in range(n, len(axes)):
        axes[j].axis("off")
    for ax in axes[max(0, n - n_col) : n]:
        ax.set_xlabel("velocity (km/s)", fontsize=9)

    tag = str(args.get("tag", run_dir.name))
    fig.suptitle(
        f"Heatmap val gallery ({tag})\n"
        f"blue dashed = true centers, orange dotted = decoded peaks, red = P(center)",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)

    meta = {
        "run_dir": str(run_dir),
        "out": str(out),
        "indices": idxs,
        "k_true": [int(ks[i]) for i in idxs],
        "decode": {"height": height, "prominence": prom, "min_sep_kms": min_sep},
        "n_each": n_each,
        "k_values": k_values,
        "seed": seed,
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Validation-set heatmap spectrum gallery.")
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Heatmap train folder with center_heatmap_net.pt + manifest.json.",
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--n-each", type=int, default=3, help="Panels per K_true bin.")
    parser.add_argument("--k-values", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-pool", type=int, default=2000)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--vel-range", type=float, nargs=2, default=None, metavar=("VLO", "VHI"))
    args = parser.parse_args()

    out = args.out
    if out is None:
        out = _MOPRA / "figures" / f"{args.run_dir.name}_val_gallery.png"

    path = plot_val_gallery(
        run_dir=args.run_dir,
        out=out,
        n_each=args.n_each,
        k_values=list(args.k_values),
        seed=args.seed,
        device=args.device,
        n_pool=args.n_pool,
        vel_range=tuple(args.vel_range) if args.vel_range is not None else None,
    )
    print(f"Wrote {path}")
    print(f"Wrote {path.with_suffix('.json')}")


if __name__ == "__main__":
    main()
