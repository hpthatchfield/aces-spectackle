#!/usr/bin/env python
"""
Per-K success / failure gallery on ACES heatmap->K synth validation.

For each K_true: exact matches, under-counts, over-counts (when present).
Uses the same val seed space as training (ACESSpectraDataset base_seed=10_000_000).

  python experiments/ACES_Heatmap/plots/plot_val_by_k_gallery.py \\
    --run-dir experiments/ACES_Heatmap/runs/<heatmap_count_run>
"""
from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

_SCRIPT = Path(__file__).resolve()
_ACES = _SCRIPT.parents[1]
_REPO = _ACES.parents[1]
sys.path.insert(0, str(_REPO / "src"))

from spectackle.data.aces_dataset import ACESSpectraDataset  ### noqa: E402
from spectackle.data.generator import _make_v_axis  ### noqa: E402
from spectackle.models import CenterHeatmapNet1DDeep, HeatmapCountNet  ### noqa: E402
from spectackle.models.center_heatmap_decode import decode_centers_from_heatmap  ### noqa: E402

COL_SPEC = "0.35"
COL_OK = "#2E7D32"
COL_FAIL = "#C62828"
COL_TRUE = "#1565C0"
COL_PRED = "#F28E2B"


def _load_heatmap_count(run_dir: Path, *, device: str) -> tuple[HeatmapCountNet, dict, np.ndarray]:
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("variant") != "heatmap_count":
        raise ValueError(f"Expected heatmap_count run, got {manifest.get('variant')!r}")
    stage1_dir = Path(manifest["stage1_run_dir"])
    if not stage1_dir.is_absolute():
        stage1_dir = (_REPO / stage1_dir).resolve()
    s1 = json.loads((stage1_dir / "manifest.json").read_text(encoding="utf-8"))
    s1_args = dict(s1.get("args", {}))
    cfg = manifest["cfg"]
    kernel_size = int(manifest.get("kernel_size", s1.get("kernel_size", 25)))
    coord = None
    if s1.get("coord", {}).get("enabled"):
        v_scale = float(s1["coord"].get("v_scale_kms", 100.0))
        coord = _make_v_axis(cfg).astype(np.float32) / v_scale
    heatmap = CenterHeatmapNet1DDeep(
        width=int(s1_args.get("width", 96)),
        n_blocks=int(s1_args.get("n_blocks", 6)),
        coord=coord,
        kernel_size=kernel_size,
    )
    heatmap.load_state_dict(torch.load(stage1_dir / "center_heatmap_net.pt", map_location="cpu"))
    args = dict(manifest.get("args", {}))
    model = HeatmapCountNet(
        heatmap,
        width=int(args.get("width", 96)),
        n_blocks=int(args.get("n_blocks", 6)),
        k_input=str(manifest.get("k_input", args.get("k_input", "spec_p"))),
        freeze_heatmap=True,
        kernel_size=kernel_size,
    )
    model.load_state_dict(torch.load(run_dir / "heatmap_count_net.pt", map_location="cpu"))
    model.to(device)
    model.eval()
    v_axis = _make_v_axis(cfg).astype(np.float64)
    return model, manifest, v_axis


@torch.no_grad()
def _eval_pool(
    model: HeatmapCountNet,
    ds: ACESSpectraDataset,
    *,
    device: str,
    v_axis: np.ndarray,
    Kmax: int,
    height: float,
    prominence: float,
    min_sep_kms: float,
    batch_size: int = 64,
) -> dict:
    """Run K head + top-K peak centers over the whole val pool."""
    n = len(ds)
    k_true = np.zeros(n, dtype=np.int64)
    k_cont = np.zeros(n, dtype=np.float32)
    k_pred = np.zeros(n, dtype=np.int64)
    specs = []
    masks = []
    v_true = []
    v_ok = []
    v_pred = np.full((n, Kmax), np.nan, dtype=np.float32)

    for b0 in range(0, n, batch_size):
        b1 = min(b0 + batch_size, n)
        batch_items = [ds[i] for i in range(b0, b1)]
        x = torch.stack([it["spec_norm"] for it in batch_items]).to(device)
        m = torch.stack([it["valid_mask"] for it in batch_items]).to(device)
        k_hat = model(x, m).cpu().numpy().astype(np.float32)
        prob = torch.sigmoid(model.heatmap_logits(x, m)).cpu().numpy()
        for j, it in enumerate(batch_items):
            i = b0 + j
            kt = int(it["K_true"].item())
            kc = float(k_hat[j])
            kp = int(np.clip(np.round(kc), 0, Kmax))
            k_true[i] = kt
            k_cont[i] = kc
            k_pred[i] = kp
            specs.append(it["spec_norm"].numpy())
            masks.append(it["valid_mask"].numpy())
            v_true.append(it["component_v_kms"].numpy())
            v_ok.append(it["component_valid"].numpy())
            if kp <= 0:
                continue
            _kd, peak_idx = decode_centers_from_heatmap(
                prob[j], v_axis, valid_mask=masks[-1],
                height=height, prominence=prominence,
                min_sep_kms=min_sep_kms, Kmax=kp,
            )
            for t, ix in enumerate(peak_idx.tolist()):
                if t >= Kmax:
                    break
                v_pred[i, t] = float(v_axis[int(ix)])

    return {
        "k_true": k_true,
        "k_pred": k_pred,
        "k_cont": k_cont,
        "specs": np.stack(specs),
        "masks": np.stack(masks),
        "v_true": np.stack(v_true),
        "v_ok": np.stack(v_ok),
        "v_pred": v_pred,
    }


def _pick_by_k(
    k_true: np.ndarray,
    k_pred: np.ndarray,
    k_cont: np.ndarray,
    *,
    k: int,
    n_each: int,
    seed: int,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed + int(k) * 17)
    pool = np.flatnonzero(k_true == k)
    exact = pool[k_pred[pool] == k]
    under = pool[k_pred[pool] < k]
    over = pool[k_pred[pool] > k]

    def _sample(idxs: np.ndarray, prefer_resid: bool) -> np.ndarray:
        if idxs.size == 0:
            return np.array([], dtype=int)
        if prefer_resid:
            resid = np.abs(k_cont[idxs] - float(k))
            order = np.argsort(resid)
            idxs = idxs[order]
            pool_n = idxs[: max(n_each * 3, n_each)]
        else:
            ### Worst |dK| first for failures.
            err = np.abs(k_pred[idxs].astype(np.int64) - k)
            order = np.argsort(-err)
            idxs = idxs[order]
            pool_n = idxs[: max(n_each * 3, n_each)]
        take = min(n_each, pool_n.size)
        return np.sort(rng.choice(pool_n, size=take, replace=False))

    return {
        "exact": _sample(exact, prefer_resid=True),
        "under": _sample(under, prefer_resid=False),
        "over": _sample(over, prefer_resid=False),
        "n_pool": int(pool.size),
        "n_exact": int(exact.size),
        "n_under": int(under.size),
        "n_over": int(over.size),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="ACES heatmap->K per-K val success/failure gallery.")
    parser.add_argument("--run-dir", type=Path, required=True, help="heatmap_count run directory")
    parser.add_argument("--n-pool", type=int, default=2000, help="Val spectra to score (from val seed space).")
    parser.add_argument("--n-each", type=int, default=3, help="Examples per K / outcome.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    model, manifest, v_axis = _load_heatmap_count(run_dir, device=args.device)
    cfg = manifest["cfg"]
    Kmax = int(manifest.get("decode_stage1", {}).get("Kmax", cfg.get("max_components", 6)))
    dec = dict(manifest.get("decode_stage1") or {})
    height = float(dec.get("height", 0.25))
    prom = float(dec.get("prominence", 0.08))
    min_sep = float(dec.get("min_sep_kms", 4.0))

    n_pool = min(int(args.n_pool), int(manifest.get("args", {}).get("n_val", args.n_pool)))
    ds = ACESSpectraDataset(cfg, n_samples=n_pool, base_seed=10_000_000)
    print(f"Scoring {n_pool} val spectra...", flush=True)
    data = _eval_pool(
        model, ds, device=args.device, v_axis=v_axis, Kmax=Kmax,
        height=height, prominence=prom, min_sep_kms=min_sep,
    )
    v_pred = data["v_pred"]

    k_values = list(range(0, Kmax + 1))
    picks = {
        k: _pick_by_k(
            data["k_true"], data["k_pred"], data["k_cont"], k=k, n_each=args.n_each, seed=args.seed
        )
        for k in k_values
    }

    ### Layout: rows = K_true; within each K up to 3 outcome bands if nonempty.
    row_specs: list[tuple[str, np.ndarray, bool]] = []
    for k in k_values:
        p = picks[k]
        if p["n_pool"] == 0:
            continue
        label_base = f"K_true={k} (n={p['n_pool']}, exact={p['n_exact']})"
        if p["exact"].size:
            row_specs.append((f"{label_base}  exact", p["exact"], True))
        if p["under"].size:
            row_specs.append((f"K_true={k} under", p["under"], False))
        if p["over"].size:
            row_specs.append((f"K_true={k} over", p["over"], False))

    if not row_specs:
        raise RuntimeError("No examples selected.")

    n_col = int(args.n_each)
    n_row = len(row_specs)
    fig, axes = plt.subplots(n_row, n_col, figsize=(3.5 * n_col, 2.15 * n_row), squeeze=False)
    for r, (label, idxs, is_ok) in enumerate(row_specs):
        for c in range(n_col):
            ax = axes[r, c]
            if c >= idxs.size:
                ax.axis("off")
                continue
            i = int(idxs[c])
            spec = data["specs"][i]
            valid = data["masks"][i] > 0.5
            ax.plot(v_axis[valid], spec[valid], color=COL_SPEC, lw=0.8)
            for vv, ok in zip(data["v_true"][i], data["v_ok"][i]):
                if ok > 0.5 and np.isfinite(vv):
                    ax.axvline(float(vv), color=COL_TRUE, ls=":", lw=0.9)
            for vv in v_pred[i]:
                if np.isfinite(vv):
                    ax.axvline(float(vv), color=COL_PRED, ls="--", lw=0.9)
            title_col = COL_OK if is_ok else COL_FAIL
            ax.set_title(
                f"K {data['k_true'][i]}->{data['k_pred'][i]}  "
                f"Khat={data['k_cont'][i]:.2f}",
                fontsize=8,
                color=title_col,
            )
            if c == 0:
                ax.set_ylabel(label, fontsize=7)
            if r == n_row - 1:
                ax.set_xlabel("v (km/s)", fontsize=8)
    fig.suptitle(
        f"ACES heatmap->K val gallery ({run_dir.name})\n"
        f"blue dotted=true centers, orange dashed=ML peaks  |  n_pool={n_pool}",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = args.out or (_ACES / "figures" / "failure_spectra" / f"{run_dir.name}_val_by_k.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)

    summary = {
        str(k): {
            "n_pool": picks[k]["n_pool"],
            "n_exact": picks[k]["n_exact"],
            "n_under": picks[k]["n_under"],
            "n_over": picks[k]["n_over"],
            "exact_frac": picks[k]["n_exact"] / max(1, picks[k]["n_pool"]),
        }
        for k in k_values
    }
    meta = out.with_suffix(".json")
    meta.write_text(json.dumps({"run_dir": str(run_dir), "n_pool": n_pool, "by_k": summary}, indent=2), encoding="utf-8")
    print(f"Wrote {out}")
    print(f"Wrote {meta}")
    for k in k_values:
        s = summary[str(k)]
        if s["n_pool"] == 0:
            continue
        print(
            f"  K={k}: n={s['n_pool']} exact={s['n_exact']} "
            f"under={s['n_under']} over={s['n_over']} ({100 * s['exact_frac']:.0f}% exact)"
        )


if __name__ == "__main__":
    main()
