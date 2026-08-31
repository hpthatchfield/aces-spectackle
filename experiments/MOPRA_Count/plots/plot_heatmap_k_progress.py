#!/usr/bin/env python
"""
Bar + by-K MAE comparison across saved Scouse stats JSONs.

Example:
  python \\
    experiments/MOPRA_Count/plots/plot_heatmap_k_progress.py \\
    --stats \\
      experiments/MOPRA_Count/figures/mopra_cmz_k_pred_simple_k6_20k_scouse_stats.json:B_simple \\
      experiments/MOPRA_Count/figures/mopra_cmz_k_pred_heatmap_simple_k6_20k_scouse_stats.json:hm_decode \\
      experiments/MOPRA_Count/figures/mopra_cmz_k_pred_hm_k_simple_k6_20k_scouse_stats.json:hm_k_head \\
    --out experiments/MOPRA_Count/figures/heatmap_k_progress_compare.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_SCRIPT = Path(__file__).resolve()
_REPO = _SCRIPT.parents[3]


def _parse_spec(s: str) -> tuple[Path, str]:
    if ":" in s:
        path_s, label = s.rsplit(":", 1)
        return Path(path_s), label
    p = Path(s)
    return p, p.stem.replace("mopra_cmz_k_pred_", "").replace("_scouse_stats", "")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Scouse K stats across runs.")
    parser.add_argument(
        "--stats",
        nargs="+",
        required=True,
        help="One or more path[:label] Scouse stats JSON files.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=_REPO / "experiments" / "MOPRA_Count" / "figures" / "heatmap_k_progress_compare.png",
    )
    parser.add_argument("--summary-json", type=Path, default=None)
    args = parser.parse_args()

    rows = []
    for spec in args.stats:
        path, label = _parse_spec(spec)
        d = json.loads(path.read_text(encoding="utf-8"))
        g = d["global"]
        by_k = d.get("by_k_true", {})
        rows.append(
            {
                "label": label,
                "path": str(path),
                "mae": float(g["mae"]),
                "exact": float(g.get("frac_exact", g.get("exact_match_frac", np.nan))),
                "frac_over": float(g.get("frac_over", np.nan)),
                "frac_under": float(g.get("frac_under", np.nan)),
                "by_k_mae": {k: float(v["mae"]) for k, v in by_k.items()},
            }
        )

    labels = [r["label"] for r in rows]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))

    ax = axes[0]
    ax.bar(x, [r["mae"] for r in rows], color="steelblue", edgecolor="k", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Scouse K MAE")
    ax.set_title("Global MAE (lower better)")
    for i, r in enumerate(rows):
        ax.text(i, r["mae"] + 0.01, f"{r['mae']:.3f}", ha="center", va="bottom", fontsize=8)

    ax = axes[1]
    k_keys = sorted({int(k) for r in rows for k in r["by_k_mae"]}, key=int)
    width = 0.8 / max(1, len(rows))
    for i, r in enumerate(rows):
        offsets = np.arange(len(k_keys)) + (i - 0.5 * (len(rows) - 1)) * width
        vals = [r["by_k_mae"].get(str(k), np.nan) for k in k_keys]
        ax.bar(offsets, vals, width=width * 0.95, label=r["label"], edgecolor="k", linewidth=0.3)
    ax.set_xticks(np.arange(len(k_keys)))
    ax.set_xticklabels([str(k) for k in k_keys])
    ax.set_xlabel("K_true")
    ax.set_ylabel("MAE")
    ax.set_title("MAE by K_true")
    ax.legend(fontsize=7, loc="upper left")

    fig.suptitle("Heatmap->K progress vs Scheme B / peak-decode", fontsize=11)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140)
    plt.close(fig)
    print(f"Wrote {args.out}", flush=True)

    summary = {
        "out": str(args.out.resolve()),
        "models": rows,
    }
    out_json = args.summary_json or args.out.with_suffix(".json")
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {out_json}", flush=True)
    for r in rows:
        print(
            f"  {r['label']}: MAE={r['mae']:.3f}  exact={r['exact']:.3f}  "
            f"over={r['frac_over']:.3f}  under={r['frac_under']:.3f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
