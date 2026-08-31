#!/usr/bin/env python
"""
Compare Scouse-eval JSON summaries across generator baselines.

Writes a text table to stdout. Optional --out:
  *.json -> JSON payload
  *.png -> bar-chart figure (also writes a sibling .json)

Example:
  python experiments/MOPRA_Count/plots/compare_scouse_baselines.py \\
    --jsons \\
      experiments/MOPRA_Count/figures/mopra_cmz_k_pred_simple_k6_20k_scouse_stats.json \\
      experiments/MOPRA_Count/figures/mopra_cmz_k_pred_simple_realamp_k6_20k_scouse_stats.json \\
    --labels simple_k6_20k simple_realamp \\
    --out experiments/MOPRA_Count/figures/simple_realamp_vs_baselines.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _edge(stats: dict, key: str) -> float | None:
    b = stats.get("edge_bins", {}).get(key)
    if not b:
        return None
    return b.get("mean_delta")


def _build_rows(labels: list[str], jsons: list[Path]) -> list[dict]:
    rows = []
    for lab, path in zip(labels, jsons):
        s = json.loads(path.read_text(encoding="utf-8"))
        g = s["global"]
        rows.append(
            {
                "label": lab,
                "path": str(path),
                "mae": g["mae"],
                "median_delta": g["median_delta"],
                "frac_over": g["frac_over"],
                "frac_under": g["frac_under"],
                "frac_exact": g["frac_exact"],
                "edge_1-2": _edge(s, "1-2"),
                "edge_10-20": _edge(s, "10-20") or _edge(s, ">=10"),
                "k1_mae": (s.get("by_k_true") or {}).get("1", {}).get("mae"),
                "k3_mae": (s.get("by_k_true") or {}).get("3", {}).get("mae"),
            }
        )
    return rows


def _format_table(rows: list[dict]) -> str:
    header = (
        f"{'run':<22} {'MAE':>7} {'meddK':>7} {'over':>6} {'under':>6} "
        f"{'edge1-2':>8} {'edge10+':>8} {'K1 MAE':>7} {'K3 MAE':>7}"
    )
    lines = [header, "-" * len(header)]
    for r in rows:
        lines.append(
            f"{r['label']:<22} {r['mae']:7.3f} {r['median_delta']:7.2f} "
            f"{r['frac_over']:6.3f} {r['frac_under']:6.3f} "
            f"{(r['edge_1-2'] if r['edge_1-2'] is not None else float('nan')):8.3f} "
            f"{(r['edge_10-20'] if r['edge_10-20'] is not None else float('nan')):8.3f} "
            f"{(r['k1_mae'] if r['k1_mae'] is not None else float('nan')):7.3f} "
            f"{(r['k3_mae'] if r['k3_mae'] is not None else float('nan')):7.3f}"
        )
    return "\n".join(lines) + "\n"


def _plot_bars(rows: list[dict], out_png: Path) -> None:
    labels = [r["label"] for r in rows]
    mae = np.asarray([r["mae"] for r in rows], dtype=float)
    k1 = np.asarray([r["k1_mae"] if r["k1_mae"] is not None else np.nan for r in rows], dtype=float)
    k3 = np.asarray([r["k3_mae"] if r["k3_mae"] is not None else np.nan for r in rows], dtype=float)
    over = np.asarray([r["frac_over"] for r in rows], dtype=float)
    under = np.asarray([r["frac_under"] for r in rows], dtype=float)

    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8), constrained_layout=True)

    axes[0].bar(x, mae, color="#4E79A7", edgecolor="k", linewidth=0.4)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    axes[0].set_ylabel("Scouse MAE")
    axes[0].set_title("Overall K MAE")

    w = 0.35
    axes[1].bar(x - w / 2, k1, width=w, label="K=1", color="#F28E2B", edgecolor="k", linewidth=0.3)
    axes[1].bar(x + w / 2, k3, width=w, label="K=3", color="#E15759", edgecolor="k", linewidth=0.3)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    axes[1].set_ylabel("MAE")
    axes[1].set_title("By K_true")
    axes[1].legend(fontsize=8, frameon=False)

    axes[2].bar(x - w / 2, over, width=w, label="over", color="#E15759", edgecolor="k", linewidth=0.3)
    axes[2].bar(x + w / 2, under, width=w, label="under", color="#4E79A7", edgecolor="k", linewidth=0.3)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    axes[2].set_ylabel("fraction")
    axes[2].set_title("Over / under count")
    axes[2].legend(fontsize=8, frameon=False)

    fig.suptitle("Scouse baseline comparison", fontsize=11)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
    print(f"Wrote {out_png}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Tabulate Scouse MAE / dK / edge bins across runs.")
    parser.add_argument("--jsons", type=Path, nargs="+", required=True)
    parser.add_argument("--labels", type=str, nargs="+", default=None)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional *.json table dump and/or *.png bar chart.",
    )
    args = parser.parse_args()

    labels = args.labels
    if labels is None:
        labels = [p.stem.replace("_scouse_stats", "") for p in args.jsons]
    if len(labels) != len(args.jsons):
        raise ValueError("--labels length must match --jsons")

    rows = _build_rows(labels, args.jsons)
    text = _format_table(rows)
    print(text, end="")

    if args.out is None:
        return

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"rows": rows, "table": text}
    suffix = args.out.suffix.lower()
    if suffix == ".png":
        _plot_bars(rows, args.out)
        json_path = args.out.with_suffix(".json")
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote {json_path}")
    else:
        ### Default / .json: write the table payload only.
        out = args.out if suffix == ".json" else args.out.with_suffix(".json")
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
