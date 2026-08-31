#!/usr/bin/env python
"""
Compare ScousePy-handoff .dat to Henshaw final_fits_updated.dat.

Matches positions by (l,b), then greedily matches components by nearest velocity.
Reports K agreement and per-component dv / d-amp / width ratios.

Example:
  python experiments/MOPRA_Count/plots/compare_scouse_fit_handoff.py \\
    --pred data/mopra_cmz_scouse_ft_v1_handoff.dat \\
    --truth data/final_fits_updated.dat
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPT = Path(__file__).resolve()
_REPO = _SCRIPT.parents[3]
sys.path.insert(0, str(_REPO / "src"))

import matplotlib.pyplot as plt
import numpy as np

from spectackle.data.scouse_fit_handoff import fwhm_to_sigma, parse_handoff_dat  ### noqa: E402


def _match_velocities(
    v_pred: np.ndarray,
    v_true: np.ndarray,
    *,
    max_dv: float,
) -> list[tuple[int, int, float]]:
    """Greedy nearest-velocity matching. Returns (ip, it, dv) pairs."""
    pairs: list[tuple[int, int, float]] = []
    used_t: set[int] = set()
    used_p: set[int] = set()
    ### Sort candidate pairs by |dv|.
    cands: list[tuple[float, int, int]] = []
    for ip, vp in enumerate(v_pred):
        for it, vt in enumerate(v_true):
            cands.append((abs(float(vp - vt)), ip, it))
    cands.sort()
    for dv, ip, it in cands:
        if dv > max_dv:
            break
        if ip in used_p or it in used_t:
            continue
        used_p.add(ip)
        used_t.add(it)
        pairs.append((ip, it, float(v_pred[ip] - v_true[it])))
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare handoff .dat to Henshaw truth.")
    parser.add_argument("--pred", type=Path, required=True)
    parser.add_argument(
        "--truth",
        type=Path,
        default=_REPO / "data" / "final_fits_updated.dat",
    )
    parser.add_argument(
        "--henshaw-width",
        type=str,
        default="fwhm",
        choices=("fwhm", "sigma"),
        help="Interpret Henshaw col7 as FWHM (default) or dispersion.",
    )
    parser.add_argument("--max-dv", type=float, default=10.0, help="Max |dv| km/s for a match.")
    parser.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="Summary JSON (default: <pred-stem>_vs_henshaw.json next to pred).",
    )
    parser.add_argument(
        "--out-png",
        type=Path,
        default=None,
        help="Diagnostic PNG (default: figures/<pred-stem>_vs_henshaw.png).",
    )
    args = parser.parse_args()

    pred = parse_handoff_dat(args.pred)
    truth = parse_handoff_dat(args.truth)
    shared = sorted(set(pred) & set(truth))
    only_pred = len(set(pred) - set(truth))
    only_truth = len(set(truth) - set(pred))

    k_pred_list = []
    k_true_list = []
    dvs = []
    damps = []
    width_ratios = []
    n_matched_comp = 0
    n_true_comp = 0
    n_pred_comp = 0

    for key in shared:
        p = pred[key]
        t = truth[key]
        kp = int(p[0, 0])
        kt = int(t[0, 0])
        k_pred_list.append(kp)
        k_true_list.append(kt)
        n_pred_comp += p.shape[0]
        n_true_comp += t.shape[0]

        v_p = p[:, 5]
        v_t = t[:, 5]
        pairs = _match_velocities(v_p, v_t, max_dv=args.max_dv)
        n_matched_comp += len(pairs)
        for ip, it, dv in pairs:
            dvs.append(dv)
            damps.append(float(p[ip, 3] - t[it, 3]))
            sig_p = float(p[ip, 7])
            w_t = float(t[it, 7])
            if args.henshaw_width == "fwhm":
                sig_t = float(fwhm_to_sigma(w_t))
            else:
                sig_t = w_t
            if sig_t > 0:
                width_ratios.append(sig_p / sig_t)

    k_pred_arr = np.asarray(k_pred_list, dtype=np.float64)
    k_true_arr = np.asarray(k_true_list, dtype=np.float64)
    dk = k_pred_arr - k_true_arr
    dvs_arr = np.asarray(dvs, dtype=np.float64) if dvs else np.zeros(0)
    damps_arr = np.asarray(damps, dtype=np.float64) if damps else np.zeros(0)
    wr_arr = np.asarray(width_ratios, dtype=np.float64) if width_ratios else np.zeros(0)

    summary = {
        "pred": str(args.pred),
        "truth": str(args.truth),
        "henshaw_width": args.henshaw_width,
        "max_dv_kms": args.max_dv,
        "n_pos_shared": len(shared),
        "n_pos_only_pred": only_pred,
        "n_pos_only_truth": only_truth,
        "k": {
            "mae": float(np.mean(np.abs(dk))) if dk.size else None,
            "median_delta": float(np.median(dk)) if dk.size else None,
            "frac_exact": float(np.mean(dk == 0)) if dk.size else None,
            "frac_over": float(np.mean(dk > 0)) if dk.size else None,
            "frac_under": float(np.mean(dk < 0)) if dk.size else None,
        },
        "components": {
            "n_pred": int(n_pred_comp),
            "n_true": int(n_true_comp),
            "n_matched": int(n_matched_comp),
            "match_frac_of_true": float(n_matched_comp / n_true_comp) if n_true_comp else None,
            "median_abs_dv": float(np.median(np.abs(dvs_arr))) if dvs_arr.size else None,
            "mean_abs_dv": float(np.mean(np.abs(dvs_arr))) if dvs_arr.size else None,
            "median_abs_damp": float(np.median(np.abs(damps_arr))) if damps_arr.size else None,
            "median_sigma_ratio_pred_over_true": float(np.median(wr_arr)) if wr_arr.size else None,
        },
    }

    out_json = args.out_json
    if out_json is None:
        out_json = args.pred.with_name(args.pred.stem + "_vs_henshaw.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    out_png = args.out_png
    if out_png is None:
        out_png = (
            _REPO
            / "experiments"
            / "MOPRA_Count"
            / "figures"
            / f"{args.pred.stem}_vs_henshaw.png"
        )
    out_png.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    ax = axes[0, 0]
    if dk.size:
        bins = np.arange(dk.min() - 0.5, dk.max() + 1.5, 1.0)
        ax.hist(dk, bins=bins, color="steelblue", edgecolor="k", linewidth=0.4)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("dK (pred - Henshaw)")
    ax.set_ylabel("Positions")
    ax.set_title(
        f"K: MAE={summary['k']['mae']:.3f}  exact={summary['k']['frac_exact']:.3f}"
        if summary["k"]["mae"] is not None
        else "K"
    )

    ax = axes[0, 1]
    if dvs_arr.size:
        ax.hist(dvs_arr, bins=40, color="darkorange", edgecolor="k", linewidth=0.3)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("dv (pred - Henshaw) [km/s]")
    ax.set_ylabel("Matched components")
    med = summary["components"]["median_abs_dv"]
    ax.set_title(f"Matched |dv| median={med:.2f}" if med is not None else "dv")

    ax = axes[1, 0]
    if damps_arr.size:
        ax.hist(damps_arr, bins=40, color="seagreen", edgecolor="k", linewidth=0.3)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("d-amp (pred - Henshaw) [K]")
    ax.set_ylabel("Matched components")
    ax.set_title("Amplitude residuals")

    ax = axes[1, 1]
    if wr_arr.size:
        ax.hist(wr_arr, bins=40, color="slateblue", edgecolor="k", linewidth=0.3)
        ax.axvline(1.0, color="k", lw=0.8)
    ax.set_xlabel("sigma_pred / sigma_true")
    ax.set_ylabel("Matched components")
    wr = summary["components"]["median_sigma_ratio_pred_over_true"]
    ax.set_title(f"Width ratio median={wr:.2f}" if wr is not None else "Width")

    fig.suptitle(
        f"Handoff vs Henshaw  (shared pos={len(shared)}, "
        f"matched comps={n_matched_comp}/{n_true_comp})",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)

    print(
        f"shared={len(shared)}  K MAE={summary['k']['mae']:.3f}  "
        f"exact={summary['k']['frac_exact']:.3f}"
    )
    print(
        f"matched comps={n_matched_comp}/{n_true_comp}  "
        f"median |dv|={summary['components']['median_abs_dv']}"
    )
    print(f"Wrote {out_json}")
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
