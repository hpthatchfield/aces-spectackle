#!/usr/bin/env python
"""
Resolvable-peak count vs Henshaw K: real Scouse-labeled spectra vs scouse_dat synthetic.

Diagnoses generator/training gap (crowded K=3/4 cores look multi-peak in synth but blended in data).

Example:
  python experiments/MOPRA_Count/plots/plot_resolvable_peak_gap.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

_SCRIPT = Path(__file__).resolve()
_REPO = _SCRIPT.parents[3]
sys.path.insert(0, str(_SCRIPT.parent))
sys.path.insert(0, str(_REPO / "src"))

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
from spectral_cube import SpectralCube

from plot_style import COL_CLEAN, COL_MAE, COL_SPEC, COL_TRAIN  ### noqa: E402
from spectackle.data.mopra_generator import build_mopra_synth_cfg  ### noqa: E402
from spectackle.data.resolvable_peaks import (  ### noqa: E402
    blend_stats_from_dat_rows,
    blend_stats_from_synth_example,
    count_resolvable_peaks,
    sample_synthetic_by_k,
)


def _scouse_components_by_pos(dat_path: Path) -> dict[tuple[float, float], list[np.ndarray]]:
    arr = np.loadtxt(dat_path)
    by_pos: dict[tuple[float, float], list[np.ndarray]] = defaultdict(list)
    for row in arr:
        key = (round(float(row[1]), 5), round(float(row[2]), 5))
        by_pos[key].append(row)
    return dict(by_pos)


def _vel_axis_from_cube(cube_path: Path) -> np.ndarray:
    cube = SpectralCube.read(str(cube_path.resolve()), use_dask=False)
    return cube.spectral_axis.to("km/s").value.astype(np.float64)


def collect_real_records(
    cache_path: Path,
    cube_path: Path,
    dat_path: Path,
    vel_kms: np.ndarray,
    *,
    prominence_sigma: float,
    min_sep_kms: float,
    vel_range: tuple[float, float],
    prominence_mode: str,
    peak_frac: float,
) -> list[dict]:
    from spectral_cube import SpectralCube

    z = np.load(cache_path)
    by_pos = _scouse_components_by_pos(dat_path)
    cube = SpectralCube.read(str(cube_path.resolve()), use_dask=False)
    arr = np.asarray(cube.filled(np.nan), dtype=np.float64)
    if arr.ndim == 4:
        arr = arr[0]

    records: list[dict] = []
    n = int(z["spec_norm"].shape[0])
    for i in range(n):
        yi = int(z["yi"][i])
        xi = int(z["xi"][i])
        spec = arr[:, yi, xi].astype(np.float64)
        k_true = int(z["K_true"][i, 0])
        l = float(z["l"][i])
        b = float(z["b"][i])
        key = (round(l, 5), round(b, 5))
        n_res, sigma = count_resolvable_peaks(
            spec,
            vel_kms,
            vel_range=vel_range,
            prominence_sigma=prominence_sigma,
            min_sep_kms=min_sep_kms,
            prominence_mode=prominence_mode,
            peak_frac=peak_frac,
        )
        blend = blend_stats_from_dat_rows(by_pos.get(key, []))
        records.append(
            {
                "source": "real",
                "k_true": k_true,
                "n_resolvable": n_res,
                "sigma_rms": sigma,
                "min_sep_sigma": blend["min_sep_sigma"],
                "min_amp_ratio": blend["min_amp_ratio"],
                "l": l,
                "b": b,
                "spec": spec,
            }
        )
    return records


def collect_synth_records(
    cfg: dict,
    vel_kms: np.ndarray,
    *,
    n_per_k: int,
    seed: int,
    prominence_sigma: float,
    min_sep_kms: float,
    vel_range: tuple[float, float],
    k_values: tuple[int, ...],
    prominence_mode: str,
    peak_frac: float,
    use_clean_spec: bool = True,
) -> list[dict]:
    examples = sample_synthetic_by_k(cfg, n_per_k=n_per_k, k_values=k_values, seed=seed)
    records: list[dict] = []
    for ex in examples:
        spec = ex["spec_clean" if use_clean_spec else "spec"].astype(np.float64)
        k_true = int(ex["k"])
        n_res, sigma = count_resolvable_peaks(
            spec,
            vel_kms,
            vel_range=vel_range,
            prominence_sigma=prominence_sigma,
            min_sep_kms=min_sep_kms,
            prominence_mode=prominence_mode,
            peak_frac=peak_frac,
            blank_value=None,
        )
        blend = blend_stats_from_synth_example(ex)
        records.append(
            {
                "source": "synth",
                "k_true": k_true,
                "n_resolvable": n_res,
                "sigma_rms": sigma,
                "min_sep_sigma": blend["min_sep_sigma"],
                "min_amp_ratio": blend["min_amp_ratio"],
                "spec": spec,
            }
        )
    return records


def _mean_by_k(records: list[dict], k_values: tuple[int, ...]) -> dict[str, dict[int, float]]:
    out: dict[str, dict[int, float]] = {
        "n_resolvable": {},
        "min_sep_sigma": {},
        "min_amp_ratio": {},
        "n": {},
    }
    for k in k_values:
        sub = [r for r in records if int(r["k_true"]) == int(k)]
        out["n"][k] = len(sub)
        if not sub:
            for key in ("n_resolvable", "min_sep_sigma", "min_amp_ratio"):
                out[key][k] = float("nan")
            continue
        out["n_resolvable"][k] = float(np.mean([r["n_resolvable"] for r in sub]))
        sep = [r["min_sep_sigma"] for r in sub if np.isfinite(r["min_sep_sigma"])]
        amp = [r["min_amp_ratio"] for r in sub if np.isfinite(r["min_amp_ratio"])]
        out["min_sep_sigma"][k] = float(np.mean(sep)) if sep else float("nan")
        out["min_amp_ratio"][k] = float(np.mean(amp)) if amp else float("nan")
    return out


def _plot_box_resolvable(
    ax,
    real: list[dict],
    synth: list[dict],
    k_values: tuple[int, ...],
) -> None:
    positions = []
    data = []
    colors = []
    labels = []
    x = 0
    for k in k_values:
        for src, recs, col in (("real", real, COL_SPEC), ("synth", synth, COL_TRAIN)):
            sub = [r["n_resolvable"] for r in recs if int(r["k_true"]) == int(k)]
            if not sub:
                continue
            positions.append(x)
            data.append(sub)
            colors.append(col)
            labels.append(f"K={k}\n{src}")
            x += 1
        x += 0.5
    bp = ax.boxplot(data, positions=positions, widths=0.6, patch_artist=True, showfliers=False)
    for patch, col in zip(bp["boxes"], colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.65)
    ax.set_xticks([p for p in positions])
    ax.set_xticklabels(labels, fontsize=7, rotation=45, ha="right")
    ax.axhline(1.0, color="0.5", ls="--", lw=0.8)
    ax.set_ylabel("Resolvable peak count (3sigma prom.)")
    ax.set_title("Resolvable peaks vs Henshaw K")


def _plot_mean_gap(ax, real: list[dict], synth: list[dict], k_values: tuple[int, ...]) -> None:
    r_mean = _mean_by_k(real, k_values)["n_resolvable"]
    s_mean = _mean_by_k(synth, k_values)["n_resolvable"]
    xs = np.arange(len(k_values), dtype=np.float64)
    w = 0.35
    ax.bar(xs - w / 2, [r_mean[k] for k in k_values], width=w, label="real", color=COL_SPEC, alpha=0.85)
    ax.bar(xs + w / 2, [s_mean[k] for k in k_values], width=w, label="synth", color=COL_TRAIN, alpha=0.85)
    ax.plot(xs, [float(k) for k in k_values], "k--", lw=1.0, label="K_true (ideal)")
    ax.set_xticks(xs)
    ax.set_xticklabels([str(k) for k in k_values])
    ax.set_xlabel("Henshaw K")
    ax.set_ylabel("Mean resolvable peaks")
    ax.legend(fontsize=8)
    ax.set_title("Mean resolvable peaks by K")


def _plot_blend_hist(
    ax,
    real: list[dict],
    synth: list[dict],
    *,
    field: str,
    title: str,
    k_min: int = 2,
) -> None:
    r_vals = [r[field] for r in real if int(r["k_true"]) >= k_min and np.isfinite(r[field])]
    s_vals = [r[field] for r in synth if int(r["k_true"]) >= k_min and np.isfinite(r[field])]
    if not r_vals and not s_vals:
        ax.set_title(f"{title} (no K>={k_min})")
        return
    lo = float(np.nanpercentile(np.concatenate([r_vals, s_vals]), 1))
    hi = float(np.nanpercentile(np.concatenate([r_vals, s_vals]), 99))
    bins = np.linspace(lo, hi, 30)
    ax.hist(r_vals, bins=bins, alpha=0.55, label=f"real (n={len(r_vals)})", color=COL_SPEC, density=True)
    ax.hist(s_vals, bins=bins, alpha=0.55, label=f"synth (n={len(s_vals)})", color=COL_TRAIN, density=True)
    ax.set_xlabel(title)
    ax.set_ylabel("density")
    ax.legend(fontsize=8)
    ax.set_title(f"{title} (K>={k_min})")


def _plot_blend_scatter(ax, real: list[dict], synth: list[dict], *, k_min: int = 2) -> None:
    for recs, col, lab in ((real, COL_SPEC, "real"), (synth, COL_TRAIN, "synth")):
        xs = [r["min_sep_sigma"] for r in recs if int(r["k_true"]) >= k_min and np.isfinite(r["min_sep_sigma"])]
        ys = [r["min_amp_ratio"] for r in recs if int(r["k_true"]) >= k_min and np.isfinite(r["min_amp_ratio"])]
        if xs:
            ax.scatter(xs, ys, s=12, alpha=0.35, c=col, label=lab, edgecolors="none")
    ax.set_xlabel("Min component separation (sigma)")
    ax.set_ylabel("Min amp ratio")
    ax.set_xscale("log")
    ax.legend(fontsize=8)
    ax.set_title(f"Blend hardness (K>={k_min})")


def _mark_peaks_on_axis(
    ax,
    spec: np.ndarray,
    vel: np.ndarray,
    sigma: float,
    prom_sigma: float,
    min_sep_kms: float,
    *,
    prominence_mode: str = "adaptive",
    peak_frac: float = 0.25,
) -> None:
    valid = np.isfinite(spec)
    y = spec[valid] - float(np.nanmedian(spec[valid]))
    v = vel[valid]
    dv = float(np.median(np.diff(v))) if v.size > 1 else 2.0
    min_dist = max(1, int(round(min_sep_kms / max(abs(dv), 1e-6))))
    peak_amp = float(np.nanmax(y)) if y.size else 0.0
    if prominence_mode == "adaptive":
        height = max(prom_sigma * sigma, peak_frac * peak_amp)
    else:
        height = prom_sigma * sigma
    peaks, _ = find_peaks(y, height=height, prominence=height, distance=min_dist)
    if peaks.size:
        ax.scatter(v[peaks], y[peaks], s=28, c=COL_MAE, zorder=5, label="3sigma peaks")


def plot_example_spectra(
    real: list[dict],
    synth: list[dict],
    vel_kms: np.ndarray,
    *,
    k_focus: int,
    n_each: int,
    prominence_sigma: float,
    min_sep_kms: float,
    vel_range: tuple[float, float],
    out_path: Path,
    seed: int,
    prominence_mode: str,
    peak_frac: float,
) -> None:
    rng = np.random.default_rng(seed)
    r_pool = [r for r in real if int(r["k_true"]) == k_focus]
    s_pool = [r for r in synth if int(r["k_true"]) == k_focus]
    if len(r_pool) < n_each or len(s_pool) < n_each:
        print(f"Warning: fewer than {n_each} K={k_focus} examples for spectra panel")
    r_pick = list(rng.choice(r_pool, size=min(n_each, len(r_pool)), replace=False)) if r_pool else []
    s_pick = list(rng.choice(s_pool, size=min(n_each, len(s_pool)), replace=False)) if s_pool else []

    fig, axes = plt.subplots(2, n_each, figsize=(3.2 * n_each, 5.0), sharex=True, sharey=False)
    if n_each == 1:
        axes = np.array([[axes[0]], [axes[1]]])
    vlo, vhi = vel_range if vel_range is not None else (float(np.nanmin(vel_kms)), float(np.nanmax(vel_kms)))
    for j, rec in enumerate(r_pick):
        ax = axes[0, j]
        spec = rec["spec"]
        ax.plot(vel_kms, spec, color=COL_SPEC, lw=0.9)
        _mark_peaks_on_axis(
            ax, spec, vel_kms, rec["sigma_rms"], prominence_sigma, min_sep_kms,
            prominence_mode=prominence_mode, peak_frac=peak_frac,
        )
        ax.set_xlim(vlo, vhi)
        ax.set_title(f"real K={k_focus}, n_res={rec['n_resolvable']}", fontsize=9)
        if j == 0:
            ax.set_ylabel("T_B (norm)")
    for j, rec in enumerate(s_pick):
        ax = axes[1, j]
        spec = rec["spec"]
        ax.plot(vel_kms, spec, color=COL_TRAIN, lw=0.9)
        _mark_peaks_on_axis(
            ax, spec, vel_kms, rec["sigma_rms"], prominence_sigma, min_sep_kms,
            prominence_mode=prominence_mode, peak_frac=peak_frac,
        )
        ax.set_xlim(vlo, vhi)
        ax.set_title(f"synth K={k_focus}, n_res={rec['n_resolvable']}", fontsize=9)
        if j == 0:
            ax.set_ylabel("T_B (norm)")
        ax.set_xlabel("v (km/s)")
    fig.suptitle(
        f"Example spectra K={k_focus} (green dots = 3sigma prominence peaks; synth = noisy training-like)",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def _sanitize_for_json(obj):
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def _json_default(obj):
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def main() -> None:
    parser = argparse.ArgumentParser(description="Resolvable peak gap: real vs scouse_dat synthetic.")
    parser.add_argument("--cache", type=Path, default=_SCRIPT.parents[1] / "cache" / "scouse_labeled_smooth60.npz")
    parser.add_argument("--dat", type=Path, default=_REPO / "data" / "final_fits_updated.dat")
    parser.add_argument("--cube", type=Path, default=_REPO / "data" / "CMZ_3mm_HNCO_60.fits")
    parser.add_argument("--out-dir", type=Path, default=_SCRIPT.parents[1] / "figures")
    parser.add_argument("--tag", type=str, default="resolvable_peak_gap")
    parser.add_argument("--n-per-k", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prominence-sigma", type=float, default=3.0)
    parser.add_argument("--prominence-mode", choices=("adaptive", "fixed_sigma"), default="adaptive")
    parser.add_argument("--peak-frac", type=float, default=0.25, help="For adaptive mode: frac of peak height")
    parser.add_argument("--min-sep-kms", type=float, default=4.0)
    parser.add_argument(
        "--vel-range",
        type=float,
        nargs=2,
        default=None,
        help="Optional velocity window (km/s). Default: full axis (matches prior synth gap diagnostic).",
    )
    parser.add_argument("--k-values", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--k-focus", type=int, default=3, help="K for example-spectrum panel")
    parser.add_argument("--n-examples", type=int, default=4)
    parser.add_argument(
        "--gen-preset",
        type=str,
        default="scouse_dat",
        help="Synthetic generator preset (e.g. scouse_dat, scouse_dat_blend_sat).",
    )
    parser.add_argument(
        "--synth-spec",
        choices=("noisy", "clean"),
        default="noisy",
        help="Plot/count synth on noisy training-like spectra (default) or clean Gaussians.",
    )
    args = parser.parse_args()

    k_values = tuple(int(k) for k in args.k_values)
    vel_range = (float(args.vel_range[0]), float(args.vel_range[1])) if args.vel_range else None
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = str(args.tag)
    use_clean_spec = args.synth_spec == "clean"

    print("Loading velocity axis...", flush=True)
    vel_kms = _vel_axis_from_cube(args.cube)

    print("Collecting real Scouse-labeled spectra...", flush=True)
    real = collect_real_records(
        args.cache,
        args.cube,
        args.dat,
        vel_kms,
        prominence_sigma=args.prominence_sigma,
        min_sep_kms=args.min_sep_kms,
        vel_range=vel_range,
        prominence_mode=args.prominence_mode,
        peak_frac=args.peak_frac,
    )
    print(f"  n_real={len(real)}", flush=True)

    print(f"Building {args.gen_preset} synth cfg and sampling ({args.synth_spec})...", flush=True)
    cfg = build_mopra_synth_cfg(repo_root=_REPO, gen_preset=args.gen_preset, axis_cube=args.cube)
    synth = collect_synth_records(
        cfg,
        vel_kms,
        n_per_k=args.n_per_k,
        seed=args.seed,
        prominence_sigma=args.prominence_sigma,
        min_sep_kms=args.min_sep_kms,
        vel_range=vel_range,
        k_values=k_values,
        prominence_mode=args.prominence_mode,
        peak_frac=args.peak_frac,
        use_clean_spec=use_clean_spec,
    )
    print(f"  n_synth={len(synth)}", flush=True)
    for k in k_values:
        print(f"    synth K={k}: {sum(1 for r in synth if r['k_true']==k)}", flush=True)

    ### Summary JSON (no spectra arrays).
    summary = {
        "cache": str(args.cache.resolve()),
        "dat": str(args.dat.resolve()),
        "cube": str(args.cube.resolve()),
        "gen_preset": args.gen_preset,
        "prominence_sigma": args.prominence_sigma,
        "prominence_mode": args.prominence_mode,
        "peak_frac": args.peak_frac,
        "real_spec": "raw_cube",
        "synth_spec": "spec_clean" if use_clean_spec else "spec_noisy",
        "min_sep_kms": args.min_sep_kms,
        "vel_range": list(vel_range) if vel_range else None,
        "k_values": list(k_values),
        "n_real": len(real),
        "n_synth": len(synth),
        "real_mean_by_k": _mean_by_k(real, k_values),
        "synth_mean_by_k": _mean_by_k(synth, k_values),
        "real_frac_undercount": {},
    }
    for k in k_values:
        sub = [r for r in real if int(r["k_true"]) == int(k)]
        if sub:
            frac = float(np.mean([r["n_resolvable"] < k for r in sub]))
            summary["real_frac_undercount"][str(k)] = frac
        synth_sub = [r for r in synth if int(r["k_true"]) == int(k)]
        if synth_sub:
            key = f"synth_frac_undercount_k{k}"
            summary[key] = float(np.mean([r["n_resolvable"] < k for r in synth_sub]))

    json_path = out_dir / f"{stem}.json"
    with open(json_path, "w") as f:
        json.dump(_sanitize_for_json(summary), f, indent=2)
    print(f"Wrote {json_path}", flush=True)

    ### Main 2x2 figure.
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    _plot_mean_gap(axes[0, 0], real, synth, k_values)
    _plot_box_resolvable(axes[0, 1], real, synth, k_values)
    _plot_blend_hist(axes[1, 0], real, synth, field="min_sep_sigma", title="Min sep (sigma)")
    _plot_blend_scatter(axes[1, 1], real, synth, k_min=2)
    fig.suptitle("Resolvable peaks vs Henshaw K: real CMZ vs scouse_dat synthetic", fontsize=11)
    fig.tight_layout()
    main_png = out_dir / f"{stem}.png"
    fig.savefig(main_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {main_png}", flush=True)

    ### Amp-ratio panel (separate for readability).
    fig2, ax2 = plt.subplots(1, 1, figsize=(6, 4))
    _plot_blend_hist(ax2, real, synth, field="min_amp_ratio", title="Min amp ratio")
    fig2.tight_layout()
    amp_png = out_dir / f"{stem}_amp_ratio.png"
    fig2.savefig(amp_png, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"Wrote {amp_png}", flush=True)

    examples_png = out_dir / f"{stem}_k{args.k_focus}_examples.png"
    plot_example_spectra(
        real,
        synth,
        vel_kms,
        k_focus=int(args.k_focus),
        n_each=int(args.n_examples),
        prominence_sigma=args.prominence_sigma,
        min_sep_kms=args.min_sep_kms,
        vel_range=vel_range,
        out_path=examples_png,
        seed=args.seed + 1,
        prominence_mode=args.prominence_mode,
        peak_frac=args.peak_frac,
    )

    print("\nMean resolvable peaks by K (real | synth):", flush=True)
    r_m = summary["real_mean_by_k"]["n_resolvable"]
    s_m = summary["synth_mean_by_k"]["n_resolvable"]
    for k in k_values:
        print(f"  K={k}: real={r_m[k]:.2f}  synth={s_m[k]:.2f}  (ideal={k})", flush=True)


if __name__ == "__main__":
    main()
