#!/usr/bin/env python
"""
Side-by-side morphology: Henshaw/Scouse real components vs synthetic generators.

Compares primary SNR, amp ratios, and example spectra so generator dials can be
matched to MOPRA before training.

Example:
  python experiments/MOPRA_Count/plots/plot_synth_vs_real_morphology.py \\
    --gen-presets simple simple_mix \\
    --out-dir experiments/MOPRA_Count/figures/morphology
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

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import matplotlib.pyplot as plt
import numpy as np
from spectral_cube import SpectralCube

from plot_style import COL_CLEAN, COL_PRED, COL_SPEC, COL_TRAIN  ### noqa: E402
from spectackle.config import deep_update  ### noqa: E402
from spectackle.data.generator import DEFAULT_GEN, _make_v_axis, generate_spectrum  ### noqa: E402
from spectackle.data.mopra_generator import (  ### noqa: E402
    _apply_mopra_artifacts,
    build_mopra_synth_cfg,
)
from spectackle.data.mopra_scouse_accept import apply_glance_visible_label  ### noqa: E402
from spectackle.data.mopra_preprocess import valid_mask_mopra  ### noqa: E402
from spectackle.wcs_plot import wcs_celestial  ### noqa: E402


### Henshaw / ScousePy ascii: ncomps, l, b, amp, amp_err, v, v_err, width, ...
_COL_AMP = 3
_COL_L = 1
_COL_B = 2
_COL_V = 5
_COL_WIDTH = 7  ### Henshaw: FWHM (km/s)
_COL_RMS = 9


def _parse_dat(dat_path: Path) -> dict[tuple[float, float], list[np.ndarray]]:
    arr = np.loadtxt(dat_path)
    by: dict[tuple[float, float], list[np.ndarray]] = defaultdict(list)
    for row in arr:
        key = (round(float(row[_COL_L]), 5), round(float(row[_COL_B]), 5))
        by[key].append(row)
    return dict(by)


def _real_component_stats(by_pos: dict) -> dict[str, np.ndarray]:
    primary_snr: list[float] = []
    min_amp_ratio: list[float] = []
    sec_ratio: list[float] = []
    peak_amp: list[float] = []
    fwhm: list[float] = []
    k_list: list[int] = []
    for rows in by_pos.values():
        k = int(rows[0][0])
        amps = np.asarray([float(r[_COL_AMP]) for r in rows], dtype=np.float64)
        widths = np.asarray([float(r[_COL_WIDTH]) for r in rows], dtype=np.float64)
        rms = float(rows[0][_COL_RMS])
        if not np.isfinite(rms) or rms <= 0 or amps.size == 0:
            continue
        amax = float(np.max(amps))
        if amax <= 0:
            continue
        k_list.append(k)
        primary_snr.append(amax / rms)
        peak_amp.append(amax)
        fwhm.extend(widths.tolist())
        if k >= 2:
            min_amp_ratio.append(float(np.min(amps) / amax))
            for a in amps:
                if a < amax:
                    sec_ratio.append(float(a / amax))
    return {
        "primary_snr": np.asarray(primary_snr, dtype=np.float64),
        "min_amp_ratio": np.asarray(min_amp_ratio, dtype=np.float64),
        "sec_ratio": np.asarray(sec_ratio, dtype=np.float64),
        "peak_amp": np.asarray(peak_amp, dtype=np.float64),
        "fwhm": np.asarray(fwhm, dtype=np.float64),
        "k": np.asarray(k_list, dtype=np.int32),
    }


def _synth_one(cfg: dict, rng: np.random.Generator, v_axis: np.ndarray) -> dict:
    merged = dict(cfg)
    merged["gen"] = deep_update(dict(DEFAULT_GEN), cfg.get("gen", {}))
    ex = generate_spectrum(merged, rng, v_axis=v_axis)
    ex["spec"] = _apply_mopra_artifacts(ex["spec"], merged["gen"], rng)
    labeled = apply_glance_visible_label(dict(ex), merged)
    kd = int(ex["k"])
    noise = float(ex["noise_std"][0])
    amps = np.asarray(ex["component_amp"][:kd], dtype=np.float64)
    sigs = np.asarray(ex["component_sigma"][:kd], dtype=np.float64)
    fwhm = sigs * (2.0 * np.sqrt(2.0 * np.log(2.0)))
    out = {
        "k_drawn": kd,
        "k_label": int(labeled["k"]),
        "noise_std": noise,
        "amps": amps,
        "mus": np.asarray(ex["component_v_kms"][:kd], dtype=np.float64),
        "fwhm": fwhm,
        "spec": np.asarray(ex["spec"], dtype=np.float64),
        "v_axis": np.asarray(ex["v_axis"], dtype=np.float64),
    }
    if kd > 0 and noise > 0 and amps.size:
        amax = float(np.max(amps))
        out["primary_snr"] = amax / noise
        out["peak_amp"] = amax
        if kd >= 2 and amax > 0:
            out["min_amp_ratio"] = float(np.min(amps) / amax)
            out["sec_ratios"] = (amps[amps < amax] / amax).tolist()
        else:
            out["min_amp_ratio"] = float("nan")
            out["sec_ratios"] = []
    else:
        out["primary_snr"] = float("nan")
        out["peak_amp"] = float("nan")
        out["min_amp_ratio"] = float("nan")
        out["sec_ratios"] = []
    return out


def _synth_stats(cfg: dict, *, n: int, seed: int) -> dict[str, np.ndarray]:
    v_axis = _make_v_axis(cfg)
    rng = np.random.default_rng(seed)
    primary_snr, min_amp_ratio, sec_ratio, peak_amp, fwhm, k_drawn = [], [], [], [], [], []
    examples = []
    for _ in range(n):
        ex = _synth_one(cfg, rng, v_axis)
        if ex["k_drawn"] <= 0:
            continue
        k_drawn.append(ex["k_drawn"])
        primary_snr.append(ex["primary_snr"])
        peak_amp.append(ex["peak_amp"])
        fwhm.extend(ex["fwhm"].tolist())
        if ex["k_drawn"] >= 2 and np.isfinite(ex["min_amp_ratio"]):
            min_amp_ratio.append(ex["min_amp_ratio"])
            sec_ratio.extend(ex["sec_ratios"])
        examples.append(ex)
    return {
        "primary_snr": np.asarray(primary_snr, dtype=np.float64),
        "min_amp_ratio": np.asarray(min_amp_ratio, dtype=np.float64),
        "sec_ratio": np.asarray(sec_ratio, dtype=np.float64),
        "peak_amp": np.asarray(peak_amp, dtype=np.float64),
        "fwhm": np.asarray(fwhm, dtype=np.float64),
        "k": np.asarray(k_drawn, dtype=np.int32),
        "examples": examples,
    }


def _summary(name: str, st: dict) -> dict:
    def pct(a, qs=(10, 50, 90)):
        a = np.asarray(a, dtype=np.float64)
        a = a[np.isfinite(a)]
        if a.size == 0:
            return {f"p{q}": None for q in qs}
        vals = np.percentile(a, list(qs))
        return {f"p{q}": float(v) for q, v in zip(qs, vals)}

    return {
        "name": name,
        "n": int(np.asarray(st["primary_snr"]).size),
        "primary_snr": pct(st["primary_snr"]),
        "min_amp_ratio": pct(st["min_amp_ratio"]),
        "sec_ratio": pct(st["sec_ratio"]),
        "peak_amp": pct(st["peak_amp"]),
        "fwhm": pct(st["fwhm"]),
        "frac_primary_snr_gt20": float(np.mean(st["primary_snr"] > 20)) if len(st["primary_snr"]) else None,
        "frac_min_amp_ratio_lt02": (
            float(np.mean(st["min_amp_ratio"] < 0.2)) if len(st["min_amp_ratio"]) else None
        ),
    }


def _hist_panel(ax, real, series, labels, colors, *, xlabel, logx=False, bins=40):
    data = [real] + series
    labs = ["real Scouse"] + labels
    cols = ["#333333"] + colors
    if logx:
        pos = [d[d > 0] for d in data]
        lo = min(np.min(d) for d in pos if d.size)
        hi = max(np.max(d) for d in pos if d.size)
        edges = np.geomspace(lo, hi, bins)
    else:
        finite = np.concatenate([d[np.isfinite(d)] for d in data if len(d)])
        edges = np.linspace(np.nanpercentile(finite, 1), np.nanpercentile(finite, 99), bins)
    for d, lab, c in zip(data, labs, cols):
        d = d[np.isfinite(d)]
        if logx:
            d = d[d > 0]
        if d.size == 0:
            continue
        ax.hist(d, bins=edges, histtype="step", density=True, lw=1.6, label=lab, color=c)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("density")
    if logx:
        ax.set_xscale("log")
    ax.legend(fontsize=7, frameon=False)


def plot_histograms(real_st, synth_map: dict, out_path: Path) -> None:
    labels = list(synth_map.keys())
    colors = [COL_SPEC, COL_TRAIN, COL_PRED, COL_CLEAN][: len(labels)]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5), constrained_layout=True)
    _hist_panel(
        axes[0, 0],
        real_st["primary_snr"],
        [synth_map[k]["primary_snr"] for k in labels],
        labels,
        colors,
        xlabel="primary component SNR (amp / rms)",
        logx=True,
    )
    _hist_panel(
        axes[0, 1],
        real_st["min_amp_ratio"],
        [synth_map[k]["min_amp_ratio"] for k in labels],
        labels,
        colors,
        xlabel=r"min/max amp ratio (K$\geq$2 pixels)",
    )
    _hist_panel(
        axes[1, 0],
        real_st["sec_ratio"],
        [synth_map[k]["sec_ratio"] for k in labels],
        labels,
        colors,
        xlabel="secondary amp / primary amp",
    )
    _hist_panel(
        axes[1, 1],
        real_st["fwhm"],
        [synth_map[k]["fwhm"] for k in labels],
        labels,
        colors,
        xlabel="component FWHM (km/s)",
    )
    fig.suptitle("Real Scouse vs synthetic morphology", fontsize=12)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"Wrote {out_path}")


def _load_real_spectrum_examples(
    cube_path: Path,
    by_pos: dict,
    *,
    k_targets: tuple[int, ...] = (1, 2, 3),
    n_each: int = 3,
    seed: int = 0,
) -> dict[int, list[dict]]:
    cube = SpectralCube.read(str(cube_path.resolve()), use_dask=False)
    arr = np.asarray(cube.filled(np.nan), dtype=np.float64)
    if arr.ndim == 4:
        arr = arr[0]
    vel = cube.spectral_axis.to("km/s").value.astype(np.float64)
    wcs = wcs_celestial(cube.header)
    rng = np.random.default_rng(seed)
    buckets: dict[int, list] = {k: [] for k in k_targets}
    keys = list(by_pos.keys())
    rng.shuffle(keys)
    for key in keys:
        rows = by_pos[key]
        k = int(rows[0][0])
        if k not in buckets or len(buckets[k]) >= n_each:
            continue
        l, b = key
        xp, yp = wcs.all_world2pix([[l, b]], 0)[0]
        xi, yi = int(round(float(xp))), int(round(float(yp)))
        if not (0 <= yi < arr.shape[1] and 0 <= xi < arr.shape[2]):
            continue
        spec = arr[:, yi, xi]
        if valid_mask_mopra(spec).sum() < 40:
            continue
        amps = np.asarray([float(r[_COL_AMP]) for r in rows], dtype=np.float64)
        mus = np.asarray([float(r[_COL_V]) for r in rows], dtype=np.float64)
        buckets[k].append(
            {
                "spec": spec.astype(np.float64),
                "v_axis": vel,
                "amps": amps,
                "mus": mus,
                "k": k,
                "l": l,
                "b": b,
                "title": f"real K={k}  l={l:.3f} b={b:.3f}",
            }
        )
        if all(len(buckets[kk]) >= n_each for kk in k_targets):
            break
    return buckets


def _pick_synth_examples(examples: list[dict], k_targets, n_each, seed=1) -> dict[int, list]:
    rng = np.random.default_rng(seed)
    buckets = {k: [] for k in k_targets}
    order = np.arange(len(examples))
    rng.shuffle(order)
    for i in order:
        ex = examples[i]
        kd = int(ex["k_drawn"])
        if kd not in buckets or len(buckets[kd]) >= n_each:
            continue
        amax = float(np.max(ex["amps"])) if ex["amps"].size else 0.0
        amin = float(np.min(ex["amps"])) if ex["amps"].size else 0.0
        ratio = amin / amax if amax > 0 and kd >= 2 else float("nan")
        buckets[kd].append(
            {
                "spec": ex["spec"],
                "v_axis": ex["v_axis"],
                "amps": ex["amps"],
                "mus": ex["mus"],
                "k": kd,
                "title": (
                    f"synth draw={kd} label={ex['k_label']}  "
                    f"SNR={ex['primary_snr']:.1f}"
                    + (f"  amin/amax={ratio:.2f}" if np.isfinite(ratio) else "")
                ),
            }
        )
        if all(len(buckets[kk]) >= n_each for kk in k_targets):
            break
    return buckets


def plot_spectrum_grid(
    real_ex: dict[int, list],
    synth_ex_map: dict[str, dict[int, list]],
    out_path: Path,
    *,
    k_targets: tuple[int, ...] = (1, 2, 3),
    n_each: int = 3,
) -> None:
    """
    Rows: real, then each synth preset.
    Columns: K groups, n_each examples each.
    """
    presets = list(synth_ex_map.keys())
    n_rows = 1 + len(presets)
    n_cols = len(k_targets) * n_each
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.4 * n_cols, 2.3 * n_rows), squeeze=False)

    def _draw_row(row, buckets, color_line):
        col = 0
        for k in k_targets:
            items = buckets.get(k, [])
            for j in range(n_each):
                ax = axes[row, col]
                if j >= len(items):
                    ax.set_axis_off()
                    col += 1
                    continue
                ex = items[j]
                v = ex["v_axis"]
                spec = ex["spec"]
                ok = np.isfinite(spec)
                ax.plot(v[ok], spec[ok], color=color_line, lw=0.9)
                order = np.argsort(-ex["amps"])
                for rank, idx in enumerate(order):
                    ax.axvline(
                        ex["mus"][idx],
                        color=COL_CLEAN if rank == 0 else COL_PRED,
                        ls=":" if rank == 0 else "--",
                        lw=0.9,
                        alpha=0.85,
                    )
                mus = ex["mus"]
                pad = 40.0
                ax.set_xlim(max(v.min(), mus.min() - pad), min(v.max(), mus.max() + pad))
                ax.set_title(ex["title"], fontsize=6.5, pad=2)
                ax.tick_params(labelsize=6)
                if row == n_rows - 1:
                    ax.set_xlabel(r"$v$", fontsize=7)
                if col == 0:
                    ax.set_ylabel(r"$T$", fontsize=7)
                col += 1

    _draw_row(0, real_ex, "#333333")
    for i, name in enumerate(presets):
        _draw_row(1 + i, synth_ex_map[name], COL_SPEC if i == 0 else COL_TRAIN)

    row_labels = ["real Scouse"] + presets
    fig.subplots_adjust(left=0.06, right=0.99, top=0.90, bottom=0.06, wspace=0.35, hspace=0.55)
    for r, lab in enumerate(row_labels):
        y0 = axes[r, 0].get_position().y0
        y1 = axes[r, 0].get_position().y1
        fig.text(
            0.01,
            0.5 * (y0 + y1),
            lab,
            rotation=90,
            va="center",
            ha="left",
            fontsize=9,
            fontweight="bold",
        )
    for ki, k in enumerate(k_targets):
        ax0 = axes[0, ki * n_each]
        pos = ax0.get_position()
        fig.text(
            pos.x0 + 0.5 * n_each * (axes[0, ki * n_each].get_position().width + 0.01),
            0.96,
            f"K = {k}",
            ha="center",
            fontsize=11,
            fontweight="bold",
            transform=fig.transFigure,
        )
    fig.suptitle(
        "Real vs synth spectra (dotted primary, dashed secondary centers)",
        fontsize=11,
        y=0.995,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dat", type=Path, default=_REPO / "data" / "final_fits_updated.dat")
    parser.add_argument("--cube", type=Path, default=_REPO / "data" / "CMZ_3mm_HNCO_60.fits")
    parser.add_argument(
        "--gen-presets",
        nargs="+",
        default=["simple", "simple_mix"],
    )
    parser.add_argument("--n-synth", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=_REPO / "experiments" / "MOPRA_Count" / "figures" / "morphology",
    )
    parser.add_argument("--tag", type=str, default="")
    args = parser.parse_args()

    by_pos = _parse_dat(args.dat)
    real_st = _real_component_stats(by_pos)
    print("Real Scouse:", json.dumps(_summary("real", real_st), indent=2))

    synth_map = {}
    summaries = [_summary("real", real_st)]
    for preset in args.gen_presets:
        cfg = build_mopra_synth_cfg(
            repo_root=_REPO,
            gen_preset=preset,
            max_components=6,
            noise_calibration_cube=args.cube,
        )
        st = _synth_stats(cfg, n=args.n_synth, seed=args.seed + hash(preset) % 1000)
        synth_map[preset] = st
        s = _summary(preset, st)
        summaries.append(s)
        print(f"{preset}:", json.dumps(s, indent=2))

    stem = args.tag.strip() or "synth_vs_real"
    out_dir = args.out_dir
    plot_histograms(real_st, synth_map, out_dir / f"{stem}_hists.png")

    real_ex = _load_real_spectrum_examples(args.cube, by_pos, n_each=3, seed=args.seed)
    synth_ex_map = {
        name: _pick_synth_examples(st["examples"], (1, 2, 3), 3, seed=args.seed + 3)
        for name, st in synth_map.items()
    }
    plot_spectrum_grid(real_ex, synth_ex_map, out_dir / f"{stem}_spectra.png")

    json_path = out_dir / f"{stem}_summary.json"
    json_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
