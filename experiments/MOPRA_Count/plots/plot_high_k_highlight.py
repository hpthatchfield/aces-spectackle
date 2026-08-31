#!/usr/bin/env python
"""
High-K highlight gallery: what we get right vs near-miss vs worse under-count.

For hm_k / Scheme B, K_true=4 is often all under-count (0 exact). This panel shows:
  - K_true=3 exact (high-K successes that do exist)
  - K_true=4 near-miss (K_pred=3, dK=-1)
  - K_true=4 worse (K_pred<=2, dK<=-2)

Example:
  python \\
    experiments/MOPRA_Count/plots/plot_high_k_highlight.py \\
    --k-pred data/mopra_cmz_k_pred_hm_k_simple_k6_20k.fits \\
    --pred-centers data/mopra_cmz_k_pred_hm_k_simple_k6_20k_centers.npz
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPT = Path(__file__).resolve()
_MOPRA = _SCRIPT.parent.parent
_REPO = _MOPRA.parents[1]
sys.path.insert(0, str(_SCRIPT.parent))
sys.path.insert(0, str(_REPO / "src"))

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from spectral_cube import SpectralCube

from plot_cube_discrepancy_spectra import (  ### noqa: E402
    PixelCase,
    _build_cases,
    _plot_panel,
    load_pred_centers_lookup,
    pred_centers_for_case,
    scouse_components_by_pos,
)
from plot_k_residual_map import k_true_map_from_dat  ### noqa: E402
from plot_style import COL_FAIL, COL_PRED, COL_SUCCESS  ### noqa: E402
from spectackle.wcs_plot import wcs_celestial  ### noqa: E402

### Near-miss: under by exactly 1 (still useful as "almost right").
COL_NEAR = COL_PRED


def _sample(pool: list[PixelCase], *, n: int, rng: np.random.Generator) -> list[PixelCase]:
    if not pool or n <= 0:
        return []
    take = min(n, len(pool))
    idx = rng.choice(len(pool), size=take, replace=False)
    return [pool[int(i)] for i in sorted(idx.tolist())]


def _pool_summary(cases: list[PixelCase], k_true: int) -> dict:
    sub = [c for c in cases if c.k_true == k_true]
    d = np.asarray([c.delta for c in sub], dtype=int)
    return {
        "n": int(len(sub)),
        "n_exact": int(np.sum(d == 0)),
        "n_near_under1": int(np.sum(d == -1)),
        "n_worse_le_m2": int(np.sum(d <= -2)),
        "delta_counts": {str(int(k)): int(v) for k, v in zip(*np.unique(d, return_counts=True))}
        if d.size
        else {},
    }


def plot_high_k_highlight(
    *,
    exact_k3: list[PixelCase],
    near_k4: list[PixelCase],
    worse_k4: list[PixelCase],
    spec_cube: np.ndarray,
    v_axis: np.ndarray,
    out: Path,
    vel_range: tuple[float, float] | None,
    pred_lookup: dict | None,
    snr_vel_range: tuple[float, float] | None,
    n_col: int,
) -> Path:
    sections: list[tuple[str, list[PixelCase], str]] = [
        (
            f"K_true=3 exact  (n_show={len(exact_k3)}; high-K successes)",
            exact_k3,
            COL_SUCCESS,
        ),
        (
            f"K_true=4 near-miss  K_pred=3  dK=-1  (n_show={len(near_k4)}; best available for K=4)",
            near_k4,
            COL_NEAR,
        ),
        (
            f"K_true=4 worse under-count  K_pred<=2  dK<=-2  (n_show={len(worse_k4)})",
            worse_k4,
            COL_FAIL,
        ),
    ]
    sections = [(lab, cs, col) for lab, cs, col in sections if cs]
    if not sections:
        raise RuntimeError("No panels to plot for high-K highlight.")

    section_rows = [max(1, (len(cs) + n_col - 1) // n_col) for _, cs, _ in sections]
    fig_h = 2.55 * sum(section_rows) + 0.75 * len(sections)
    fig = plt.figure(figsize=(4.7 * n_col, fig_h))
    outer = fig.add_gridspec(
        len(sections),
        1,
        height_ratios=section_rows,
        hspace=0.50,
        left=0.055,
        right=0.98,
        top=0.92,
        bottom=0.05,
    )

    section_first_axes: list[tuple[str, str, object]] = []
    n_sec = len(sections)
    for sec_i, ((label, cs, color), n_rows) in enumerate(zip(sections, section_rows)):
        inner = outer[sec_i].subgridspec(n_rows, n_col, hspace=0.28, wspace=0.22)
        first_ax = None
        last_section = sec_i == n_sec - 1
        for j, case in enumerate(cs):
            r, c = divmod(j, n_col)
            ax = fig.add_subplot(inner[r, c])
            if first_ax is None:
                first_ax = ax
            pred_v = pred_centers_for_case(case, pred_lookup) if pred_lookup else None
            _plot_panel(
                ax,
                case,
                spec_cube=spec_cube,
                v_axis=v_axis,
                vlim=vel_range,
                snr_vel_range=snr_vel_range,
                line_color=color,
                vline_color="0.25",
                show_xlabel=(r == n_rows - 1 and last_section),
                show_xtick_labels=(r == n_rows - 1),
                show_ylabel=(c == 0),
                show_ytick_labels=(c == 0),
                show_heatmap_ylabel=False,
                compact_title=True,
                include_centers=True,
                pad_kms=10.0,
                pred_v=pred_v,
                heatmap_prob=None,
            )
        if first_ax is not None:
            section_first_axes.append((label, color, first_ax))

    fig.canvas.draw()
    for label, color, ax in section_first_axes:
        pos = ax.get_position()
        fig.text(
            pos.x0,
            min(pos.y1 + 0.018, 0.955),
            label,
            fontsize=10,
            fontweight="bold",
            color=color,
            ha="left",
            va="bottom",
        )

    fig.suptitle(
        "High-K highlight: Scouse centers (dark) vs ML centers (orange). "
        "K_true=4 has 0 exact on this run.",
        fontsize=11,
        y=0.985,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="High-K exact / near-miss / worse gallery.")
    parser.add_argument(
        "--k-pred",
        type=Path,
        default=_REPO / "data" / "mopra_cmz_k_pred_hm_k_simple_k6_20k.fits",
    )
    parser.add_argument("--cube", type=Path, default=_REPO / "data" / "CMZ_3mm_HNCO_60.fits")
    parser.add_argument("--dat", type=Path, default=_REPO / "data" / "final_fits_updated.dat")
    parser.add_argument(
        "--pred-centers",
        type=Path,
        default=_REPO / "data" / "mopra_cmz_k_pred_hm_k_simple_k6_20k_centers.npz",
    )
    parser.add_argument("--n-each", type=int, default=8, help="Spectra per section.")
    parser.add_argument("--n-col", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vel-min", type=float, default=40.0)
    parser.add_argument("--vel-max", type=float, default=140.0)
    parser.add_argument(
        "--out",
        type=Path,
        default=_MOPRA / "figures" / "failure_spectra" / "hm_k_simple_k6_20k_high_k_highlight.png",
    )
    parser.add_argument("--cases-json", type=Path, default=None)
    args = parser.parse_args()

    print(f"Loading cube: {args.cube}", flush=True)
    cube = SpectralCube.read(str(args.cube.resolve()), use_dask=False)
    spec = np.asarray(cube.filled(np.nan), dtype=np.float32)
    if spec.ndim == 4:
        spec = spec[0]
    v_axis = cube.spectral_axis.to("km/s").value.astype(np.float64)
    wcs = wcs_celestial(cube.header)

    k_pred = fits.getdata(args.k_pred).astype(np.float32)
    k_true, _ = k_true_map_from_dat(args.dat, shape=k_pred.shape, wcs=wcs)
    by_pos = scouse_components_by_pos(args.dat)
    snr_vel_range = (float(args.vel_min), float(args.vel_max))
    cases = _build_cases(
        k_pred,
        k_true,
        wcs=wcs,
        by_pos=by_pos,
        spec_cube=spec,
        v_axis=v_axis,
        snr_vel_range=snr_vel_range,
    )
    print(f"Comparable labeled pixels: {len(cases)}", flush=True)

    sum3 = _pool_summary(cases, 3)
    sum4 = _pool_summary(cases, 4)
    print(f"K_true=3 pool: {sum3}", flush=True)
    print(f"K_true=4 pool: {sum4}", flush=True)

    rng = np.random.default_rng(args.seed)
    exact_k3 = _sample(
        [c for c in cases if c.k_true == 3 and c.delta == 0],
        n=args.n_each,
        rng=rng,
    )
    near_k4 = _sample(
        [c for c in cases if c.k_true == 4 and c.delta == -1],
        n=args.n_each,
        rng=rng,
    )
    worse_k4 = _sample(
        [c for c in cases if c.k_true == 4 and c.delta <= -2],
        n=args.n_each,
        rng=rng,
    )
    print(
        f"Show: exact_k3={len(exact_k3)}  near_k4={len(near_k4)}  worse_k4={len(worse_k4)}",
        flush=True,
    )

    pred_lookup = None
    if args.pred_centers is not None and args.pred_centers.is_file():
        pred_lookup = load_pred_centers_lookup(args.pred_centers)
        print(f"Loaded ML centers for {len(pred_lookup)} pixels", flush=True)

    vel_range = (float(args.vel_min), float(args.vel_max))
    out = plot_high_k_highlight(
        exact_k3=exact_k3,
        near_k4=near_k4,
        worse_k4=worse_k4,
        spec_cube=spec,
        v_axis=v_axis,
        out=args.out,
        vel_range=vel_range,
        pred_lookup=pred_lookup,
        snr_vel_range=vel_range,
        n_col=int(args.n_col),
    )
    print(f"Wrote {out}", flush=True)

    meta = {
        "k_pred": str(args.k_pred.resolve()),
        "pool_k3": sum3,
        "pool_k4": sum4,
        "exact_k3": [
            {"l": c.l, "b": c.b, "yi": c.yi, "xi": c.xi, "k_pred": c.k_pred, "delta": c.delta}
            for c in exact_k3
        ],
        "near_k4": [
            {"l": c.l, "b": c.b, "yi": c.yi, "xi": c.xi, "k_pred": c.k_pred, "delta": c.delta}
            for c in near_k4
        ],
        "worse_k4": [
            {"l": c.l, "b": c.b, "yi": c.yi, "xi": c.xi, "k_pred": c.k_pred, "delta": c.delta}
            for c in worse_k4
        ],
    }
    cases_json = args.cases_json or out.with_suffix(".json")
    cases_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote {cases_json}", flush=True)


if __name__ == "__main__":
    main()
