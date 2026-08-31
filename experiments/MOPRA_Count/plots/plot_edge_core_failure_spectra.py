#!/usr/bin/env python
"""
Edge over-count vs core under-count failure-mode spectra (real cube, noisy).

Compares the two sides of the Scouse residual tradeoff for a K_pred map:
  - Edge: K_true=1, K_pred>=2, near labeled-region edge
  - Core: K_true>=3, K_pred=1, deep in labeled interior

Uses the same smoothed cube spectra the model is fit on (noise included).

Example:
  python experiments/MOPRA_Count/plots/plot_edge_core_failure_spectra.py \\
    --k-pred data/mopra_cmz_k_pred_scouse_dat_repro.fits \\
    --cube data/CMZ_3mm_HNCO_60.fits
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
from astropy.wcs import WCS
from scipy.ndimage import distance_transform_edt
from spectral_cube import SpectralCube

from plot_cube_discrepancy_spectra import (  ### noqa: E402
    PixelCase,
    _plot_panel,
    load_heatmap_probs_for_cases,
    load_pred_centers_lookup,
    pred_centers_for_case,
    scouse_components_by_pos,
)
from plot_k_residual_map import k_true_map_from_dat  ### noqa: E402
from plot_style import COL_FAIL, COL_SPEC, COL_SUCCESS  ### noqa: E402
from spectackle.data.mopra_preprocess import (  ### noqa: E402
    snr_peak_rms_mopra,
    snr_peak_scouse_mopra,
)


def _lb_from_pixel(yi: int, xi: int, wcs: WCS) -> tuple[float, float]:
    l, b = wcs.all_pix2world(xi, yi, 0)
    return float(l), float(b)


def build_failure_cases(
    k_pred: np.ndarray,
    k_true: np.ndarray,
    *,
    wcs: WCS,
    by_pos: dict,
    spec_cube: np.ndarray,
    v_axis: np.ndarray,
    snr_vel_range: tuple[float, float] | None,
) -> tuple[list[PixelCase], np.ndarray]:
    """All labeled+finite pixels with dist_to_edge on the labeled mask."""
    labeled = np.isfinite(k_true)
    compare = labeled & np.isfinite(k_pred)
    dist = distance_transform_edt(labeled)
    cases: list[PixelCase] = []
    dists: list[float] = []
    ys, xs = np.where(compare)
    for yi, xi in zip(ys.tolist(), xs.tolist()):
        l, b = _lb_from_pixel(int(yi), int(xi), wcs)
        key = (round(l, 5), round(b, 5))
        rows = by_pos.get(key, [])
        comp_v = tuple(float(r[5]) for r in rows)
        spec = spec_cube[:, int(yi), int(xi)].astype(np.float64)
        kt = int(k_true[yi, xi])
        kp = int(round(float(k_pred[yi, xi])))
        cases.append(
            PixelCase(
                yi=int(yi),
                xi=int(xi),
                l=l,
                b=b,
                k_true=kt,
                k_pred=kp,
                delta=kp - kt,
                snr_global=float(snr_peak_rms_mopra(spec)),
                snr_scouse=float(
                    snr_peak_scouse_mopra(spec, vel_kms=v_axis, vel_range=snr_vel_range)
                ),
                comp_v_kms=comp_v,
            )
        )
        dists.append(float(dist[yi, xi]))
    return cases, np.asarray(dists, dtype=np.float64)


def _sample_pool(
    pool: list[tuple[PixelCase, float]],
    *,
    rng: np.random.Generator,
    n: int,
    prefer_large_abs_delta: bool = True,
) -> list[PixelCase]:
    if not pool:
        return []
    if prefer_large_abs_delta:
        pool = sorted(pool, key=lambda t: abs(t[0].delta), reverse=True)
        half = pool[: max(n, len(pool) // 2)]
    else:
        half = pool
    if len(half) <= n:
        return [c for c, _ in half]
    idx = rng.choice(len(half), size=n, replace=False)
    return [half[int(i)][0] for i in np.sort(idx)]


def plot_edge_core_failures(
    edge_over: list[PixelCase],
    core_under: list[PixelCase],
    edge_ok: list[PixelCase],
    *,
    spec_cube: np.ndarray,
    v_axis: np.ndarray,
    out: Path,
    vlim: tuple[float, float] | None = None,
    snr_vel_range: tuple[float, float] | None = None,
    n_col: int = 4,
    pred_lookup: dict[tuple[int, int], tuple[float, ...]] | None = None,
    heatmap_probs: dict[tuple[int, int], np.ndarray] | None = None,
) -> Path:
    sections: list[tuple[str, list[PixelCase], str]] = [
        (
            "Edge over-count: K_true=1, K_pred>=2, near labeled edge (avoid inventing components)",
            edge_over,
            COL_FAIL,
        ),
        (
            "Core under-count: K_true>=3, K_pred=1, labeled interior (learn glance-visible blends)",
            core_under,
            COL_SPEC,
        ),
    ]
    if edge_ok:
        sections.append(
            (
                "Edge K=1 correct: K_true=K_pred=1 near edge (keep these as singles)",
                edge_ok,
                COL_SUCCESS,
            )
        )

    n_col = max(1, int(n_col))
    section_rows = [max(1, (len(cases) + n_col - 1) // n_col) for _, cases, _ in sections]
    height_ratios: list[float] = []
    for n_rows in section_rows:
        height_ratios.append(0.08)
        height_ratios.extend([1.0] * n_rows)

    fig = plt.figure(
        figsize=(3.8 * n_col, 2.7 * sum(section_rows) + 0.5 * len(sections))
    )
    gs = fig.add_gridspec(
        len(height_ratios),
        n_col,
        height_ratios=height_ratios,
        hspace=0.55,
        wspace=0.35 if heatmap_probs else 0.28,
        left=0.07,
        right=0.96 if heatmap_probs else 0.98,
        top=0.90,
        bottom=0.08,
    )
    share_x = None
    grid_row = 0
    for (label, cases, color), n_rows in zip(sections, section_rows):
        ax_label = fig.add_subplot(gs[grid_row, :])
        ax_label.axis("off")
        note = ""
        if pred_lookup is not None:
            note = "  |  dotted=Scouse  dashed=ML"
        if heatmap_probs is not None:
            note += "  purple=P(center)"
        ax_label.text(
            0.0, 0.5, label + note, ha="left", va="center", fontsize=10, color=color, fontweight="bold"
        )
        grid_row += 1
        for case_row in range(n_rows):
            for col_i in range(n_col):
                case_i = case_row * n_col + col_i
                ax = fig.add_subplot(gs[grid_row, col_i], sharex=share_x)
                if share_x is None:
                    share_x = ax
                if case_i < len(cases):
                    ### x-axis only where no panel sits below in this section.
                    is_bottom = case_i + n_col >= len(cases)
                    case = cases[case_i]
                    key = (case.yi, case.xi)
                    _plot_panel(
                        ax,
                        case,
                        spec_cube=spec_cube,
                        v_axis=v_axis,
                        vlim=vlim,
                        snr_vel_range=snr_vel_range,
                        line_color=color,
                        vline_color=color,
                        show_xlabel=is_bottom,
                        pred_v=pred_centers_for_case(case, pred_lookup) if pred_lookup is not None else None,
                        heatmap_prob=None if heatmap_probs is None else heatmap_probs.get(key),
                    )
                else:
                    ax.set_visible(False)
            grid_row += 1

    v_lo = float(np.min(v_axis)) if vlim is None else float(vlim[0])
    v_hi = float(np.max(v_axis)) if vlim is None else float(vlim[1])
    fig.suptitle(
        f"Failure-mode spectra (noisy cube, v=[{v_lo:.0f}, {v_hi:.0f}] km/s); "
        "dotted v = Scouse component centers"
        + ("; dashed = ML centers" if pred_lookup is not None else ""),
        fontsize=11,
        y=0.97,
    )
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Edge over-count vs core under-count spectra.")
    parser.add_argument("--k-pred", type=Path, required=True)
    parser.add_argument("--cube", type=Path, default=_REPO / "data" / "CMZ_3mm_HNCO_60.fits")
    parser.add_argument("--dat", type=Path, default=_REPO / "data" / "final_fits_updated.dat")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--n-each", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--edge-max-dist", type=float, default=2.0, help="Max dist_to_edge (pix) for edge rows.")
    parser.add_argument("--core-min-dist", type=float, default=6.0, help="Min dist_to_edge (pix) for core row.")
    parser.add_argument("--core-k-min", type=int, default=3)
    parser.add_argument("--include-edge-ok", action="store_true", help="Add a third row of correct edge K=1.")
    parser.add_argument("--vel-range", type=float, nargs=2, default=None, metavar=("V_MIN", "V_MAX"))
    parser.add_argument("--snr-vel-range", type=float, nargs=2, default=None, metavar=("V_MIN", "V_MAX"))
    parser.add_argument("--n-col", type=int, default=4)
    parser.add_argument(
        "--pred-centers",
        type=Path,
        default=None,
        help="Optional *_centers.npz from run_cube_heatmap_map (ML peak velocities).",
    )
    parser.add_argument(
        "--heatmap-run-dir",
        type=Path,
        default=None,
        help="Optional heatmap training run dir; overlays P(center) for plotted panels.",
    )
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    k_pred = fits.getdata(args.k_pred).astype(np.float32)
    cube_header = fits.getheader(args.cube)
    ny, nx = cube_header["NAXIS2"], cube_header["NAXIS1"]
    if k_pred.shape != (ny, nx):
        raise ValueError(f"k-pred shape {k_pred.shape} != cube ({ny}, {nx})")

    wcs = WCS(cube_header).celestial
    k_true, _ = k_true_map_from_dat(args.dat, shape=k_pred.shape, wcs=wcs)
    by_pos = scouse_components_by_pos(args.dat)

    print(f"Loading cube: {args.cube}", flush=True)
    cube = SpectralCube.read(str(args.cube.resolve()), use_dask=False)
    spec_cube = np.asarray(cube.filled(np.nan), dtype=np.float32)
    if spec_cube.ndim == 4:
        spec_cube = spec_cube[0]
    v_axis = cube.spectral_axis.to("km/s").value.astype(np.float64)

    vlim = tuple(args.vel_range) if args.vel_range is not None else None
    if args.snr_vel_range is not None:
        snr_vel_range = tuple(args.snr_vel_range)
    elif vlim is not None:
        snr_vel_range = vlim
    else:
        snr_vel_range = None

    cases, dists = build_failure_cases(
        k_pred,
        k_true,
        wcs=wcs,
        by_pos=by_pos,
        spec_cube=spec_cube,
        v_axis=v_axis,
        snr_vel_range=snr_vel_range,
    )
    print(f"Comparable labeled pixels: {len(cases)}", flush=True)

    edge_over_pool = [
        (c, d)
        for c, d in zip(cases, dists)
        if c.k_true == 1 and c.k_pred >= 2 and d <= float(args.edge_max_dist)
    ]
    core_under_pool = [
        (c, d)
        for c, d in zip(cases, dists)
        if c.k_true >= int(args.core_k_min) and c.k_pred == 1 and d >= float(args.core_min_dist)
    ]
    edge_ok_pool = [
        (c, d)
        for c, d in zip(cases, dists)
        if c.k_true == 1 and c.k_pred == 1 and d <= float(args.edge_max_dist)
    ]
    print(
        f"  edge over-count pool: {len(edge_over_pool)}  "
        f"core under-count pool: {len(core_under_pool)}  "
        f"edge K=1 ok pool: {len(edge_ok_pool)}",
        flush=True,
    )

    rng = np.random.default_rng(args.seed)
    n = int(args.n_each)
    edge_over = _sample_pool(edge_over_pool, rng=rng, n=n)
    core_under = _sample_pool(core_under_pool, rng=rng, n=n)
    edge_ok = _sample_pool(edge_ok_pool, rng=rng, n=n, prefer_large_abs_delta=False) if args.include_edge_ok else []

    out = args.out
    if out is None:
        out = _MOPRA / "figures" / f"{args.k_pred.stem}_edge_core_failures.png"

    pred_lookup = None
    if args.pred_centers is not None:
        pred_lookup = load_pred_centers_lookup(args.pred_centers)
        print(f"Loaded ML centers for {len(pred_lookup)} pixels from {args.pred_centers}", flush=True)

    heatmap_probs = None
    if args.heatmap_run_dir is not None:
        selected = list(edge_over) + list(core_under) + list(edge_ok)
        seen: set[tuple[int, int]] = set()
        uniq: list[PixelCase] = []
        for c in selected:
            key = (c.yi, c.xi)
            if key in seen:
                continue
            seen.add(key)
            uniq.append(c)
        print(f"Computing P(center) for {len(uniq)} panels from {args.heatmap_run_dir}", flush=True)
        heatmap_probs = load_heatmap_probs_for_cases(
            uniq,
            run_dir=args.heatmap_run_dir,
            spec_cube=spec_cube,
            device=args.device,
        )

    fig_path = plot_edge_core_failures(
        edge_over,
        core_under,
        edge_ok,
        spec_cube=spec_cube,
        v_axis=v_axis,
        out=out,
        vlim=vlim,
        snr_vel_range=snr_vel_range,
        n_col=int(args.n_col),
        pred_lookup=pred_lookup,
        heatmap_probs=heatmap_probs,
    )

    summary = {
        "k_pred": str(args.k_pred.resolve()),
        "cube": str(args.cube.resolve()),
        "dat": str(args.dat.resolve()),
        "seed": int(args.seed),
        "edge_max_dist": float(args.edge_max_dist),
        "core_min_dist": float(args.core_min_dist),
        "core_k_min": int(args.core_k_min),
        "n_edge_over_pool": len(edge_over_pool),
        "n_core_under_pool": len(core_under_pool),
        "n_edge_ok_pool": len(edge_ok_pool),
        "spec_source": "noisy_cube",
        "edge_over": [
            {"l": c.l, "b": c.b, "yi": c.yi, "xi": c.xi, "k_true": c.k_true, "k_pred": c.k_pred, "delta": c.delta}
            for c in edge_over
        ],
        "core_under": [
            {"l": c.l, "b": c.b, "yi": c.yi, "xi": c.xi, "k_true": c.k_true, "k_pred": c.k_pred, "delta": c.delta}
            for c in core_under
        ],
        "edge_ok": [
            {"l": c.l, "b": c.b, "yi": c.yi, "xi": c.xi, "k_true": c.k_true, "k_pred": c.k_pred, "delta": c.delta}
            for c in edge_ok
        ],
    }
    cases_json = out.with_suffix(".json")
    cases_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {fig_path}", flush=True)
    print(f"Wrote {cases_json}", flush=True)


if __name__ == "__main__":
    main()
