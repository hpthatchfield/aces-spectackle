#!/usr/bin/env python
"""
Residual stats vs Henshaw K_true, plus failure-example spectra per K bin.

Example:
  python experiments/MOPRA_Count/plots/plot_residual_by_k_failures.py \\
    --k-pred data/mopra_cmz_k_pred_heatmap_realamp_k6_20k.fits \\
    --pred-centers data/mopra_cmz_k_pred_heatmap_realamp_k6_20k_centers.npz \\
    --heatmap-run-dir experiments/MOPRA_Count/runs/mopra_heatmap_<ts>_<tag>
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
    load_heatmap_probs_for_cases,
    load_pred_centers_lookup,
    pred_centers_for_case,
    scouse_components_by_pos,
)
from plot_k_residual_map import k_true_map_from_dat  ### noqa: E402
from plot_style import COL_FAIL, COL_SPEC, COL_SUCCESS  ### noqa: E402
from spectackle.wcs_plot import wcs_celestial  ### noqa: E402


def _stats_by_k(cases: list[PixelCase]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    ks = sorted({c.k_true for c in cases})
    for k in ks:
        sub = [c for c in cases if c.k_true == k]
        d = np.asarray([c.delta for c in sub], dtype=np.float64)
        out[str(k)] = {
            "n": int(len(sub)),
            "mae": float(np.mean(np.abs(d))),
            "mean_delta": float(np.mean(d)),
            "median_delta": float(np.median(d)),
            "frac_exact": float(np.mean(d == 0)),
            "frac_over": float(np.mean(d > 0)),
            "frac_under": float(np.mean(d < 0)),
            "p10_delta": float(np.percentile(d, 10)),
            "p90_delta": float(np.percentile(d, 90)),
        }
    return out


def _format_table(by_k: dict[str, dict]) -> str:
    header = (
        f"{'K_Henshaw':>10} {'n':>6} {'MAE':>7} {'meandelta':>7} {'meddelta':>6} "
        f"{'exact':>6} {'over':>6} {'under':>6} {'p10delta':>6} {'p90delta':>6}"
    )
    lines = [header, "-" * len(header)]
    for k in sorted(by_k, key=int):
        s = by_k[k]
        lines.append(
            f"{k:>10} {s['n']:6d} {s['mae']:7.3f} {s['mean_delta']:7.3f} "
            f"{s['median_delta']:6.2f} {s['frac_exact']:6.3f} {s['frac_over']:6.3f} "
            f"{s['frac_under']:6.3f} {s['p10_delta']:6.2f} {s['p90_delta']:6.2f}"
        )
    return "\n".join(lines) + "\n"


def _pick_failures(
    cases: list[PixelCase],
    *,
    k_true: int,
    n_over: int,
    n_under: int,
    seed: int,
) -> tuple[list[PixelCase], list[PixelCase]]:
    rng = np.random.default_rng(seed + 17 * k_true)
    over = [c for c in cases if c.k_true == k_true and c.delta > 0]
    under = [c for c in cases if c.k_true == k_true and c.delta < 0]
    ### Prefer largest |dK|, then random among ties.
    over = sorted(over, key=lambda c: (-abs(c.delta), c.yi, c.xi))
    under = sorted(under, key=lambda c: (-abs(c.delta), c.yi, c.xi))
    ### Take top half of severe failures, then sample for diversity.
    def sample_top(pool: list[PixelCase], n: int) -> list[PixelCase]:
        if not pool or n <= 0:
            return []
        top = pool[: max(n * 3, n)]
        take = min(n, len(top))
        idx = rng.choice(len(top), size=take, replace=False)
        return [top[int(i)] for i in sorted(idx.tolist())]

    return sample_top(over, n_over), sample_top(under, n_under)


def plot_failures_by_k(
    cases: list[PixelCase],
    *,
    spec_cube: np.ndarray,
    v_axis: np.ndarray,
    out: Path,
    k_values: list[int],
    n_over: int,
    n_under: int,
    seed: int,
    vel_range: tuple[float, float] | None,
    pred_lookup: dict | None,
    heatmap_probs: dict | None,
    snr_vel_range: tuple[float, float] | None,
) -> tuple[Path, dict]:
    sections: list[tuple[str, list[PixelCase], str]] = []
    meta_cases: dict[str, dict] = {}
    for k in k_values:
        over, under = _pick_failures(
            cases, k_true=k, n_over=n_over, n_under=n_under, seed=seed
        )
        meta_cases[str(k)] = {
            "over": [
                {"l": c.l, "b": c.b, "yi": c.yi, "xi": c.xi, "k_pred": c.k_pred, "delta": c.delta}
                for c in over
            ],
            "under": [
                {"l": c.l, "b": c.b, "yi": c.yi, "xi": c.xi, "k_pred": c.k_pred, "delta": c.delta}
                for c in under
            ],
        }
        if over:
            sections.append((f"K_Henshaw={k} over-count failures", over, COL_FAIL))
        if under:
            sections.append((f"K_Henshaw={k} under-count failures", under, COL_FAIL))
        ### If no failures, show a few exact matches for context.
        if not over and not under:
            exact = [c for c in cases if c.k_true == k and c.delta == 0]
            if exact:
                rng = np.random.default_rng(seed + k)
                take = min(n_over, len(exact))
                pick = rng.choice(len(exact), size=take, replace=False)
                chosen = [exact[int(i)] for i in pick]
                sections.append((f"K_Henshaw={k} exact (no failures in bin)", chosen, COL_SUCCESS))

    if not sections:
        raise RuntimeError("No panels to plot.")

    n_col = max(n_over, n_under, 1)
    section_rows = [max(1, (len(cs) + n_col - 1) // n_col) for _, cs, _ in sections]
    ### Nested gridspec: gap between sections is larger than gap within a section.
    fig_h = 2.55 * sum(section_rows) + 0.7 * len(sections)
    fig = plt.figure(figsize=(4.7 * n_col, fig_h))
    outer = fig.add_gridspec(
        len(sections),
        1,
        height_ratios=section_rows,
        hspace=0.48,
        left=0.055,
        right=0.91 if heatmap_probs else 0.98,
        top=0.93,
        bottom=0.05,
    )

    section_first_axes: list[tuple[str, str, object]] = []
    n_sec = len(sections)
    for sec_i, ((label, cs, color), n_rows) in enumerate(zip(sections, section_rows)):
        inner = outer[sec_i].subgridspec(
            n_rows,
            n_col,
            hspace=0.28,
            wspace=0.28 if heatmap_probs else 0.22,
        )
        first_ax = None
        last_section = sec_i == n_sec - 1
        for j, case in enumerate(cs):
            r = j // n_col
            c = j % n_col
            ax = fig.add_subplot(inner[r, c])
            if first_ax is None:
                first_ax = ax
            pred_v = pred_centers_for_case(case, pred_lookup) if pred_lookup else None
            hprob = None if heatmap_probs is None else heatmap_probs.get((case.yi, case.xi))
            bottom_row = r == n_rows - 1
            _plot_panel(
                ax,
                case,
                spec_cube=spec_cube,
                v_axis=v_axis,
                vlim=vel_range,
                snr_vel_range=snr_vel_range,
                line_color=color,
                vline_color=COL_SPEC,
                ### One shared x-label on the last section only; keep ticks on every bottom row.
                show_xlabel=(bottom_row and last_section),
                show_xtick_labels=bottom_row,
                show_ylabel=(c == 0),
                show_ytick_labels=(c == 0),
                show_heatmap_ylabel=(heatmap_probs is not None and c == n_col - 1),
                compact_title=True,
                include_centers=True,
                pad_kms=10.0,
                pred_v=pred_v,
                heatmap_prob=hprob,
            )
        if first_ax is not None:
            section_first_axes.append((label, color, first_ax))

    fig.canvas.draw()
    for label, color, ax in section_first_axes:
        pos = ax.get_position()
        fig.text(
            pos.x0,
            min(pos.y1 + 0.018, 0.965),
            label,
            fontsize=10,
            fontweight="bold",
            color=color,
            ha="left",
            va="bottom",
        )

    fig.suptitle(
        "Residual failures by Henshaw K  (blue Scouse centers, orange ML centers"
        + (", purple=P(center)" if heatmap_probs is not None else "")
        + ")",
        fontsize=11,
        y=0.99,
    )
    ### Avoid tight_layout: it fights gridspec margins and re-crowds twin axes.
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out, meta_cases


def main() -> None:
    parser = argparse.ArgumentParser(description="By-K residual stats + failure spectra.")
    parser.add_argument(
        "--k-pred",
        type=Path,
        default=_REPO / "data" / "mopra_cmz_k_pred_heatmap_realamp_k6_20k.fits",
    )
    parser.add_argument("--cube", type=Path, default=_REPO / "data" / "CMZ_3mm_HNCO_60.fits")
    parser.add_argument("--dat", type=Path, default=_REPO / "data" / "final_fits_updated.dat")
    parser.add_argument(
        "--pred-centers",
        type=Path,
        default=_REPO / "data" / "mopra_cmz_k_pred_heatmap_realamp_k6_20k_centers.npz",
    )
    parser.add_argument(
        "--heatmap-run-dir",
        type=Path,
        required=True,
        help="Heatmap train folder with center_heatmap_net.pt + manifest.json.",
    )
    parser.add_argument("--out-prefix", type=Path, default=None)
    parser.add_argument("--k-values", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--n-over", type=int, default=4)
    parser.add_argument("--n-under", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vel-range", type=float, nargs=2, default=[40.0, 140.0])
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    stem = args.k_pred.stem
    prefix = args.out_prefix or (_MOPRA / "figures" / "failure_spectra" / f"{stem}_by_k")
    prefix.parent.mkdir(parents=True, exist_ok=True)

    with fits.open(args.k_pred) as hdul:
        k_pred = np.asarray(hdul[0].data, dtype=np.float64)
    cube = SpectralCube.read(str(args.cube), use_dask=False)
    spec_cube = np.asarray(cube.unmasked_data[:].value, dtype=np.float64)
    wcs = wcs_celestial(cube.header)
    v_axis = cube.spectral_axis.to("km/s").value.astype(np.float64)
    k_true, _ = k_true_map_from_dat(args.dat, shape=k_pred.shape, wcs=wcs)
    by_pos = scouse_components_by_pos(args.dat)
    vel_range = tuple(args.vel_range) if args.vel_range is not None else None

    cases = _build_cases(
        k_pred,
        k_true,
        wcs=wcs,
        by_pos=by_pos,
        spec_cube=spec_cube,
        v_axis=v_axis,
        snr_vel_range=vel_range,
    )
    by_k = _stats_by_k(cases)
    table = _format_table(by_k)
    print(table)

    stats_path = Path(str(prefix) + "_stats.json")
    stats_path.write_text(json.dumps({"by_k_true": by_k, "n_compare": len(cases)}, indent=2), encoding="utf-8")
    print(f"Wrote {stats_path}")

    pred_lookup = None
    if args.pred_centers is not None and args.pred_centers.is_file():
        pred_lookup = load_pred_centers_lookup(args.pred_centers)

    ### Collect failure cases first so we only run heatmap inference on those pixels.
    fail_pool: list[PixelCase] = []
    for k in args.k_values:
        o, u = _pick_failures(cases, k_true=k, n_over=args.n_over, n_under=args.n_under, seed=args.seed)
        fail_pool.extend(o)
        fail_pool.extend(u)

    heatmap_probs = None
    if args.heatmap_run_dir is not None and args.heatmap_run_dir.is_dir() and fail_pool:
        heatmap_probs = load_heatmap_probs_for_cases(
            fail_pool,
            run_dir=args.heatmap_run_dir,
            spec_cube=spec_cube,
            device=args.device,
        )

    fig_path = Path(str(prefix) + "_failures.png")
    fig_path, meta = plot_failures_by_k(
        cases,
        spec_cube=spec_cube,
        v_axis=v_axis,
        out=fig_path,
        k_values=list(args.k_values),
        n_over=args.n_over,
        n_under=args.n_under,
        seed=args.seed,
        vel_range=vel_range,
        pred_lookup=pred_lookup,
        heatmap_probs=heatmap_probs,
        snr_vel_range=vel_range,
    )
    cases_path = Path(str(prefix) + "_failures.json")
    cases_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote {fig_path}")
    print(f"Wrote {cases_path}")


if __name__ == "__main__":
    main()
