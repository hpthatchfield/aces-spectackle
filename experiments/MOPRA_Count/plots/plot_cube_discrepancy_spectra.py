#!/usr/bin/env python
"""
Spot-check smoothed-cube spectra where ML K_pred agrees vs disagrees with Scouse/Henshaw K.

Example:
  python experiments/MOPRA_Count/plots/plot_cube_discrepancy_spectra.py \\
    --k-pred data/mopra_cmz_k_pred_smooth60_snr5.fits \\
    --cube data/CMZ_3mm_HNCO_60.fits \\
    --dat data/final_fits_updated.dat \\
    --out experiments/MOPRA_Count/figures/mopra_discrepancy_spectra_snr5.png
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
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
from spectral_cube import SpectralCube

from plot_k_residual_map import k_true_map_from_dat  ### noqa: E402
from plot_style import COL_FAIL, COL_HEAT, COL_PRED, COL_SPEC, COL_SUCCESS  ### noqa: E402
from spectackle.data.mopra_preprocess import (  ### noqa: E402
    prepare_mopra_input,
    snr_peak_rms_mopra,
    snr_peak_scouse_mopra,
)


@dataclass(frozen=True)
class PixelCase:
    yi: int
    xi: int
    l: float
    b: float
    k_true: int
    k_pred: int
    delta: int
    snr_global: float
    snr_scouse: float
    comp_v_kms: tuple[float, ...]


def scouse_components_by_pos(dat_path: Path) -> dict[tuple[float, float], list[np.ndarray]]:
    """Group .dat rows by (l, b); each row is one Gaussian component."""
    arr = np.loadtxt(dat_path)
    by_pos: dict[tuple[float, float], list[np.ndarray]] = defaultdict(list)
    for row in arr:
        key = (round(float(row[1]), 5), round(float(row[2]), 5))
        by_pos[key].append(row)
    return dict(by_pos)


def _lb_from_pixel(yi: int, xi: int, wcs: WCS) -> tuple[float, float]:
    l, b = wcs.all_pix2world(xi, yi, 0)
    return float(l), float(b)


def _pixel_from_lb(l: float, b: float, wcs: WCS) -> tuple[int, int]:
    xp, yp = wcs.all_world2pix([[l, b]], 0)[0]
    return int(round(xp)), int(round(yp))


def _build_cases(
    k_pred: np.ndarray,
    k_true: np.ndarray,
    *,
    wcs: WCS,
    by_pos: dict[tuple[float, float], list[np.ndarray]],
    spec_cube: np.ndarray,
    v_axis: np.ndarray,
    snr_vel_range: tuple[float, float] | None,
) -> list[PixelCase]:
    """All labeled pixels with finite K_pred."""
    cases: list[PixelCase] = []
    labeled = np.isfinite(k_true)
    compare = labeled & np.isfinite(k_pred)
    ys, xs = np.where(compare)
    for yi, xi in zip(ys.tolist(), xs.tolist()):
        l, b = _lb_from_pixel(int(yi), int(xi), wcs)
        key = (round(l, 5), round(b, 5))
        rows = by_pos.get(key, [])
        comp_v = tuple(float(r[5]) for r in rows)
        spec = spec_cube[:, int(yi), int(xi)].astype(np.float64)
        kt = int(k_true[yi, xi])
        kp = int(round(k_pred[yi, xi]))
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
                    snr_peak_scouse_mopra(
                        spec,
                        vel_kms=v_axis,
                        vel_range=snr_vel_range,
                    )
                ),
                comp_v_kms=comp_v,
            )
        )
    return cases


def _sample_cases(
    cases: list[PixelCase],
    *,
    rng: np.random.Generator,
    n_each: int,
    agree_max_delta: int,
    min_discrep: int,
) -> tuple[list[PixelCase], list[PixelCase]]:
    agree = [c for c in cases if abs(c.delta) <= agree_max_delta]
    discrep = [c for c in cases if abs(c.delta) >= min_discrep]
    if len(agree) < n_each:
        print(f"Warning: only {len(agree)} agree cases (|dK|<={agree_max_delta}); requested {n_each}")
    if len(discrep) < n_each:
        print(f"Warning: only {len(discrep)} large-discrep cases (|dK|>={min_discrep}); requested {n_each}")
    agree_pick = (
        list(rng.choice(agree, size=n_each, replace=False)) if len(agree) >= n_each else agree
    )
    if len(discrep) >= n_each:
        ### Prefer the largest errors, then randomize within the worst half.
        discrep = sorted(discrep, key=lambda c: abs(c.delta), reverse=True)
        pool = discrep[: max(n_each, len(discrep) // 2)]
        idx = rng.choice(len(pool), size=min(n_each, len(pool)), replace=False)
        discrep_pick = [pool[int(i)] for i in np.sort(idx)]
    else:
        discrep_pick = discrep
    return agree_pick, discrep_pick


def _scouse_vlines(ax, comp_v: tuple[float, ...], *, color: str) -> None:
    for v in comp_v:
        ax.axvline(v, color=color, ls=":", lw=0.9, alpha=0.85)


def _pred_vlines(ax, pred_v: tuple[float, ...] | None, *, color: str = COL_PRED) -> None:
    if not pred_v:
        return
    for v in pred_v:
        if np.isfinite(v):
            ax.axvline(float(v), color=color, ls="--", lw=1.0, alpha=0.9)


def load_pred_centers_lookup(path: Path) -> dict[tuple[int, int], tuple[float, ...]]:
    """Map (yi, xi) -> finite predicted center velocities (km/s)."""
    z = np.load(path)
    yi = np.asarray(z["yi"], dtype=np.int64)
    xi = np.asarray(z["xi"], dtype=np.int64)
    v = np.asarray(z["center_v_kms"], dtype=np.float64)
    out: dict[tuple[int, int], tuple[float, ...]] = {}
    for i in range(yi.size):
        row = v[i]
        vals = tuple(float(x) for x in row[np.isfinite(row)])
        out[(int(yi[i]), int(xi[i]))] = vals
    return out


def pred_centers_for_case(
    case: PixelCase,
    lookup: dict[tuple[int, int], tuple[float, ...]] | None,
) -> tuple[float, ...]:
    if lookup is None:
        return ()
    return lookup.get((case.yi, case.xi), ())


def load_heatmap_probs_for_cases(
    cases: list[PixelCase],
    *,
    run_dir: Path,
    spec_cube: np.ndarray,
    blank_value: float = -1.0,
    device: str = "cpu",
) -> dict[tuple[int, int], np.ndarray]:
    """
    Recompute P(center) curves for the selected panels only.
    Returns (yi, xi) -> length-C probability array.
    """
    import json
    import torch
    from spectackle.data.generator import _make_v_axis
    from spectackle.models import CenterHeatmapNet1DDeep

    run_dir = run_dir.resolve()
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    args = dict(manifest.get("args", {}))
    coord = None
    if manifest.get("coord", {}).get("enabled"):
        v_scale = float(manifest["coord"].get("v_scale_kms", 100.0))
        coord = _make_v_axis(manifest["cfg"]).astype(np.float32) / v_scale
    model = CenterHeatmapNet1DDeep(
        width=int(args.get("width", 96)),
        n_blocks=int(args.get("n_blocks", 6)),
        coord=coord,
    )
    state = torch.load(run_dir / "center_heatmap_net.pt", map_location="cpu")
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    specs = np.stack(
        [spec_cube[:, c.yi, c.xi].astype(np.float64) for c in cases],
        axis=0,
    )
    x_norm, chan_mask = prepare_mopra_input(specs, blank_value=blank_value)
    with torch.no_grad():
        logits = model(
            torch.from_numpy(x_norm).float().to(device),
            torch.from_numpy(chan_mask).float().to(device),
        )
        prob = torch.sigmoid(logits).cpu().numpy()
    return {(c.yi, c.xi): prob[i] for i, c in enumerate(cases)}


def _plot_vlim(
    v_axis: np.ndarray,
    vlim: tuple[float, float] | None,
    *center_groups: tuple[float, ...] | None,
    pad_kms: float = 10.0,
    include_centers: bool = False,
) -> tuple[float, float]:
    """
    Default x-axis: optional preferred window, else full cube velocity span.
    With include_centers=True, expand so every finite Scouse/ML center stays on-axis
    (clipped to the cube spectral range, with pad_kms padding).
    """
    axis_lo = float(np.nanmin(v_axis))
    axis_hi = float(np.nanmax(v_axis))
    vals: list[float] = []
    if vlim is not None:
        vals.extend([float(vlim[0]), float(vlim[1])])
    if include_centers:
        for group in center_groups:
            if not group:
                continue
            for v in group:
                if np.isfinite(v):
                    vals.append(float(v))
    if not vals:
        return axis_lo, axis_hi
    lo = min(vals)
    hi = max(vals)
    if include_centers:
        lo -= float(pad_kms)
        hi += float(pad_kms)
    return max(axis_lo, lo), min(axis_hi, hi)


def _snr_scouse_label(snr: float, snr_vel_range: tuple[float, float] | None) -> str:
    if snr_vel_range is None:
        return f"SNR_scouse~{snr:.1f}"
    lo, hi = snr_vel_range
    return f"SNR_scouse[{lo:.0f},{hi:.0f}]~{snr:.1f}"


def _plot_panel(
    ax,
    case: PixelCase,
    *,
    spec_cube: np.ndarray,
    v_axis: np.ndarray,
    vlim: tuple[float, float] | None,
    snr_vel_range: tuple[float, float] | None,
    line_color: str,
    vline_color: str,
    show_xlabel: bool = True,
    show_xtick_labels: bool = True,
    show_ylabel: bool = True,
    show_ytick_labels: bool = True,
    show_heatmap_ylabel: bool = True,
    compact_title: bool = False,
    include_centers: bool = False,
    pad_kms: float = 10.0,
    pred_v: tuple[float, ...] | None = None,
    heatmap_prob: np.ndarray | None = None,
) -> None:
    spec = spec_cube[:, case.yi, case.xi]
    ax.plot(v_axis, spec, color=COL_SPEC, lw=0.85)
    _scouse_vlines(ax, case.comp_v_kms, color=vline_color)
    _pred_vlines(ax, pred_v, color=COL_PRED)
    if heatmap_prob is not None:
        ax2 = ax.twinx()
        ax2.plot(v_axis, heatmap_prob, color=COL_HEAT, lw=0.9, alpha=0.85)
        ax2.set_ylim(-0.05, 1.05)
        if show_heatmap_ylabel:
            ax2.set_ylabel("P(center)", fontsize=7, color=COL_HEAT)
            ax2.tick_params(axis="y", labelsize=6, colors=COL_HEAT)
        else:
            ax2.set_ylabel("")
            ax2.tick_params(axis="y", labelright=False, length=0)
    n_pred = 0 if not pred_v else sum(1 for v in pred_v if np.isfinite(v))
    sign = f"+{case.delta}" if case.delta > 0 else str(case.delta)
    if compact_title:
        ### Caption inside axes: avoids title/spine collisions in dense galleries.
        cap = (
            f"K={case.k_true}->{case.k_pred} ({sign})  "
            f"SNR~{case.snr_scouse:.1f}"
            + (f"  n_pred={n_pred}" if pred_v is not None else "")
        )
        ax.text(
            0.03,
            0.93,
            cap,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7,
            color=line_color,
            bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "none", "alpha": 0.82},
        )
    else:
        ax.set_title(
            f"({case.l:.3f}, {case.b:.3f})  K={case.k_true}->{case.k_pred} ({sign})\n"
            f"{_snr_scouse_label(case.snr_scouse, snr_vel_range)}  "
            f"SNR_global~{case.snr_global:.1f}  n_scouse={len(case.comp_v_kms)}"
            + (f"  n_pred={n_pred}" if pred_v is not None else ""),
            fontsize=8,
            color=line_color,
            pad=4,
        )
    ### Only the bottom row of a section carries the velocity axis; interior rows
    ### drop label + ticks so panel titles don't collide with the row above.
    if show_xlabel:
        ax.set_xlabel("v (km/s)", fontsize=8)
    else:
        ax.set_xlabel("")
        if not show_xtick_labels:
            ax.tick_params(labelbottom=False)
    if show_ylabel:
        ax.set_ylabel("T (K)", fontsize=8)
    else:
        ax.set_ylabel("")
    if not show_ytick_labels:
        ax.tick_params(labelleft=False)
    ax.tick_params(axis="both", labelsize=7)
    ax.set_xlim(
        _plot_vlim(
            v_axis,
            vlim,
            case.comp_v_kms,
            pred_v,
            pad_kms=pad_kms,
            include_centers=include_centers,
        )
    )


def plot_discrepancy_spectra(
    agree: list[PixelCase],
    discrep: list[PixelCase],
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
    n_col = max(1, n_col)
    sections: list[tuple[str, list[PixelCase], str]] = []
    if agree:
        sections.append(("Agree with Scouse/Henshaw K", agree, COL_SUCCESS))
    if discrep:
        sections.append(("Large discrepancy vs Scouse/Henshaw K", discrep, COL_FAIL))
    if not sections:
        raise ValueError("No cases to plot.")

    ### GridSpec: one small label row + enough spectrum rows for every selected case.
    section_rows = [max(1, (len(cases) + n_col - 1) // n_col) for _, cases, _ in sections]
    height_ratios: list[float] = []
    for n_rows in section_rows:
        height_ratios.append(0.07)
        height_ratios.extend([1.0] * n_rows)

    fig_h = 2.6 * sum(section_rows) + 0.5 * len(sections)
    fig = plt.figure(figsize=(3.8 * n_col, fig_h))
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

    share_x: plt.Axes | None = None
    grid_row = 0
    for (label, cases, color), n_rows in zip(sections, section_rows):
        ax_label = fig.add_subplot(gs[grid_row, :])
        ax_label.axis("off")
        note = ""
        if pred_lookup is not None:
            note = "  |  dotted=Scouse  dashed=ML centers"
        if heatmap_probs is not None:
            note += "  purple=P(center)"
        ax_label.text(
            0.0,
            0.5,
            label + note,
            ha="left",
            va="center",
            fontsize=10,
            color=color,
            fontweight="bold",
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

    v_lo, v_hi = _plot_vlim(v_axis, vlim)
    if vlim is None:
        xrange_note = f"full cube v=[{v_lo:.0f}, {v_hi:.0f}] km/s"
    else:
        xrange_note = f"v=[{v_lo:.0f}, {v_hi:.0f}] km/s"
    fig.suptitle(
        f"Smoothed-cube spectra ({xrange_note}): dotted v = Scouse component centers"
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
    parser = argparse.ArgumentParser(
        description="Random agree/disagree spectrum panels vs Scouse K labels."
    )
    parser.add_argument("--k-pred", type=Path, required=True)
    parser.add_argument("--cube", type=Path, default=_REPO / "data" / "CMZ_3mm_HNCO_60.fits")
    parser.add_argument("--dat", type=Path, default=_REPO / "data" / "final_fits_updated.dat")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--n-each", type=int, default=4, help="Random panels per row.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--agree-max-delta",
        type=int,
        default=0,
        help="Agree row: |K_pred - K_scouse| <= this (default 0 = exact).",
    )
    parser.add_argument(
        "--min-discrep",
        type=int,
        default=4,
        help="Discrep row: only |dK| >= this (default 4).",
    )
    parser.add_argument(
        "--vel-range",
        type=float,
        nargs=2,
        default=None,
        metavar=("V_MIN", "V_MAX"),
        help="Optional plot zoom in km/s (default: full cube velocity axis).",
    )
    parser.add_argument(
        "--snr-vel-range",
        type=float,
        nargs=2,
        default=None,
        metavar=("V_MIN", "V_MAX"),
        help="SNR_scouse window in km/s (default: full cube, or match --vel-range if set).",
    )
    parser.add_argument("--n-col", type=int, default=4)
    parser.add_argument(
        "--cases-json",
        type=Path,
        default=None,
        help="Optional JSON listing sampled (l,b) for reproducibility.",
    )
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

    cases = _build_cases(
        k_pred,
        k_true,
        wcs=wcs,
        by_pos=by_pos,
        spec_cube=spec_cube,
        v_axis=v_axis,
        snr_vel_range=snr_vel_range,
    )
    print(f"Comparable labeled pixels: {len(cases)}", flush=True)

    rng = np.random.default_rng(args.seed)
    agree, discrep = _sample_cases(
        cases,
        rng=rng,
        n_each=int(args.n_each),
        agree_max_delta=int(args.agree_max_delta),
        min_discrep=int(args.min_discrep),
    )

    pred_lookup = None
    if args.pred_centers is not None:
        pred_lookup = load_pred_centers_lookup(args.pred_centers)
        print(f"Loaded ML centers for {len(pred_lookup)} pixels from {args.pred_centers}", flush=True)

    heatmap_probs = None
    if args.heatmap_run_dir is not None:
        selected = list(agree) + list(discrep)
        ### Deduplicate while preserving order.
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

    out = args.out
    if out is None:
        out = _MOPRA / "figures" / f"{args.k_pred.stem}_discrepancy_spectra.png"

    fig_path = plot_discrepancy_spectra(
        agree,
        discrep,
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
        "agree_max_delta": int(args.agree_max_delta),
        "min_discrep": int(args.min_discrep),
        "vel_range": list(vlim) if vlim is not None else None,
        "snr_vel_range": list(snr_vel_range) if snr_vel_range is not None else None,
        "agree": [
            {"l": c.l, "b": c.b, "yi": c.yi, "xi": c.xi, "k_true": c.k_true, "k_pred": c.k_pred}
            for c in agree
        ],
        "discrep": [
            {"l": c.l, "b": c.b, "yi": c.yi, "xi": c.xi, "k_true": c.k_true, "k_pred": c.k_pred}
            for c in discrep
        ],
    }
    cases_json = args.cases_json if args.cases_json is not None else out.with_suffix(".json")
    cases_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Agree panels: {len(agree)}   Discrep panels: {len(discrep)}", flush=True)
    for label, group in [("agree", agree), ("discrep", discrep)]:
        for c in group:
            print(
                f"  [{label}] l={c.l:.5f} b={c.b:.5f}  K {c.k_true}->{c.k_pred}  "
                f"SNR_scouse~{c.snr_scouse:.1f}  SNR_global~{c.snr_global:.1f}",
                flush=True,
            )
    print(f"Wrote {fig_path}", flush=True)
    print(f"Wrote {cases_json}", flush=True)


if __name__ == "__main__":
    main()
