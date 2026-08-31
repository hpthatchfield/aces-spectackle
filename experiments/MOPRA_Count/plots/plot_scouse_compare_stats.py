#!/usr/bin/env python
"""
Statistical trends for ML vs Scouse K comparison on the real CMZ cube.

Example:
  python experiments/MOPRA_Count/plots/plot_scouse_compare_stats.py \\
    --k-pred data/mopra_cmz_k_pred_scouse_ft_v1.fits \\
    --cube data/CMZ_3mm_HNCO_60.fits
"""
from __future__ import annotations

import argparse
import json
import os
import sys
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
from astropy.io import fits
from astropy.wcs import WCS
from scipy.ndimage import distance_transform_edt
from spectral_cube import SpectralCube

from plot_k_residual_map import k_true_map_from_dat  ### noqa: E402
from plot_style import COL_FAIL, COL_MAE, COL_SPEC, COL_SUCCESS  ### noqa: E402
from spectackle.data.mopra_preprocess import snr_peak_scouse_mopra, valid_mask_mopra  ### noqa: E402
from spectackle.plotting import mae_by_true_k  ### noqa: E402


def collect_compare_pixels(
    k_pred: np.ndarray,
    *,
    dat_path: Path,
    cube_path: Path,
    vel_range: tuple[float, float] | None = None,
    use_row_count: bool = False,
    blank_value: float = -1.0,
) -> dict:
    """Per-pixel Scouse comparison table at labeled (l,b) positions."""
    cube_header = fits.getheader(cube_path)
    wcs = WCS(cube_header).celestial
    ny, nx = k_pred.shape
    if (ny, nx) != (cube_header["NAXIS2"], cube_header["NAXIS1"]):
        raise ValueError(
            f"k-pred shape {k_pred.shape} != cube spatial "
            f"({cube_header['NAXIS2']}, {cube_header['NAXIS1']})"
        )

    k_true, labeled = k_true_map_from_dat(
        dat_path,
        shape=k_pred.shape,
        wcs=wcs,
        use_row_count=use_row_count,
    )
    compare = labeled & np.isfinite(k_pred)
    n = int(compare.sum())
    if n == 0:
        raise ValueError("No overlapping labeled + inferred pixels for comparison.")

    ys, xs = np.where(compare)
    kt = k_true[compare].astype(int)
    kp = k_pred[compare].astype(int)
    delta = kp - kt

    cube = SpectralCube.read(str(cube_path.resolve()), use_dask=False)
    arr = np.asarray(cube.filled(np.nan), dtype=np.float32)
    vel = cube.spectral_axis.to("km/s").value.astype(np.float64)
    if arr.ndim == 4:
        arr = arr[0]

    vmask = np.ones(vel.size, dtype=bool)
    if vel_range is not None:
        vlo, vhi = float(vel_range[0]), float(vel_range[1])
        vmask = (vel >= vlo) & (vel <= vhi)

    peak_t = np.full(n, np.nan, dtype=np.float32)
    scouse_snr = np.full(n, np.nan, dtype=np.float32)
    for i, (yi, xi) in enumerate(zip(ys, xs)):
        spec = arr[:, yi, xi]
        valid = valid_mask_mopra(spec, blank_value=blank_value) & vmask
        if valid.any():
            peak_t[i] = float(np.nanmax(spec[valid]))
        scouse_snr[i] = float(
            snr_peak_scouse_mopra(
                spec,
                blank_value=blank_value,
                vel_kms=vel,
                vel_range=vel_range,
            )
        )

    dist_edge = distance_transform_edt(labeled)[compare].astype(np.float32)

    return {
        "n_compare": n,
        "yi": ys.astype(int),
        "xi": xs.astype(int),
        "k_true": kt,
        "k_pred": kp,
        "delta": delta.astype(int),
        "peak_t_k": peak_t,
        "scouse_snr": scouse_snr,
        "dist_to_edge_pix": dist_edge,
        "vel_range": list(vel_range) if vel_range is not None else None,
    }


def _summary_by_k_true(table: dict, *, Kmax: int = 10) -> dict:
    kt = table["k_true"]
    delta = table["delta"]
    out: dict[str, dict] = {}
    for k in range(Kmax + 1):
        m = kt == k
        if not np.any(m):
            continue
        d = delta[m]
        out[str(k)] = {
            "n": int(m.sum()),
            "mae": float(np.mean(np.abs(d))),
            "mean_delta": float(np.mean(d)),
            "median_delta": float(np.median(d)),
            "frac_exact": float(np.mean(d == 0)),
            "frac_over": float(np.mean(d > 0)),
            "frac_under": float(np.mean(d < 0)),
        }
    return out


def _summary_by_value_bins(
    values: np.ndarray,
    delta: np.ndarray,
    *,
    edges: np.ndarray,
    label: str,
) -> dict[str, dict]:
    """Bin dK outcomes by a continuous amplitude proxy (SNR or peak T)."""
    out: dict[str, dict] = {}
    ok = np.isfinite(values) & np.isfinite(delta)
    for i in range(len(edges) - 1):
        lo, hi = float(edges[i]), float(edges[i + 1])
        if np.isfinite(hi):
            m = ok & (values >= lo) & (values < hi)
            key = f"{lo:g}-{hi:g}"
        else:
            m = ok & (values >= lo)
            key = f">={lo:g}"
        if not np.any(m):
            continue
        d = delta[m]
        out[key] = {
            "n": int(m.sum()),
            "mae": float(np.mean(np.abs(d))),
            "mean_delta": float(np.mean(d)),
            "median_delta": float(np.median(d)),
            "frac_exact": float(np.mean(d == 0)),
            "frac_over": float(np.mean(d > 0)),
            "frac_under": float(np.mean(d < 0)),
            "bin_var": label,
        }
    return out


def plot_amp_failure_bins(
    table: dict,
    *,
    out: Path,
    title: str | None = None,
    dpi: int = 150,
) -> tuple[Path, dict, dict]:
    """
    Failure mix vs amplitude proxies (Scouse SNR and peak T).

    Independent of blend involvement: isolates whether over-count concentrates at low SNR.
    """
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    delta = table["delta"]
    snr = table["scouse_snr"]
    peak = table["peak_t_k"]

    snr_edges = np.array([0.0, 3.0, 5.0, 8.0, 12.0, 20.0, 40.0, np.inf])
    peak_edges = np.array([0.0, 0.1, 0.2, 0.4, 0.8, 1.5, 3.0, np.inf])
    by_snr = _summary_by_value_bins(snr, delta, edges=snr_edges, label="scouse_snr")
    by_peak = _summary_by_value_bins(peak, delta, edges=peak_edges, label="peak_t")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    if title:
        fig.suptitle(title, fontsize=11, y=0.98)

    for ax, bins, xlab in (
        (axes[0], by_snr, r"Scouse SNR bin"),
        (axes[1], by_peak, r"peak $T$ bin (K)"),
    ):
        keys = list(bins.keys())
        xs = np.arange(len(keys))
        frac_over = [bins[k]["frac_over"] for k in keys]
        frac_under = [bins[k]["frac_under"] for k in keys]
        frac_exact = [bins[k]["frac_exact"] for k in keys]
        ns = [bins[k]["n"] for k in keys]
        ax.bar(xs, frac_exact, color="#bdbdbd", edgecolor="white", label="exact")
        ax.bar(xs, frac_over, bottom=frac_exact, color=COL_FAIL, edgecolor="white", label="over")
        bottom = np.array(frac_exact) + np.array(frac_over)
        ax.bar(xs, frac_under, bottom=bottom, color=COL_SUCCESS, edgecolor="white", label="under")
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{k}\nn={n}" for k, n in zip(keys, ns)], fontsize=7)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel(xlab)
        ax.set_ylabel("fraction")
        ax.legend(loc="upper right", fontsize=8)

    axes[0].set_title("Outcome vs Scouse SNR (amp proxy)")
    axes[1].set_title("Outcome vs peak brightness")
    fig.tight_layout(rect=(0, 0, 1, 0.94) if title else None)
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return out, by_snr, by_peak


def _binned_mean(x: np.ndarray, y: np.ndarray, *, edges: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return bin centres, mean y, and counts per bin."""
    centers = 0.5 * (edges[:-1] + edges[1:])
    means = np.full(centers.size, np.nan, dtype=np.float64)
    counts = np.zeros(centers.size, dtype=int)
    for i in range(centers.size):
        m = (x >= edges[i]) & (x < edges[i + 1])
        counts[i] = int(m.sum())
        if counts[i] > 0:
            means[i] = float(np.mean(y[m]))
    return centers, means, counts


def plot_scouse_compare_stats(
    table: dict,
    *,
    out: Path,
    title: str | None = None,
    Kmax: int = 4,
    dpi: int = 150,
) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    kt = table["k_true"]
    delta = table["delta"]
    peak_t = table["peak_t_k"]
    dist_edge = table["dist_to_edge_pix"]

    k_levels = [k for k in range(Kmax + 1) if np.any(kt == k)]
    mae_k = mae_by_true_k(kt, kt + delta, Kmax)

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.5))
    if title:
        fig.suptitle(title, fontsize=12, y=0.98)

    ### Panel 1: dK distribution by K_true
    ax = axes[0, 0]
    data = [delta[kt == k] for k in k_levels]
    bp = ax.boxplot(data, positions=k_levels, widths=0.55, patch_artist=True, showfliers=False)
    for box in bp["boxes"]:
        box.set_facecolor(COL_SPEC)
        box.set_alpha(0.55)
    ax.axhline(0.0, color="k", lw=0.8, alpha=0.45)
    ax.set_xlabel(r"$K_{\rm true}$")
    ax.set_ylabel(r"$\Delta K = K_{\rm pred} - K_{\rm true}$")
    ax.set_title(r"$\Delta K$ by $K_{\rm true}$")
    ax.set_xticks(k_levels)

    ### Panel 2: MAE by K_true
    ax = axes[0, 1]
    mae_vals = [mae_k.get(k, float("nan")) for k in k_levels]
    bars = ax.bar(k_levels, mae_vals, color=COL_MAE, edgecolor="white")
    ax.set_xlabel(r"$K_{\rm true}$")
    ax.set_ylabel("MAE")
    ax.set_title("MAE by $K_{\\rm true}$")
    ax.set_xticks(k_levels)
    for bar, k in zip(bars, k_levels):
        n_k = int((kt == k).sum())
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"n={n_k}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ### Panel 3: mean dK by K_true (signed bias)
    ax = axes[0, 2]
    mean_delta = [float(np.mean(delta[kt == k])) for k in k_levels]
    colors = [COL_FAIL if m > 0 else COL_SUCCESS if m < 0 else "#888888" for m in mean_delta]
    ax.bar(k_levels, mean_delta, color=colors, edgecolor="white")
    ax.axhline(0.0, color="k", lw=0.8, alpha=0.45)
    ax.set_xlabel(r"$K_{\rm true}$")
    ax.set_ylabel(r"mean $\Delta K$")
    ax.set_title("Signed bias by $K_{\\rm true}$")
    ax.set_xticks(k_levels)

    ### Panel 4: dK vs peak T
    ax = axes[1, 0]
    ok = np.isfinite(peak_t)
    hb = ax.hexbin(
        peak_t[ok],
        delta[ok],
        gridsize=35,
        cmap="coolwarm",
        mincnt=1,
        linewidths=0.2,
    )
    ax.axhline(0.0, color="k", lw=0.8, alpha=0.45)
    ax.set_xlabel("peak T in window (K)")
    ax.set_ylabel(r"$\Delta K$")
    ax.set_title(r"$\Delta K$ vs peak brightness")
    cb = fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("count")

    ### Panel 5: mean dK vs distance to labeled-region edge
    ax = axes[1, 1]
    edge_edges = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 10.0, 20.0, np.inf])
    centers, means, counts = _binned_mean(dist_edge, delta.astype(np.float64), edges=edge_edges)
    ax.plot(centers, means, "o-", color=COL_SPEC, lw=1.5)
    ax.axhline(0.0, color="k", lw=0.8, alpha=0.45)
    ax.set_xlabel("distance to labeled-region edge (pix)")
    ax.set_ylabel(r"mean $\Delta K$")
    ax.set_title("Edge trend (mean $\\Delta K$ vs radius)")
    for x, y, n_bin in zip(centers, means, counts):
        if n_bin > 0 and np.isfinite(y):
            ax.annotate(str(n_bin), (x, y), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=7)

    ### Panel 6: exact / over / under fractions by K_true
    ax = axes[1, 2]
    frac_exact = [float(np.mean(delta[kt == k] == 0)) for k in k_levels]
    frac_over = [float(np.mean(delta[kt == k] > 0)) for k in k_levels]
    frac_under = [float(np.mean(delta[kt == k] < 0)) for k in k_levels]
    ax.bar(k_levels, frac_exact, color="#bdbdbd", edgecolor="white", label="exact")
    ax.bar(k_levels, frac_over, bottom=frac_exact, color=COL_FAIL, edgecolor="white", label="over")
    bottom = np.array(frac_exact) + np.array(frac_over)
    ax.bar(k_levels, frac_under, bottom=bottom, color=COL_SUCCESS, edgecolor="white", label="under")
    ax.set_xlim(min(k_levels) - 0.6, max(k_levels) + 0.6)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel(r"$K_{\rm true}$")
    ax.set_ylabel("fraction")
    ax.set_title("Outcome mix by $K_{\\rm true}$")
    ax.set_xticks(k_levels)
    ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout(rect=(0, 0, 1, 0.96) if title else None)
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Scouse comparison trend plots on real cube.")
    parser.add_argument("--k-pred", type=Path, required=True)
    parser.add_argument("--dat", type=Path, default=_REPO / "data" / "final_fits_updated.dat")
    parser.add_argument("--cube", type=Path, default=_REPO / "data" / "CMZ_3mm_HNCO_60.fits")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--stats-json", type=Path, default=None)
    parser.add_argument("--title", type=str, default=None)
    parser.add_argument("--Kmax", type=int, default=4)
    parser.add_argument("--use-row-count", action="store_true")
    parser.add_argument(
        "--vel-range",
        type=float,
        nargs=2,
        default=(40.0, 140.0),
        metavar=("V_MIN", "V_MAX"),
        help="Velocity window for peak T and Scouse SNR (default: 40 140 km/s).",
    )
    args = parser.parse_args()

    k_pred = fits.getdata(args.k_pred).astype(np.float32)
    vel_range = (float(args.vel_range[0]), float(args.vel_range[1]))
    table = collect_compare_pixels(
        k_pred,
        dat_path=args.dat,
        cube_path=args.cube,
        vel_range=vel_range,
        use_row_count=args.use_row_count,
    )

    out = args.out
    if out is None:
        out = _SCRIPT.parent.parent / "figures" / f"{args.k_pred.stem}_scouse_stats.png"

    title = args.title
    if title is None:
        title = f"Scouse compare trends: {args.k_pred.stem}"

    fig_path = plot_scouse_compare_stats(
        table,
        out=out,
        title=title,
        Kmax=int(args.Kmax),
    )

    amp_png = out.with_name(out.stem.replace("_scouse_stats", "_amp_failures") + ".png")
    if amp_png == out:
        amp_png = out.with_name(out.stem + "_amp_failures.png")
    amp_path, by_snr, by_peak = plot_amp_failure_bins(
        table,
        out=amp_png,
        title=f"Amplitude-binned failures: {args.k_pred.stem}",
    )

    summary = {
        "k_pred": str(args.k_pred.resolve()),
        "cube": str(args.cube.resolve()),
        "dat": str(args.dat.resolve()),
        "n_compare": int(table["n_compare"]),
        "vel_range": table["vel_range"],
        "global": {
            "mae": float(np.mean(np.abs(table["delta"]))),
            "mean_delta": float(np.mean(table["delta"])),
            "median_delta": float(np.median(table["delta"])),
            "frac_exact": float(np.mean(table["delta"] == 0)),
            "frac_over": float(np.mean(table["delta"] > 0)),
            "frac_under": float(np.mean(table["delta"] < 0)),
        },
        "by_k_true": _summary_by_k_true(table, Kmax=int(args.Kmax)),
        "by_scouse_snr": by_snr,
        "by_peak_t": by_peak,
        "edge_bins": {},
    }

    edge_edges = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 10.0, 20.0, np.inf])
    centers, means, counts = _binned_mean(
        table["dist_to_edge_pix"],
        table["delta"].astype(np.float64),
        edges=edge_edges,
    )
    for i, (c, m, n_bin) in enumerate(zip(centers, means, counts)):
        if n_bin <= 0:
            continue
        lo = float(edge_edges[i])
        hi = float(edge_edges[i + 1])
        key = f"{lo:g}-{hi:g}" if np.isfinite(hi) else f">={lo:g}"
        summary["edge_bins"][key] = {
            "n": int(n_bin),
            "mean_delta": float(m) if np.isfinite(m) else None,
        }

    stats_json = args.stats_json
    if stats_json is None:
        stats_json = out.with_suffix(".json")
    stats_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Compared {summary['n_compare']} pixels")
    print(f"  MAE={summary['global']['mae']:.3f}  median dK={summary['global']['median_delta']:.1f}")
    print(f"Wrote {fig_path}")
    print(f"Wrote {amp_path}")
    print(f"Wrote {stats_json}")


if __name__ == "__main__":
    main()
