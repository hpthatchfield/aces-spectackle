#!/usr/bin/env python
"""
Synth vs region1 morphology galleries for the ACES heatmap generator.

The generator does not tag blend vs isolated; we split synth after the fact
by nearest-neighbor |dv|/(sig_i+sig_j). Real cutout spectra have no K labels.

  python experiments/ACES_Heatmap/plots/plot_synth_vs_real_gallery.py
"""
from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from astropy.io import fits

_SCRIPT = Path(__file__).resolve()
_ACES = _SCRIPT.parents[1]
_REPO = _ACES.parents[1]
sys.path.insert(0, str(_REPO / "src"))

from spectackle.data.aces_generator import (  ### noqa: E402
    ACES_N_CHANNELS,
    ACES_VRANGE,
    build_aces_synth_cfg,
    generate_aces_spectrum,
)
from spectackle.data.generator import _make_v_axis, channel_width_kms  ### noqa: E402
from spectackle.data.preprocess import prepare_spectrum_input, valid_mask  ### noqa: E402
from spectackle.data.resolvable_peaks import (  ### noqa: E402
    blend_stats_from_synth_example,
    count_resolvable_peaks,
)
from spectackle.training import build_center_target_map  ### noqa: E402

_DEFAULT_CUBE = _REPO / "data" / "hnco_region1_native_aces.fits"
COL_SPEC = "0.35"
COL_TRUE = "#1565C0"
COL_TGT = "#C62828"

### Post-hoc cuts for the labeled synth figure only.
### Partial blend: overlapping but not fully stacked on one center.
BLEND_SEP_SIGMA_LO = 0.45
BLEND_SEP_SIGMA_HI = 1.3
ISOLATED_SEP_SIGMA = 2.0


def _sep_sigma(ex: dict) -> float:
    st = blend_stats_from_synth_example(ex)
    return float(st["min_sep_sigma"])


def _sample_labeled_rows(
    cfg: dict,
    v_axis: np.ndarray,
    *,
    n_each: int,
    seed: int,
    max_tries: int = 40000,
) -> list[tuple[str, list[dict]]]:
    """
    Draw from the generator as-is; bucket by K / islands / sep/sig.

    Islands preset: K=1, K=2 one-family, K=2 two-islands, K=3+ one-family, K>=5.
    Other presets: K=1, K=2 isolated, K=2 blended, K=3 blended.
    """
    rng = np.random.default_rng(seed)
    use_islands = str(cfg.get("gen", {}).get("k_mode", "")) == "islands"
    if use_islands:
        keys = ("k1", "k2_fam", "k2_two", "k3_fam", "k_high")
    else:
        keys = ("k1", "k2_isol", "k2_blend", "k3_blend")
    buckets: dict[str, list[dict]] = {k: [] for k in keys}
    for _ in range(max_tries):
        if all(len(buckets[k]) >= n_each for k in keys):
            break
        ex = generate_aces_spectrum(
            cfg, np.random.default_rng(int(rng.integers(0, 2**31 - 1))), v_axis=v_axis
        )
        k = int(ex["k"])
        if use_islands:
            n_isl = int(ex.get("n_islands", 0))
            if k == 1 and len(buckets["k1"]) < n_each:
                buckets["k1"].append(ex)
            elif k == 2 and n_isl == 1 and len(buckets["k2_fam"]) < n_each:
                buckets["k2_fam"].append(ex)
            elif k == 2 and n_isl >= 2 and len(buckets["k2_two"]) < n_each:
                buckets["k2_two"].append(ex)
            elif k >= 3 and n_isl == 1 and len(buckets["k3_fam"]) < n_each:
                buckets["k3_fam"].append(ex)
            elif k >= 5 and len(buckets["k_high"]) < n_each:
                buckets["k_high"].append(ex)
            continue
        if k == 1 and len(buckets["k1"]) < n_each:
            buckets["k1"].append(ex)
            continue
        if k < 2:
            continue
        ss = _sep_sigma(ex)
        if not np.isfinite(ss):
            continue
        if k == 2 and ss >= ISOLATED_SEP_SIGMA and len(buckets["k2_isol"]) < n_each:
            buckets["k2_isol"].append(ex)
        elif k == 2 and BLEND_SEP_SIGMA_LO <= ss <= BLEND_SEP_SIGMA_HI and len(buckets["k2_blend"]) < n_each:
            buckets["k2_blend"].append(ex)
        elif k == 3 and BLEND_SEP_SIGMA_LO <= ss <= BLEND_SEP_SIGMA_HI and len(buckets["k3_blend"]) < n_each:
            buckets["k3_blend"].append(ex)

    if use_islands:
        return [
            ("K=1", buckets["k1"]),
            ("K=2  one family", buckets["k2_fam"]),
            ("K=2  two islands", buckets["k2_two"]),
            ("K>=3  one family", buckets["k3_fam"]),
            ("K>=5", buckets["k_high"]),
        ]
    return [
        ("K=1", buckets["k1"]),
        (f"K=2 isolated  (sep/sig>={ISOLATED_SEP_SIGMA:g})", buckets["k2_isol"]),
        (f"K=2 blended  ({BLEND_SEP_SIGMA_LO:g}<=sep/sig<={BLEND_SEP_SIGMA_HI:g})", buckets["k2_blend"]),
        (f"K=3 blended  ({BLEND_SEP_SIGMA_LO:g}<=sep/sig<={BLEND_SEP_SIGMA_HI:g})", buckets["k3_blend"]),
    ]


def _sample_unlabeled_synth(
    cfg: dict,
    v_axis: np.ndarray,
    *,
    n: int,
    seed: int,
    k_min: int = 1,
) -> list[dict]:
    rng = np.random.default_rng(seed)
    out: list[dict] = []
    for _ in range(n * 200):
        if len(out) >= n:
            break
        ex = generate_aces_spectrum(
            cfg, np.random.default_rng(int(rng.integers(0, 2**31 - 1))), v_axis=v_axis
        )
        if int(ex["k"]) >= k_min:
            out.append(ex)
    return out


def _fits_velocity_axis(header) -> np.ndarray:
    n = int(header["NAXIS3"])
    cdelt = float(header["CDELT3"])
    crval = float(header["CRVAL3"])
    crpix = float(header["CRPIX3"])
    ch = np.arange(n, dtype=np.float64)
    return (ch + 1.0 - crpix) * cdelt + crval


def _window_indices(v_cube: np.ndarray, vlo: float, vhi: float) -> tuple[int, int]:
    i0 = int(np.argmin(np.abs(v_cube - float(vlo))))
    i1 = int(np.argmin(np.abs(v_cube - float(vhi))))
    if i1 < i0:
        i0, i1 = i1, i0
    return i0, i1 + 1


def _sample_real_spectra(
    cube: np.ndarray,
    v_win: np.ndarray,
    *,
    n: int,
    seed: int,
) -> list[tuple[int, int, np.ndarray]]:
    """
    Bright region1 pixels, mixed 1-bump / 2+ bump by a peak finder (sampling only).
    Returns (y, x, spec_window) with no K.
    """
    rng = np.random.default_rng(seed)
    C, ny, nx = cube.shape
    peak = np.nanmax(np.where(np.isfinite(cube), cube, np.nan), axis=0)
    finite = np.isfinite(peak)
    if not np.any(finite):
        raise RuntimeError("No finite pixels in real cutout.")
    thresh = float(np.nanpercentile(peak[finite], 70.0))
    ys, xs = np.where(finite & (peak >= thresh))
    order = rng.permutation(ys.size)

    one_bump: list[tuple[int, int, np.ndarray]] = []
    multi: list[tuple[int, int, np.ndarray]] = []
    want_each = max(1, n // 2)
    for idx in order:
        if len(one_bump) >= want_each and len(multi) >= n - want_each:
            break
        y, x = int(ys[idx]), int(xs[idx])
        spec = np.asarray(cube[:, y, x], dtype=np.float64)
        n_pk, _ = count_resolvable_peaks(
            spec,
            v_win,
            blank_value=None,
            vel_range=None,
            prominence_mode="adaptive",
            prominence_sigma=3.0,
            peak_frac=0.15,
            min_sep_kms=4.0,
        )
        item = (y, x, spec)
        if n_pk <= 1 and len(one_bump) < want_each:
            one_bump.append(item)
        elif n_pk >= 2 and len(multi) < n - want_each:
            multi.append(item)
    out = one_bump + multi
    if len(out) < n:
        ### Fill from leftover bright pixels.
        have = {(y, x) for y, x, _ in out}
        for idx in order:
            if len(out) >= n:
                break
            y, x = int(ys[idx]), int(xs[idx])
            if (y, x) in have:
                continue
            out.append((y, x, np.asarray(cube[:, y, x], dtype=np.float64)))
    return out[:n]


def _plot_labeled_synth(rows: list[tuple[str, list[dict]]], v_synth: np.ndarray, label_sigma: float, out: Path) -> None:
    n_row = len(rows)
    n_col = max(len(exs) for _, exs in rows)
    fig, axes = plt.subplots(n_row, n_col, figsize=(2.9 * n_col, 2.4 * n_row), squeeze=False)
    v_t = torch.from_numpy(v_synth.astype(np.float32))
    for r, (row_name, exs) in enumerate(rows):
        for c in range(n_col):
            ax = axes[r, c]
            if c >= len(exs):
                ax.axis("off")
                continue
            ex = exs[c]
            spec = np.asarray(ex["spec"], dtype=np.float64)
            vm = valid_mask(spec)
            xn, _ = prepare_spectrum_input(spec)
            ax.plot(v_synth[vm], xn[vm], color=COL_SPEC, lw=0.85)
            kk = int(ex["k"])
            centers = torch.from_numpy(ex["component_v_kms"][:kk].astype(np.float32))
            tgt = build_center_target_map(
                centers.unsqueeze(0),
                torch.ones(1, kk),
                v_t,
                label_sigma_kms=float(label_sigma),
            ).numpy()[0]
            ax2 = ax.twinx()
            ax2.plot(v_synth[vm], tgt[vm], color=COL_TGT, lw=0.9, alpha=0.85)
            ax2.set_ylim(-0.05, 1.05)
            ax2.set_yticklabels([])
            for vv in ex["component_v_kms"][:kk]:
                ax.axvline(float(vv), color=COL_TRUE, ls=":", lw=0.9)
            ss = _sep_sigma(ex)
            ss_txt = f"  sep/sig={ss:.2f}" if np.isfinite(ss) else ""
            n_isl = int(ex.get("n_islands", -1))
            isl_txt = f"  isl={n_isl}" if n_isl >= 0 else ""
            ax.set_title(f"K={kk}{ss_txt}{isl_txt}", fontsize=8)
            if c == 0:
                ax.set_ylabel(row_name, fontsize=8)
            if r == n_row - 1:
                ax.set_xlabel("v (km/s)", fontsize=7)
    fig.suptitle(
        "Labeled synth (blue=true centers, red=heatmap target). "
        "Blend vs isolated is measured after drawing, not a generator flag.",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"Wrote {out}")


def _plot_unlabeled_pair(
    synth_ex: list[dict],
    real_items: list[tuple[int, int, np.ndarray]],
    v_synth: np.ndarray,
    v_real: np.ndarray,
    out: Path,
) -> None:
    n = min(len(synth_ex), len(real_items))
    fig, axes = plt.subplots(n, 2, figsize=(9.5, 1.85 * n), squeeze=False)
    for i in range(n):
        ax_s = axes[i, 0]
        ex = synth_ex[i]
        spec = np.asarray(ex["spec"], dtype=np.float64)
        xn, vm = prepare_spectrum_input(spec)
        m = vm > 0.5
        ax_s.plot(v_synth[m], xn[m], color=COL_SPEC, lw=0.85)
        ax_s.set_xlim(float(v_synth[0]), float(v_synth[-1]))
        ax_s.set_ylabel("synth", fontsize=8)
        if i == 0:
            ax_s.set_title("synth (no labels)", fontsize=9)
        if i == n - 1:
            ax_s.set_xlabel("v (km/s)", fontsize=7)

        ax_r = axes[i, 1]
        y, x, spec_r = real_items[i]
        xn_r, vm_r = prepare_spectrum_input(spec_r)
        mr = vm_r > 0.5
        ax_r.plot(v_real[mr], xn_r[mr], color=COL_SPEC, lw=0.85)
        ax_r.set_xlim(float(v_synth[0]), float(v_synth[-1]))
        ax_r.set_ylabel("real", fontsize=8)
        ax_r.set_title(f"real ({y},{x})", fontsize=8)
        if i == 0:
            ax_r.set_title(f"real ({y},{x})  (no K)", fontsize=9)
        if i == n - 1:
            ax_r.set_xlabel("v (km/s)", fontsize=7)
    fig.suptitle(
        "Unlabeled morphology: same generator as training, next to region1 native cutout spectra.",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"Wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Labeled synth + unlabeled synth vs region1 galleries.")
    parser.add_argument("--cube", type=Path, default=_DEFAULT_CUBE)
    parser.add_argument("--gen-preset", type=str, default="simple_snr")
    parser.add_argument("--Kmax", type=int, default=6)
    parser.add_argument("--v-half-kms", type=float, default=80.0)
    parser.add_argument("--label-sigma-kms", type=float, default=1.0)
    parser.add_argument("--n-each", type=int, default=4)
    parser.add_argument("--n-unlabeled", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    cfg = build_aces_synth_cfg(
        Kmax=args.Kmax,
        v_half_width_kms=float(args.v_half_kms),
        gen_preset=args.gen_preset,
    )
    v_synth = _make_v_axis(cfg).astype(np.float64)
    cw = float(channel_width_kms(cfg))
    gen = cfg["gen"]
    out_dir = args.out_dir or (_ACES / "figures" / "sanity")
    print(
        f"Preset={args.gen_preset}  n_ch={cfg['n_channels']}  dv={cw:.4f}  "
        f"blend_cluster_prob={gen.get('blend_cluster_prob')}  "
        f"cluster_width={gen.get('cluster_width_range')}  "
        f"cluster_width_sample={gen.get('cluster_width_sample')}  "
        f"min_sep_ch={gen.get('min_sep_channels')}  "
        f"sep_factor={gen.get('min_component_separation')}",
        flush=True,
    )

    rows = _sample_labeled_rows(cfg, v_synth, n_each=args.n_each, seed=args.seed)
    for name, exs in rows:
        seps = [_sep_sigma(ex) for ex in exs]
        seps = [s for s in seps if np.isfinite(s)]
        extra = f"  sep/sig {np.min(seps):.2f}-{np.max(seps):.2f}" if seps else ""
        print(f"  {name}: {len(exs)}/{args.n_each}{extra}", flush=True)

    _plot_labeled_synth(
        rows,
        v_synth,
        args.label_sigma_kms,
        out_dir / f"labeled_synth_{args.gen_preset}.png",
    )

    unlabeled = _sample_unlabeled_synth(
        cfg, v_synth, n=args.n_unlabeled, seed=args.seed + 11
    )
    print(f"Loading real cutout from {args.cube}", flush=True)
    cube_hdu = fits.open(args.cube)
    hdr = cube_hdu[0].header
    arr = np.asarray(cube_hdu[0].data, dtype=np.float64)
    cube_hdu.close()
    if arr.ndim == 4:
        arr = arr[0]
    v_full = _fits_velocity_axis(hdr)
    if v_full.size != arr.shape[0]:
        ### Fall back to the ACES linspace axis if header naxis is odd.
        v_full = np.linspace(ACES_VRANGE[0], ACES_VRANGE[1], ACES_N_CHANNELS)
    i0, i1 = _window_indices(v_full, float(v_synth[0]), float(v_synth[-1]))
    cube_win = arr[i0:i1]
    v_real = v_full[i0:i1]
    print(f"  real window ch [{i0}:{i1}]  v={v_real[0]:.2f}..{v_real[-1]:.2f} km/s", flush=True)
    real_items = _sample_real_spectra(cube_win, v_real, n=args.n_unlabeled, seed=args.seed + 21)
    _plot_unlabeled_pair(
        unlabeled,
        real_items,
        v_synth,
        v_real,
        out_dir / f"synth_{args.gen_preset}_vs_region1_unlabeled.png",
    )


if __name__ == "__main__":
    main()
