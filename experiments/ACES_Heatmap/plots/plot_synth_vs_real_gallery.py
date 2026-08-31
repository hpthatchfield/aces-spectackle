#!/usr/bin/env python
"""
Side-by-side gallery: simple_snr labeled synth vs real region1 spectra.

Left columns: synthetic spectra with true (SNR-kept) centers.
Right columns: real mosaic cutout spectra (same K_pred buckets when possible).

  python experiments/ACES_Heatmap/plots/plot_synth_vs_real_gallery.py
"""
from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from astropy.io import fits
from spectral_cube import SpectralCube

_SCRIPT = Path(__file__).resolve()
_ACES = _SCRIPT.parents[1]
_REPO = _ACES.parents[1]
sys.path.insert(0, str(_REPO / "src"))

from spectackle.data.aces_generator import build_aces_synth_cfg, generate_aces_spectrum  ### noqa: E402
from spectackle.data.generator import _make_v_axis, channel_width_kms  ### noqa: E402
from spectackle.data.preprocess import prepare_spectrum_input, valid_mask  ### noqa: E402
from spectackle.training import build_center_target_map  ### noqa: E402

_DEFAULT_MOSAIC = _REPO / "data" / (
    "group.uid___A001_X1590_X30a9.lp_slongmore.cmz_mosaic.12m7mTP.HNCO_7m12mTP.cube.pbcor.fits"
)
COL_SPEC = "0.35"
COL_TRUE = "#1565C0"
COL_TGT = "#C62828"


def _sample_synth_by_k(
    cfg: dict,
    v_axis: np.ndarray,
    *,
    k_values: list[int],
    n_each: int,
    seed: int,
    max_tries: int = 20000,
    prefer_close_sep_kms: float | None = 20.0,
) -> dict[int, list[dict]]:
    """
    Sample n_each examples per K.

    For K>=2, prefer nearest-neighbor separation <= prefer_close_sep_kms so the
    gallery shows blended morphologies (fallback to any if not enough).
    """
    rng = np.random.default_rng(seed)
    buckets: dict[int, list[dict]] = {k: [] for k in k_values}
    half = max_tries // 2

    for t in range(max_tries):
        if all(len(buckets[k]) >= n_each for k in k_values):
            break
        ex = generate_aces_spectrum(cfg, np.random.default_rng(int(rng.integers(0, 2**31 - 1))), v_axis=v_axis)
        k = int(ex["k"])
        if k not in buckets or len(buckets[k]) >= n_each:
            continue
        if k >= 2 and prefer_close_sep_kms is not None and t < half:
            sep = _nearest_sep_kms(ex)
            if not (np.isfinite(sep) and sep <= float(prefer_close_sep_kms)):
                continue
        buckets[k].append(ex)
    return buckets


def _nearest_sep_kms(ex: dict) -> float:
    k = int(ex["k"])
    if k < 2:
        return float("nan")
    mus = np.sort(np.asarray(ex["component_v_kms"][:k], dtype=np.float64))
    return float(np.min(np.diff(mus)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Synth simple_snr vs region1 spectra gallery.")
    parser.add_argument("--k-pred", type=Path, default=_REPO / "data" / "hnco_region1_aces_hm_k_pred.fits")
    parser.add_argument("--centers", type=Path, default=None)
    parser.add_argument("--cube", type=Path, default=_DEFAULT_MOSAIC)
    parser.add_argument("--gen-preset", type=str, default="simple_snr")
    parser.add_argument("--Kmax", type=int, default=6)
    parser.add_argument("--v-half-kms", type=float, default=80.0)
    parser.add_argument("--label-sigma-kms", type=float, default=1.0)
    parser.add_argument("--n-each", type=int, default=4)
    parser.add_argument("--k-values", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    cfg = build_aces_synth_cfg(
        Kmax=args.Kmax,
        v_half_width_kms=float(args.v_half_kms),
        gen_preset=args.gen_preset,
    )
    v_synth = _make_v_axis(cfg).astype(np.float64)
    cw = float(channel_width_kms(cfg))
    gen = cfg["gen"]
    print(
        f"Preset={args.gen_preset}  n_ch={cfg['n_channels']}  dv={cw:.4f}  "
        f"cap={gen.get('glance_cap_mode')}  snr_tol={gen.get('glance_snr_tol')}  "
        f"min_sep_ch={gen.get('min_sep_channels')}  sep_factor={gen.get('min_component_separation')}",
        flush=True,
    )

    buckets = _sample_synth_by_k(
        cfg, v_synth, k_values=list(args.k_values), n_each=args.n_each, seed=args.seed
    )
    for k in args.k_values:
        seps = [_nearest_sep_kms(ex) for ex in buckets[k] if int(ex["k"]) >= 2]
        print(
            f"  synth K={k}: got {len(buckets[k])}/{args.n_each}"
            + (f"  min_sep~{np.nanmin(seps):.2f}-{np.nanmax(seps):.2f} km/s" if seps else ""),
            flush=True,
        )

    ### Real cutout.
    k_map = fits.getdata(args.k_pred).astype(np.float32)
    centers_path = args.centers or args.k_pred.with_name(f"{args.k_pred.stem}_centers.npz")
    z = np.load(centers_path, allow_pickle=True)
    y0, x0 = int(z["y0"]), int(z["x0"])
    i0, i1 = int(z["i0"]), int(z["i1"])
    v_real = np.asarray(z["v_axis"], dtype=np.float64)
    yi = np.asarray(z["yi"], dtype=np.int64)
    xi = np.asarray(z["xi"], dtype=np.int64)
    kp = np.asarray(z["k_pred"], dtype=np.float32)

    print(f"Loading real cutout from {args.cube}", flush=True)
    cube = SpectralCube.read(str(args.cube.resolve()), use_dask=True)
    ny, nx = k_map.shape
    sub = cube[i0:i1, y0 : y0 + ny, x0 : x0 + nx].filled(np.nan)
    if hasattr(sub, "compute"):
        sub = sub.compute()
    cut = np.asarray(sub, dtype=np.float32)

    rng = np.random.default_rng(args.seed + 7)
    real_coords: dict[int, list[tuple[int, int]]] = {}
    for k in args.k_values:
        pool = np.flatnonzero(np.round(kp) == k)
        if pool.size == 0:
            ### Fallback: quietest / random finite for missing bins.
            print(f"  real K_pred={k}: empty; sampling mixed finite", flush=True)
            pool = np.arange(kp.size)
        take = min(args.n_each, int(pool.size))
        chosen = rng.choice(pool, size=take, replace=False)
        real_coords[k] = [(int(yi[j]), int(xi[j])) for j in sorted(chosen.tolist())]

    n_row = len(args.k_values)
    n_col = args.n_each
    ### Two blocks side by side: synth | real
    fig, axes = plt.subplots(
        n_row, 2 * n_col,
        figsize=(2.8 * 2 * n_col, 2.35 * n_row),
        squeeze=False,
    )
    for r, k in enumerate(args.k_values):
        ### Synth block.
        for c in range(n_col):
            ax = axes[r, c]
            if c >= len(buckets[k]):
                ax.axis("off")
                continue
            ex = buckets[k][c]
            spec = np.asarray(ex["spec"], dtype=np.float64)
            vm = valid_mask(spec)
            xn, _ = prepare_spectrum_input(spec)
            ax.plot(v_synth[vm], xn[vm], color=COL_SPEC, lw=0.85)
            ### Heatmap training target (gaussian splat on centers).
            kk = int(ex["k"])
            v_t = torch.from_numpy(v_synth.astype(np.float32))
            centers = torch.from_numpy(ex["component_v_kms"][:kk].astype(np.float32))
            tgt = build_center_target_map(
                centers.unsqueeze(0),
                torch.ones(1, kk),
                v_t,
                label_sigma_kms=float(args.label_sigma_kms),
            ).numpy()[0]
            ax2 = ax.twinx()
            ax2.plot(v_synth[vm], tgt[vm], color=COL_TGT, lw=0.9, alpha=0.85)
            ax2.set_ylim(-0.05, 1.05)
            ax2.set_yticklabels([])
            for vv in ex["component_v_kms"][: int(ex["k"])]:
                ax.axvline(float(vv), color=COL_TRUE, ls=":", lw=0.9)
            kd = int(ex.get("k_drawn", ex["k"]))
            ax.set_title(f"synth K={int(ex['k'])} (drawn {kd})", fontsize=8)
            if c == 0:
                ax.set_ylabel(f"K~{k}\nsynth T", fontsize=8)
            if r == n_row - 1:
                ax.set_xlabel("v (km/s)", fontsize=7)

        ### Real block.
        for c in range(n_col):
            ax = axes[r, n_col + c]
            if c >= len(real_coords[k]):
                ax.axis("off")
                continue
            ly, lx = real_coords[k][c]
            spec = cut[:, ly, lx].astype(np.float64)
            xn, vm = prepare_spectrum_input(spec)
            m = vm > 0.5
            ax.plot(v_real[m], xn[m], color=COL_SPEC, lw=0.85)
            ax.set_title(f"real ({y0 + ly},{x0 + lx})  K_pred={k}", fontsize=8)
            if c == 0:
                ax.set_ylabel(f"K_pred={k}\nreal T", fontsize=8)
            if r == n_row - 1:
                ax.set_xlabel("v (km/s)", fontsize=7)

    sep_ch = float(gen.get("min_sep_channels") or 0.0)
    fig.suptitle(
        f"ACES {args.gen_preset}: labeled synth (blue=true centers, red=heatmap target)  |  "
        f"region1 real (by K_pred)\n"
        f"SNR prune only; draw min_sep~{sep_ch * cw:.2f} km/s floor + "
        f"{gen.get('min_component_separation')}x(sigma_i+sigma_j)",
        fontsize=10,
    )
    ### Light vertical divider between synth and real.
    fig.canvas.draw()
    for r in range(n_row):
        axes[r, n_col].spines["left"].set_linewidth(2.0)
        axes[r, n_col].spines["left"].set_color("0.5")

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = args.out or (
        _ACES / "figures" / "sanity" / f"synth_{args.gen_preset}_vs_region1_gallery.png"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
