#!/usr/bin/env python
"""
Spectrum gallery from an ACES heatmap->K cube map (no labels).

  python experiments/ACES_Heatmap/plots/plot_region_spectra.py \\
    --k-pred data/hnco_region1_aces_hm_k_pred.fits \\
    --centers data/hnco_region1_aces_hm_k_pred_centers.npz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy.io import fits
from spectral_cube import SpectralCube

_SCRIPT = Path(__file__).resolve()
_ACES = _SCRIPT.parents[1]
_REPO = _SCRIPT.parents[3]
sys.path.insert(0, str(_SCRIPT.parent))
sys.path.insert(0, str(_REPO / "src"))

from plot_region_style import galactic_lb, lb_title  ### noqa: E402
from spectackle.wcs_plot import wcs_celestial  ### noqa: E402

_DEFAULT_MOSAIC = _REPO / "data" / (
    "group.uid___A001_X1590_X30a9.lp_slongmore.cmz_mosaic.12m7mTP.HNCO_7m12mTP.cube.pbcor.fits"
)


def main() -> None:
    parser = argparse.ArgumentParser(description="ACES region K_pred spectrum gallery.")
    parser.add_argument("--k-pred", type=Path, required=True)
    parser.add_argument("--centers", type=Path, default=None)
    parser.add_argument("--cube", type=Path, default=_DEFAULT_MOSAIC)
    parser.add_argument("--n-each", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    k_pred = fits.getdata(args.k_pred).astype(np.float32)
    wcs = wcs_celestial(fits.getheader(args.k_pred))
    centers_path = args.centers or args.k_pred.with_name(f"{args.k_pred.stem}_centers.npz")
    z = np.load(centers_path, allow_pickle=True)
    y0 = int(z["y0"]) if "y0" in z.files else 0
    x0 = int(z["x0"]) if "x0" in z.files else 0
    i0 = int(z["i0"]) if "i0" in z.files else 0
    i1 = int(z["i1"]) if "i1" in z.files else None
    v_axis = np.asarray(z["v_axis"], dtype=np.float64) if "v_axis" in z.files else None
    yi = np.asarray(z["yi"], dtype=np.int64)
    xi = np.asarray(z["xi"], dtype=np.int64)
    kp = np.asarray(z["k_pred"], dtype=np.float32)
    cv = np.asarray(z["center_v_kms"], dtype=np.float32)

    print(f"Loading cube (lazy): {args.cube}", flush=True)
    ### Lazy read, then pull the spatial cutout once (109x109x770 is tiny vs full mosaic).
    cube = SpectralCube.read(str(args.cube.resolve()), use_dask=True)
    if i1 is None:
        i1 = cube.shape[0]
    if v_axis is None:
        v_axis = cube.spectral_axis.to(u.km / u.s).value.astype(np.float64)[i0:i1]

    ny_cut = int(k_pred.shape[0])
    nx_cut = int(k_pred.shape[1])
    ny_cube, nx_cube = int(cube.shape[-2]), int(cube.shape[-1])
    if (y0 + ny_cut > ny_cube or x0 + nx_cut > nx_cube) and (ny_cube, nx_cube) == (ny_cut, nx_cut):
        print(f"Cube is already the cutout ({ny_cut}x{nx_cut}); ignoring mosaic offsets y0={y0}, x0={x0}", flush=True)
        y0, x0 = 0, 0
    y1, x1 = y0 + ny_cut, x0 + nx_cut
    print(f"Extracting cutout [{i0}:{i1}, {y0}:{y1}, {x0}:{x1}]", flush=True)
    sub = cube[i0:i1, y0:y1, x0:x1].filled(np.nan)
    if hasattr(sub, "compute"):
        sub = sub.compute()
    cut = np.asarray(sub, dtype=np.float64)  ### (C, ny, nx)

    rng = np.random.default_rng(args.seed)
    pools = {
        r"$K_{\mathrm{pred}}\geq 3$": np.flatnonzero(kp >= 3),
        r"$K_{\mathrm{pred}}=2$": np.flatnonzero(np.round(kp) == 2),
        r"$K_{\mathrm{pred}}=1$": np.flatnonzero(np.round(kp) == 1),
        r"$K_{\mathrm{pred}}=0$": np.flatnonzero(np.round(kp) == 0),
    }
    sections = []
    for label, idx in pools.items():
        if idx.size == 0:
            print(f"  skip empty pool: {label}", flush=True)
            continue
        take = min(args.n_each, idx.size)
        pick = rng.choice(idx, size=take, replace=False)
        sections.append((label, sorted(pick.tolist())))

    n_col = args.n_each
    n_row = len(sections)
    if n_row == 0:
        raise RuntimeError("No finite K_pred pools to plot.")
    fig, axes = plt.subplots(n_row, n_col, figsize=(3.6 * n_col, 2.4 * n_row), squeeze=False)
    for r, (label, picks) in enumerate(sections):
        for c in range(n_col):
            ax = axes[r, c]
            if c >= len(picks):
                ax.axis("off")
                continue
            j = picks[c]
            ly = int(yi[j])
            lx = int(xi[j])
            spec = cut[:, ly, lx]
            m = np.isfinite(spec) & (spec != 0)
            ax.plot(v_axis[m], spec[m], color="0.3", lw=0.85)
            for vv in cv[j]:
                if np.isfinite(vv):
                    ax.axvline(float(vv), color="#F28E2B", ls="--", lw=0.9)
            l_deg, b_deg = galactic_lb(wcs, lx, ly)
            ax.set_title(lb_title(l_deg, b_deg), fontsize=8)
            if r == n_row - 1:
                ax.set_xlabel(r"$v$ (km/s)", fontsize=8)
            if c == 0:
                ax.set_ylabel(f"{label}\n$T$", fontsize=8)
    fig.suptitle("HNCO region1", fontsize=11)
    fig.tight_layout()
    out = args.out or (
        _SCRIPT.parents[1] / "figures" / "region1" / f"{args.k_pred.stem}_spectra_mixed.png"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
