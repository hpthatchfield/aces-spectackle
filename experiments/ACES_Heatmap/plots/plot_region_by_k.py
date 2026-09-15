#!/usr/bin/env python
"""
One unlabeled region1 gallery per estimated K (K_pred from the cube map).

No ground truth. Orange dashed lines are the model's stored centers.

  python experiments/ACES_Heatmap/plots/plot_region_by_k.py \\
    --k-pred experiments/ACES_Heatmap/records/<stage2>/hnco_region1_aces_hm_k_pred_islands.fits \\
    --cube data/hnco_region1_native_aces.fits \\
    --tag islands
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
_DEFAULT_CUBE = _REPO / "data" / "hnco_region1_native_aces.fits"
COL_SPEC = "0.3"
COL_PRED = "#F28E2B"


def _load_cutout(k_pred_path: Path, centers_path: Path, cube_path: Path):
    k_map = fits.getdata(k_pred_path).astype(np.float32)
    k_hdr = fits.getheader(k_pred_path)
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

    print(f"Loading cube (lazy): {cube_path}", flush=True)
    cube = SpectralCube.read(str(cube_path.resolve()), use_dask=True)
    if i1 is None:
        i1 = cube.shape[0]
    if v_axis is None:
        v_axis = cube.spectral_axis.to(u.km / u.s).value.astype(np.float64)[i0:i1]

    ny_cut, nx_cut = int(k_map.shape[0]), int(k_map.shape[1])
    ny_cube, nx_cube = int(cube.shape[-2]), int(cube.shape[-1])
    if (y0 + ny_cut > ny_cube or x0 + nx_cut > nx_cube) and (ny_cube, nx_cube) == (ny_cut, nx_cut):
        print(f"Cube is already the cutout ({ny_cut}x{nx_cut}); ignoring mosaic offsets y0={y0}, x0={x0}", flush=True)
        y0, x0 = 0, 0
    y1, x1 = y0 + ny_cut, x0 + nx_cut
    print(f"Extracting cutout [{i0}:{i1}, {y0}:{y1}, {x0}:{x1}]", flush=True)
    sub = cube[i0:i1, y0:y1, x0:x1].filled(np.nan)
    if hasattr(sub, "compute"):
        sub = sub.compute()
    cut = np.asarray(sub, dtype=np.float64)
    wcs = wcs_celestial(k_hdr)
    return dict(cut=cut, v_axis=v_axis, y0=y0, x0=x0, yi=yi, xi=xi, kp=kp, cv=cv, wcs=wcs)


def _plot_k_page(
    *,
    k: int,
    idxs: np.ndarray,
    data: dict,
    n_pool: int,
    tag: str,
    out: Path,
) -> None:
    n = int(idxs.size)
    n_col = min(4, n)
    n_row = int(np.ceil(n / n_col))
    fig, axes = plt.subplots(n_row, n_col, figsize=(3.6 * n_col, 2.4 * n_row), squeeze=False)
    v_axis = data["v_axis"]
    cut = data["cut"]
    wcs = data["wcs"]
    for c, ax in enumerate(axes.ravel()):
        if c >= n:
            ax.axis("off")
            continue
        j = int(idxs[c])
        ly = int(data["yi"][j])
        lx = int(data["xi"][j])
        spec = cut[:, ly, lx]
        m = np.isfinite(spec) & (spec != 0)
        ax.plot(v_axis[m], spec[m], color=COL_SPEC, lw=0.85)
        for vv in data["cv"][j]:
            if np.isfinite(vv):
                ax.axvline(float(vv), color=COL_PRED, ls="--", lw=0.9)
        l_deg, b_deg = galactic_lb(wcs, lx, ly)
        ax.set_title(lb_title(l_deg, b_deg), fontsize=8)
        if c // n_col == n_row - 1:
            ax.set_xlabel(r"$v$ (km/s)", fontsize=8)
        if c % n_col == 0:
            ax.set_ylabel("$T$", fontsize=8)
    fig.suptitle(rf"$K_{{\mathrm{{pred}}}}={k}$  ($n={n_pool}$)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"Wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Region1 unlabeled galleries, one figure per K_pred.")
    parser.add_argument("--k-pred", type=Path, required=True)
    parser.add_argument("--centers", type=Path, default=None)
    parser.add_argument("--cube", type=Path, default=_DEFAULT_CUBE)
    parser.add_argument("--tag", type=str, default="islands", help="Name prefix in the output files.")
    parser.add_argument("--n-each", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    centers_path = args.centers or args.k_pred.with_name(f"{args.k_pred.stem}_centers.npz")
    data = _load_cutout(args.k_pred, centers_path, args.cube)
    kp_round = np.round(data["kp"]).astype(np.int64)
    out_dir = args.out_dir or (_ACES / "figures" / "region1")
    rng = np.random.default_rng(args.seed)

    ks = sorted(int(k) for k in np.unique(kp_round) if np.isfinite(k))
    if not ks:
        raise RuntimeError("No finite K_pred values in the centers file.")
    print(f"K_pred values: {ks}", flush=True)
    for k in ks:
        pool = np.flatnonzero(kp_round == k)
        n_pool = int(pool.size)
        if n_pool == 0:
            continue
        take = min(int(args.n_each), n_pool)
        pick = np.sort(rng.choice(pool, size=take, replace=False))
        print(f"  K={k}: n={n_pool}, plotting {take}", flush=True)
        _plot_k_page(
            k=k,
            idxs=pick,
            data=data,
            n_pool=n_pool,
            tag=args.tag,
            out=out_dir / f"{args.tag}_k{k}.png",
        )


if __name__ == "__main__":
    main()
