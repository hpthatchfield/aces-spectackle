#!/usr/bin/env python
"""
Region1 diagnostic gallery: spectrum + predicted heatmap P(center).

Rows are by K_pred from the cube map (0/1/2/3). K_pred=0 is empty on the
current map, so that row samples NaN/filtered cutout pixels and re-runs the
model for display (shows what the heatmap does on quiet/empty spectra).

  python experiments/ACES_Heatmap/plots/plot_region_heatmap_gallery.py \\
    --k-pred data/hnco_region1_aces_hm_k_pred.fits \\
    --run-dir experiments/ACES_Heatmap/runs/<heatmap_count_run>
"""
from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import json
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from astropy import units as u
from astropy.io import fits
from spectral_cube import SpectralCube

_SCRIPT = Path(__file__).resolve()
_ACES = _SCRIPT.parents[1]
_REPO = _ACES.parents[1]
sys.path.insert(0, str(_REPO / "src"))

from spectackle.data.generator import _make_v_axis  ### noqa: E402
from spectackle.data.preprocess import prepare_spectrum_input  ### noqa: E402
from spectackle.models import CenterHeatmapNet1DDeep, HeatmapCountNet  ### noqa: E402
from spectackle.models.center_heatmap_decode import decode_centers_from_heatmap  ### noqa: E402

_DEFAULT_MOSAIC = _REPO / "data" / (
    "group.uid___A001_X1590_X30a9.lp_slongmore.cmz_mosaic.12m7mTP.HNCO_7m12mTP.cube.pbcor.fits"
)


def _load_model(run_dir: Path, *, device: str) -> tuple[HeatmapCountNet, dict, np.ndarray]:
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    stage1_dir = Path(manifest["stage1_run_dir"])
    if not stage1_dir.is_absolute():
        stage1_dir = (_REPO / stage1_dir).resolve()
    s1 = json.loads((stage1_dir / "manifest.json").read_text(encoding="utf-8"))
    s1_args = dict(s1.get("args", {}))
    args = dict(manifest.get("args", {}))
    cfg = manifest["cfg"]
    kernel_size = int(manifest.get("kernel_size", s1.get("kernel_size", 25)))
    heatmap = CenterHeatmapNet1DDeep(
        width=int(s1_args.get("width", 96)),
        n_blocks=int(s1_args.get("n_blocks", 6)),
        kernel_size=kernel_size,
    )
    heatmap.load_state_dict(torch.load(stage1_dir / "center_heatmap_net.pt", map_location="cpu"))
    model = HeatmapCountNet(
        heatmap,
        width=int(args.get("width", 96)),
        n_blocks=int(args.get("n_blocks", 6)),
        k_input=str(manifest.get("k_input", "spec_p")),
        freeze_heatmap=True,
        kernel_size=kernel_size,
    )
    model.load_state_dict(torch.load(run_dir / "heatmap_count_net.pt", map_location="cpu"))
    model.to(device)
    model.eval()
    v_model = _make_v_axis(cfg).astype(np.float64)
    return model, manifest, v_model


@torch.no_grad()
def _infer_one(
    model: HeatmapCountNet,
    spec_raw: np.ndarray,
    *,
    device: str,
    v_axis: np.ndarray,
    Kmax: int,
    height: float,
    prominence: float,
    min_sep_kms: float,
) -> dict:
    xn, vm = prepare_spectrum_input(spec_raw)
    t = torch.from_numpy(xn[None]).to(device)
    m = torch.from_numpy(vm[None]).to(device)
    k_hat = float(model(t, m).cpu().numpy().reshape(-1)[0])
    k_round = int(np.clip(np.round(k_hat), 0, Kmax))
    prob = torch.sigmoid(model.heatmap_logits(t, m)).cpu().numpy()[0]
    v_slots = np.full(Kmax, np.nan, dtype=np.float32)
    if k_round > 0:
        _kd, peak_idx = decode_centers_from_heatmap(
            prob, v_axis, valid_mask=vm,
            height=height, prominence=prominence,
            min_sep_kms=min_sep_kms, Kmax=k_round,
        )
        for j, ix in enumerate(peak_idx.tolist()):
            if j >= Kmax:
                break
            v_slots[j] = float(v_axis[int(ix)])
    return {
        "spec_norm": xn,
        "valid": vm,
        "prob": prob.astype(np.float32),
        "k_hat": k_hat,
        "k_round": k_round,
        "v_slots": v_slots,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Region1 spectra + heatmap diagnostic gallery.")
    parser.add_argument("--k-pred", type=Path, default=_REPO / "data" / "hnco_region1_aces_hm_k_pred.fits")
    parser.add_argument("--centers", type=Path, default=None)
    parser.add_argument("--run-dir", type=Path, required=True, help="heatmap_count run directory")
    parser.add_argument("--cube", type=Path, default=_DEFAULT_MOSAIC)
    parser.add_argument("--n-each", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    k_map = fits.getdata(args.k_pred).astype(np.float32)
    centers_path = args.centers or args.k_pred.with_name(f"{args.k_pred.stem}_centers.npz")
    z = np.load(centers_path, allow_pickle=True)
    y0, x0 = int(z["y0"]), int(z["x0"])
    i0, i1 = int(z["i0"]), int(z["i1"])
    v_axis = np.asarray(z["v_axis"], dtype=np.float64)
    yi = np.asarray(z["yi"], dtype=np.int64)
    xi = np.asarray(z["xi"], dtype=np.int64)
    kp = np.asarray(z["k_pred"], dtype=np.float32)

    model, manifest, _v_model = _load_model(args.run_dir.resolve(), device=args.device)
    dec = dict(manifest.get("decode_stage1") or {})
    height = float(dec.get("height", 0.25))
    prom = float(dec.get("prominence", 0.08))
    min_sep = float(dec.get("min_sep_kms", 4.0))
    Kmax = int(dec.get("Kmax", 6))

    print(f"Loading mosaic cutout (lazy): {args.cube}", flush=True)
    cube = SpectralCube.read(str(args.cube.resolve()), use_dask=True)
    ny, nx = k_map.shape
    y1, x1 = y0 + ny, x0 + nx
    print(f"Extracting [{i0}:{i1}, {y0}:{y1}, {x0}:{x1}]", flush=True)
    sub = cube[i0:i1, y0:y1, x0:x1].filled(np.nan)
    if hasattr(sub, "compute"):
        sub = sub.compute()
    cut = np.asarray(sub, dtype=np.float32)  ### (C, ny, nx)

    rng = np.random.default_rng(args.seed)
    ### Pixel picks as (ly, lx) in cutout coords.
    picks: list[tuple[str, list[tuple[int, int]]]] = []

    ### Quiet row: lowest finite peak among cutout spectra (proxy for K~0 / empty).
    peak_yx = []
    for ly in range(ny):
        for lx in range(nx):
            spec = cut[:, ly, lx]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                peak = float(np.nanmax(np.where(np.isfinite(spec) & (spec != 0), spec, np.nan)))
            if not np.isfinite(peak):
                continue
            n_fin = int(np.sum(np.isfinite(spec) & (spec != 0)))
            if n_fin < 50:
                continue
            peak_yx.append((peak, ly, lx))
    peak_yx.sort(key=lambda t: t[0])
    if peak_yx:
        take = min(args.n_each, len(peak_yx))
        pool = peak_yx[: max(take * 5, take)]
        chosen = rng.choice(len(pool), size=take, replace=False)
        coords = [(pool[i][1], pool[i][2]) for i in sorted(chosen.tolist())]
        picks.append(("quietest (low peak)", coords))

    for k_want in (1, 2, 3):
        pool_j = np.flatnonzero(np.round(kp) == k_want)
        if pool_j.size == 0:
            print(f"  skip K_pred={k_want}: empty", flush=True)
            continue
        take = min(args.n_each, int(pool_j.size))
        chosen = rng.choice(pool_j, size=take, replace=False)
        coords = [(int(yi[j]), int(xi[j])) for j in sorted(chosen.tolist())]
        picks.append((f"K_pred={k_want}", coords))

    if not picks:
        raise RuntimeError("No spectra selected.")

    n_row, n_col = len(picks), args.n_each
    fig, axes = plt.subplots(n_row, n_col, figsize=(3.8 * n_col, 2.6 * n_row), squeeze=False)
    for r, (label, coords) in enumerate(picks):
        for c in range(n_col):
            ax = axes[r, c]
            if c >= len(coords):
                ax.axis("off")
                continue
            ly, lx = coords[c]
            spec_raw = cut[:, ly, lx]
            out = _infer_one(
                model, spec_raw, device=args.device, v_axis=v_axis, Kmax=Kmax,
                height=height, prominence=prom, min_sep_kms=min_sep,
            )
            valid = out["valid"] > 0.5
            ax.plot(v_axis[valid], out["spec_norm"][valid], color="0.35", lw=0.9)
            ax2 = ax.twinx()
            ax2.plot(v_axis[valid], out["prob"][valid], color="#C62828", lw=1.0, alpha=0.9)
            ax2.set_ylim(-0.05, 1.05)
            ax2.tick_params(axis="y", labelsize=7, colors="#C62828")
            if c == n_col - 1:
                ax2.set_ylabel("P(center)", fontsize=8, color="#C62828")
            else:
                ax2.set_yticklabels([])
            for vv in out["v_slots"]:
                if np.isfinite(vv):
                    ax.axvline(float(vv), color="#F28E2B", ls="--", lw=0.9)
            ax.set_title(
                f"({y0 + ly},{x0 + lx})  Khat={out['k_hat']:.2f}->{out['k_round']}",
                fontsize=8,
            )
            if c == 0:
                ax.set_ylabel(f"{label}\nT (norm)", fontsize=8)
            if r == n_row - 1:
                ax.set_xlabel("v (km/s)", fontsize=8)
    fig.suptitle(
        f"Region1 spectra + heatmap  ({args.run_dir.name})\n"
        f"grey=norm spectrum, red=P(center), orange dashed=top-K peaks",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = args.out or (
        _ACES / "figures" / "failure_spectra" / f"{args.k_pred.stem}_heatmap_gallery.png"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
