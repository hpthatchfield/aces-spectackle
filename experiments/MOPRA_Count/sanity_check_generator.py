#!/usr/bin/env python
"""
Sanity checks for MOPRA-axis synthetic spectra (histograms + example grid).

Run from repository root:
  python experiments/MOPRA_Count/sanity_check_generator.py

Artifacts: experiments/MOPRA_Count/figures/sanity/
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_SCRIPT = Path(__file__).resolve()
_REPO = _SCRIPT.parents[2]
sys.path.insert(0, str(_REPO / "src"))

from spectackle.data.generator import _make_v_axis  ### noqa: E402
from spectackle.data.mopra_generator import (  ### noqa: E402
    build_mopra_synth_cfg,
    estimate_mopra_noise_from_cube,
    generate_mopra_spectrum,
)
from spectackle.data.mopra_preprocess import valid_mask_mopra  ### noqa: E402
from spectackle.data.scouse_saa import estimate_spectrum_rms  ### noqa: E402

N_SAMPLES = 500
SEED = 42


def _cube_rms_sample(cube_path: Path, *, n_sample: int = 800, seed: int = 0) -> np.ndarray:
    from spectral_cube import SpectralCube

    cube = SpectralCube.read(str(cube_path.resolve()), use_dask=False)
    arr = np.asarray(cube.filled(np.nan), dtype=np.float32)
    if arr.ndim == 4:
        arr = arr[0]
    nv, ny, nx = arr.shape
    rng = np.random.default_rng(seed)
    idx = rng.choice(ny * nx, size=min(n_sample, ny * nx), replace=False)
    out: list[float] = []
    for flat in idx:
        y, x = divmod(int(flat), nx)
        spec = arr[:, y, x].astype(np.float64)
        ok = valid_mask_mopra(spec, blank_value=-1.0)
        if ok.sum() < max(40, nv // 6):
            continue
        r = estimate_spectrum_rms(spec[ok])
        if np.isfinite(r) and r > 0.0:
            out.append(float(r))
    return np.asarray(out, dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--Kmax", type=int, default=10)
    parser.add_argument(
        "--cube",
        type=Path,
        default=_REPO / "data" / "CMZ_3mm_HNCO.fits",
        help="Axis metadata cube for synthetic spectra.",
    )
    parser.add_argument(
        "--noise-cube",
        type=Path,
        default=_REPO / "data" / "CMZ_3mm_HNCO_60.fits",
        help="Smoothed cube for sigma_rms comparison histogram.",
    )
    parser.add_argument(
        "--gen-preset",
        choices=("default", "scouse_smooth60", "legacy"),
        default="default",
    )
    args = parser.parse_args()

    cfg = build_mopra_synth_cfg(
        repo_root=_REPO,
        cube_path=args.cube,
        max_components=args.Kmax,
        gen_preset=args.gen_preset,
        noise_calibration_cube=args.noise_cube if args.noise_cube.is_file() else None,
    )
    meta = cfg.get("mopra_meta", {})
    gen = cfg.get("gen", {})
    v = _make_v_axis(cfg)

    if args.noise_cube.is_file():
        cal = estimate_mopra_noise_from_cube(args.noise_cube)
        print(f"Noise calibration from {args.noise_cube.name}: {cal}")
    print(f"gen.noise_std_range={gen.get('noise_std_range')}  snr_range={gen.get('snr_range')}")
    print(f"n_channels={cfg['n_channels']}  vrange={cfg['vrange']}")
    print(f"channel_width_kms~{meta.get('channel_width_kms', '?')}  bunit={meta.get('bunit', '?')}")

    out_dir = _SCRIPT.parent / "figures" / "sanity"
    out_dir.mkdir(parents=True, exist_ok=True)

    ks, peaks, sig_min, noise, snr_drawn = [], [], [], [], []
    for i in range(N_SAMPLES):
        ex = generate_mopra_spectrum(cfg, np.random.default_rng(SEED + i), v_axis=v)
        ks.append(ex["k"])
        valid = np.isfinite(ex["spec"]) & (ex["spec"] != 0.0)
        peaks.append(float(np.max(ex["spec_clean"][valid])) if valid.any() else 0.0)
        ns = float(ex["noise_std"][0])
        noise.append(ns)
        if ex["k"] > 0:
            sig_min.append(float(np.min(ex["component_sigma"][: ex["k"]])))
            amps = ex["component_amp"][: ex["k"]]
            snr_drawn.append(float(np.max(amps) / ns))

    ks = np.array(ks)
    noise = np.array(noise)
    print(f"K histogram: mean={ks.mean():.2f}  max={ks.max()}")
    print(f"synth noise_std: med={np.median(noise):.4f}  snr_drawn max med={np.median(snr_drawn):.2f}")

    fig, axes = plt.subplots(1, 4, figsize=(14, 3))
    axes[0].hist(ks, bins=np.arange(-0.5, args.Kmax + 1.5), color="#4E79A7", edgecolor="white")
    axes[0].set_xlabel("k (components)")
    axes[0].set_title("Component count")
    axes[1].hist(peaks, bins=40, color="#F28E2B", edgecolor="white")
    axes[1].set_xlabel("peak spec_clean (K)")
    axes[1].set_title("Peak height")
    if sig_min:
        axes[2].hist(sig_min, bins=40, color="#59A14F", edgecolor="white")
    axes[2].set_xlabel("min sigma (km/s) per spectrum")
    axes[2].set_title("Narrowest component sigma")
    axes[3].hist(noise, bins=40, color="#76B7B2", edgecolor="white", alpha=0.85, label="synth")
    if args.noise_cube.is_file():
        real_rms = _cube_rms_sample(args.noise_cube)
        axes[3].hist(
            real_rms,
            bins=40,
            color="#E15759",
            edgecolor="white",
            alpha=0.55,
            label=f"{args.noise_cube.stem}",
        )
        axes[3].legend(fontsize=7)
    axes[3].set_xlabel("sigma_rms (K)")
    axes[3].set_title("Noise scale")
    fig.tight_layout()
    p_hist = out_dir / "mopra_synth_histograms.png"
    fig.savefig(p_hist, dpi=120)
    plt.close(fig)
    print(f"Wrote {p_hist}")

    fig, axes = plt.subplots(2, 3, figsize=(12, 5), sharex=True)
    for ax, idx in zip(axes.ravel(), [0, 1, 2, 3, 4, 5]):
        ex = generate_mopra_spectrum(cfg, np.random.default_rng(SEED + 1000 + idx), v_axis=v)
        ax.plot(v, ex["spec"], lw=0.9, color="#4E79A7", label="noisy")
        ax.plot(v, ex["spec_clean"], lw=0.9, color="#E15759", alpha=0.85, label="clean")
        ax.set_title(f"k={ex['k']}  sigma_noise={float(ex['noise_std'][0]):.3f}")
        ax.set_xlim(v[0], v[-1])
    axes[0, 0].legend(fontsize=7)
    for ax in axes[1]:
        ax.set_xlabel("v (km/s)")
    fig.suptitle(f"MOPRA synthetic examples ({args.gen_preset})")
    fig.tight_layout()
    p_grid = out_dir / "mopra_synth_examples.png"
    fig.savefig(p_grid, dpi=120)
    plt.close(fig)
    print(f"Wrote {p_grid}")


if __name__ == "__main__":
    main()
