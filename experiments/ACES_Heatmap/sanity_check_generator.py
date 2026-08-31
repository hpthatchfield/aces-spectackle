#!/usr/bin/env python
"""
Sanity checks for ACES synth generator (histograms + example spectra/heatmaps).

Default preset is simple_snr (SNR prune, soft min sep, narrow lognormal FWHM).

Run from repo root:
  python experiments/ACES_Heatmap/sanity_check_generator.py
  python experiments/ACES_Heatmap/sanity_check_generator.py --gen-preset simple_glance
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

_SCRIPT = Path(__file__).resolve()
_REPO = _SCRIPT.parents[2]
sys.path.insert(0, str(_REPO / "src"))

from spectackle.data.aces_generator import build_aces_synth_cfg, generate_aces_spectrum  ### noqa: E402
from spectackle.data.generator import _make_v_axis, channel_width_kms  ### noqa: E402
from spectackle.data.preprocess import valid_mask  ### noqa: E402
from spectackle.training import build_center_target_map  ### noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="ACES generator sanity histograms.")
    parser.add_argument("--gen-preset", choices=("simple_snr", "simple_glance", "default"), default="simple_snr")
    parser.add_argument("--Kmax", type=int, default=6)
    parser.add_argument("--v-half-kms", type=float, default=80.0)
    parser.add_argument("--n-samples", type=int, default=400)
    parser.add_argument("--label-sigma-kms", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=_SCRIPT.parent / "figures" / "sanity")
    args = parser.parse_args()

    cfg = build_aces_synth_cfg(
        Kmax=args.Kmax,
        v_half_width_kms=float(args.v_half_kms),
        gen_preset=args.gen_preset,
    )
    v = _make_v_axis(cfg)
    cw = float(channel_width_kms(cfg))
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    ks_drawn, ks_label, fwhm_kms, fwhm_ch, valid_fracs = [], [], [], [], []
    examples: list[dict] = []
    for i in range(args.n_samples):
        ex = generate_aces_spectrum(cfg, np.random.default_rng(args.seed + i), v_axis=v)
        k = int(ex["k"])
        kd = int(ex.get("k_drawn", k))
        ks_label.append(k)
        ks_drawn.append(kd)
        vm = valid_mask(ex["spec"])
        valid_fracs.append(float(vm.mean()))
        for j in range(k):
            sig = float(ex["component_sigma"][j])
            fwhm = 2.355 * sig
            fwhm_kms.append(fwhm)
            fwhm_ch.append(fwhm / max(cw, 1e-6))
        if len(examples) < 8:
            examples.append(ex)

    fig, axes = plt.subplots(2, 3, figsize=(12, 6.2))
    axes[0, 0].hist(ks_drawn, bins=np.arange(-0.5, args.Kmax + 1.5), color="#4E79A7", edgecolor="white")
    axes[0, 0].set_title("K drawn")
    axes[0, 1].hist(ks_label, bins=np.arange(-0.5, args.Kmax + 1.5), color="#59A14F", edgecolor="white")
    axes[0, 1].set_title("K glance label")
    axes[0, 2].hist(np.asarray(ks_drawn) - np.asarray(ks_label), bins=np.arange(-0.5, args.Kmax + 1.5),
                     color="#E15759", edgecolor="white")
    axes[0, 2].set_title("K_drawn - K_label")
    axes[1, 0].hist(fwhm_kms, bins=40, color="#F28E2B", edgecolor="white")
    axes[1, 0].set_xlabel("FWHM (km/s)")
    axes[1, 0].set_title("Component FWHM")
    axes[1, 1].hist(fwhm_ch, bins=40, color="#B07AA1", edgecolor="white")
    axes[1, 1].set_xlabel("FWHM (channels)")
    axes[1, 1].set_title(f"FWHM in channels (dv={cw:.3f})")
    axes[1, 2].hist(valid_fracs, bins=30, color="#76B7B2", edgecolor="white")
    axes[1, 2].set_xlabel("valid fraction")
    axes[1, 2].set_title("ALMA mask coverage")
    fig.suptitle(
        f"ACES {args.gen_preset} sanity (n_ch={cfg['n_channels']}, +/-{args.v_half_kms:g} km/s, Kmax={args.Kmax})"
    )
    fig.tight_layout()
    fig.savefig(out_dir / "histograms.png", dpi=120)
    plt.close(fig)

    v_t = torch.as_tensor(v, dtype=torch.float32)
    n = len(examples)
    fig, axes = plt.subplots(n, 1, figsize=(10, 1.7 * n), sharex=True)
    if n == 1:
        axes = [axes]
    for ax, ex in zip(axes, examples):
        m = valid_mask(ex["spec"])
        ax.plot(v[m], ex["spec"][m], color="0.35", lw=0.8)
        k = int(ex["k"])
        mus = ex["component_v_kms"][:k]
        for mu in mus:
            ax.axvline(float(mu), color="tab:blue", ls="--", lw=0.8)
        ### Soft heatmap target for eyeballing label_sigma.
        if k > 0:
            vc = torch.as_tensor(ex["component_v_kms"][:k], dtype=torch.float32).view(1, -1)
            ok = torch.ones(1, k)
            tgt = build_center_target_map(vc, ok, v_t, label_sigma_kms=args.label_sigma_kms)[0].numpy()
            ax2 = ax.twinx()
            ax2.plot(v[m], tgt[m], color="tab:red", lw=0.9, alpha=0.85)
            ax2.set_ylim(-0.05, 1.05)
            ax2.set_ylabel("target P", fontsize=7, color="tab:red")
        ax.set_ylabel("T", fontsize=8)
        ax.set_title(f"K_label={k}  K_drawn={ex.get('k_drawn', k)}", fontsize=9)
    axes[-1].set_xlabel("v (km/s)")
    fig.suptitle(f"Examples + heatmap target (label_sigma={args.label_sigma_kms} km/s)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "example_spectra_targets.png", dpi=120)
    plt.close(fig)

    print(f"n_ch={cfg['n_channels']}  dv={cw:.4f} km/s  vrange={cfg['vrange']}")
    print(f"K_label mean={np.mean(ks_label):.2f}  K_drawn mean={np.mean(ks_drew := ks_drawn):.2f}")
    print(f"Wrote {out_dir / 'histograms.png'}")
    print(f"Wrote {out_dir / 'example_spectra_targets.png'}")


if __name__ == "__main__":
    main()
