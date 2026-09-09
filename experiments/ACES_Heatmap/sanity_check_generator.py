#!/usr/bin/env python
"""
Sanity checks for ACES synth generator (histograms + example spectra/heatmaps).

Default preset is simple_snr (SNR prune, soft min sep, narrow lognormal FWHM).

Run from repo root:
  python experiments/ACES_Heatmap/sanity_check_generator.py
  python experiments/ACES_Heatmap/sanity_check_generator.py --gen-preset islands --Kmax 10
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
from spectackle.data.resolvable_peaks import blend_stats_from_synth_example  ### noqa: E402
from spectackle.training import build_center_target_map  ### noqa: E402

### Post-hoc morphology cuts (not generator flags). sep_sigma = |dv| / (sigma_i + sigma_j).
BLEND_SEP_SIGMA = 1.2
ISOLATED_SEP_SIGMA = 2.0


def main() -> None:
    parser = argparse.ArgumentParser(description="ACES generator sanity histograms.")
    parser.add_argument("--gen-preset", choices=("simple_snr", "simple_glance", "default", "islands"), default="simple_snr")
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
    sep_sigma, amp_ratio = [], []
    n_islands_l, island_sizes_l = [], []
    n_kge2 = n_blend = n_isol = n_mid = 0
    blended_ex: list[dict] = []
    isolated_ex: list[dict] = []
    family_ex: list[dict] = []
    two_island_ex: list[dict] = []
    highk_ex: list[dict] = []
    for i in range(args.n_samples):
        ex = generate_aces_spectrum(cfg, np.random.default_rng(args.seed + i), v_axis=v)
        k = int(ex["k"])
        kd = int(ex.get("k_drawn", k))
        ks_label.append(k)
        ks_drawn.append(kd)
        vm = valid_mask(ex["spec"])
        valid_fracs.append(float(vm.mean()))
        if "n_islands" in ex:
            n_islands_l.append(int(ex["n_islands"]))
            island_sizes_l.extend(int(s) for s in np.asarray(ex["island_sizes"]).tolist() if int(s) > 0)
            n_isl = int(ex["n_islands"])
            if k >= 2 and n_isl == 1 and len(family_ex) < 3:
                family_ex.append(ex)
            if k >= 2 and n_isl >= 2 and len(two_island_ex) < 3:
                two_island_ex.append(ex)
        if k >= 5 and len(highk_ex) < 3:
            highk_ex.append(ex)
        for j in range(k):
            sig = float(ex["component_sigma"][j])
            fwhm = 2.355 * sig
            fwhm_kms.append(fwhm)
            fwhm_ch.append(fwhm / max(cw, 1e-6))
        st = blend_stats_from_synth_example(ex)
        if k >= 2 and np.isfinite(st["min_sep_sigma"]):
            n_kge2 += 1
            ss = float(st["min_sep_sigma"])
            sep_sigma.append(ss)
            amp_ratio.append(float(st["min_amp_ratio"]))
            if ss <= BLEND_SEP_SIGMA:
                n_blend += 1
                if len(blended_ex) < 4:
                    blended_ex.append(ex)
            elif ss >= ISOLATED_SEP_SIGMA:
                n_isol += 1
                if len(isolated_ex) < 4:
                    isolated_ex.append(ex)
            else:
                n_mid += 1
        elif k == 1 and len(isolated_ex) < 4:
            isolated_ex.append(ex)

    if family_ex or two_island_ex or highk_ex:
        examples = isolated_ex[:2] + family_ex[:2] + two_island_ex[:2] + highk_ex[:2]
    else:
        examples = isolated_ex[:4] + blended_ex[:4]

    fig, axes = plt.subplots(2, 4, figsize=(14.5, 6.2))
    axes[0, 0].hist(ks_drawn, bins=np.arange(-0.5, args.Kmax + 1.5), color="#4E79A7", edgecolor="white")
    axes[0, 0].set_title("K drawn")
    axes[0, 1].hist(ks_label, bins=np.arange(-0.5, args.Kmax + 1.5), color="#59A14F", edgecolor="white")
    axes[0, 1].set_title("K SNR label")
    axes[0, 2].hist(np.asarray(ks_drawn) - np.asarray(ks_label), bins=np.arange(-0.5, args.Kmax + 1.5),
                     color="#E15759", edgecolor="white")
    axes[0, 2].set_title("K_drawn - K_label")
    axes[0, 3].hist(sep_sigma, bins=30, color="#E15759", edgecolor="white")
    axes[0, 3].axvline(BLEND_SEP_SIGMA, color="0.3", ls="--", lw=0.8)
    axes[0, 3].axvline(ISOLATED_SEP_SIGMA, color="0.3", ls=":", lw=0.8)
    axes[0, 3].set_xlabel("|dv| / (sig_i+sig_j)")
    axes[0, 3].set_title("K>=2 nearest sep (sigma units)")
    axes[1, 0].hist(fwhm_kms, bins=40, color="#F28E2B", edgecolor="white")
    axes[1, 0].set_xlabel("FWHM (km/s)")
    axes[1, 0].set_title("Component FWHM")
    axes[1, 1].hist(fwhm_ch, bins=40, color="#B07AA1", edgecolor="white")
    axes[1, 1].set_xlabel("FWHM (channels)")
    axes[1, 1].set_title(f"FWHM in channels (dv={cw:.3f})")
    axes[1, 2].hist(valid_fracs, bins=30, color="#76B7B2", edgecolor="white")
    axes[1, 2].set_xlabel("valid fraction")
    axes[1, 2].set_title("ALMA mask coverage")
    axes[1, 3].hist(amp_ratio, bins=20, color="#EDC948", edgecolor="white")
    axes[1, 3].set_xlabel("min(amp)/max(amp)")
    axes[1, 3].set_title("K>=2 amp ratio")
    fig.suptitle(
        f"ACES {args.gen_preset} sanity (n_ch={cfg['n_channels']}, +/-{args.v_half_kms:g} km/s, Kmax={args.Kmax})"
    )
    fig.tight_layout()
    fig.savefig(out_dir / "histograms.png", dpi=120)
    plt.close(fig)

    if n_islands_l:
        fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4))
        axes[0].hist(n_islands_l, bins=np.arange(-0.5, 5.5), color="#4E79A7", edgecolor="white")
        axes[0].set_title("n_islands (K>0 draws)")
        axes[0].set_xlabel("islands")
        axes[1].hist(island_sizes_l, bins=np.arange(0.5, args.Kmax + 1.5), color="#F28E2B", edgecolor="white")
        axes[1].set_title("components per island")
        axes[1].set_xlabel("chain length")
        axes[2].hist(ks_label, bins=np.arange(-0.5, args.Kmax + 1.5), color="#59A14F", edgecolor="white")
        axes[2].set_title("K SNR label")
        axes[2].set_xlabel("K")
        fig.suptitle(f"ACES {args.gen_preset} island / chain counts (n={args.n_samples})")
        fig.tight_layout()
        fig.savefig(out_dir / "island_counts.png", dpi=120)
        plt.close(fig)
        print(f"Wrote {out_dir / 'island_counts.png'}")

    v_t = torch.as_tensor(v, dtype=torch.float32)
    n = len(examples)
    if n == 0:
        print("No mixed examples collected (increase --n-samples).")
    else:
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
            st = blend_stats_from_synth_example(ex)
            ss = st["min_sep_sigma"]
            ss_txt = f"  sep/sig={ss:.2f}" if np.isfinite(ss) else ""
            n_isl = int(ex.get("n_islands", -1))
            isl_txt = f"  islands={n_isl}" if n_isl >= 0 else ""
            ax.set_title(f"K_label={k}  K_drawn={ex.get('k_drawn', k)}{ss_txt}{isl_txt}", fontsize=9)
        axes[-1].set_xlabel("v (km/s)")
        fig.suptitle(
            f"Examples + heatmap target (label_sigma={args.label_sigma_kms} km/s); "
            f"top isolated, bottom blended",
            fontsize=11,
        )
        fig.tight_layout()
        fig.savefig(out_dir / "example_spectra_targets.png", dpi=120)
        plt.close(fig)

    print(f"n_ch={cfg['n_channels']}  dv={cw:.4f} km/s  vrange={cfg['vrange']}")
    print(f"K_label mean={np.mean(ks_label):.2f}  K_drawn mean={np.mean(ks_drawn):.2f}")
    if n_kge2:
        print(
            f"K>=2 n={n_kge2}: blended(sep/sig<={BLEND_SEP_SIGMA:g})={n_blend} "
            f"({100 * n_blend / n_kge2:.1f}%)  "
            f"isolated(>={ISOLATED_SEP_SIGMA:g})={n_isol} ({100 * n_isol / n_kge2:.1f}%)  "
            f"mid={n_mid}"
        )
        print(f"K>=2 median sep/sig={np.median(sep_sigma):.2f}  median amp ratio={np.median(amp_ratio):.2f}")
    if n_islands_l:
        arr_k = np.asarray(ks_label)
        print(
            f"n_islands mean={np.mean(n_islands_l):.2f}  "
            f"per-island size mean={np.mean(island_sizes_l):.2f}  "
            f"P(K>=8)={100 * np.mean(arr_k >= 8):.1f}%  P(K=10)={100 * np.mean(arr_k == 10):.1f}%"
        )
    print(f"Wrote {out_dir / 'histograms.png'}")
    print(f"Wrote {out_dir / 'example_spectra_targets.png'}")


if __name__ == "__main__":
    main()
