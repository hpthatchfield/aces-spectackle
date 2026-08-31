#!/usr/bin/env python
"""Quick label-distribution sanity check for simple vs simple_matched."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from spectackle.data.mopra_generator import build_mopra_synth_cfg, generate_mopra_spectrum

_REPO = Path(__file__).resolve().parents[3]


def summarize(preset: str, n: int = 3000, seed: int = 0) -> None:
    cfg = build_mopra_synth_cfg(
        repo_root=_REPO,
        gen_preset=preset,
        max_components=6,
        noise_calibration_cube=_REPO / "data" / "CMZ_3mm_HNCO_60.fits",
    )
    rng = np.random.default_rng(seed)
    ks = []
    multi = multi_le1 = d3 = d3ok = 0
    t0 = time.time()
    for _ in range(n):
        ex = generate_mopra_spectrum(cfg, rng)
        k = int(ex["k"])
        kd = int(ex.get("k_drawn", k))
        ks.append(k)
        if kd >= 2:
            multi += 1
            if k <= 1:
                multi_le1 += 1
        if kd >= 3:
            d3 += 1
            if k >= 3:
                d3ok += 1
    ks = np.asarray(ks)
    print(f"=== {preset} (n={n}, {time.time() - t0:.1f}s) ===")
    print(
        f"  mean K={ks.mean():.3f}  P(K=1)={(ks == 1).mean():.3f}  "
        f"P(K>=3)={(ks >= 3).mean():.3f}"
    )
    print(
        f"  multi-><=1={multi_le1 / max(multi, 1):.3f}  "
        f"drawn>=3->label>=3={d3ok / max(d3, 1):.3f}"
    )
    hist = np.bincount(ks, minlength=7)[:7] / n
    print("  P(K=0..6): " + " ".join(f"{h:.3f}" for h in hist))
    assert cfg["gen"].get("glance_cap_mode") in (None, "resolvable", "matched", "residual")


def main() -> None:
    for p in ("simple", "simple_matched"):
        summarize(p)


if __name__ == "__main__":
    main()
