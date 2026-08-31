#!/usr/bin/env python
"""
Build NPZ cache of Scouse-labeled spectra from final_fits_updated.dat + smooth60 cube.

Run from repo root:
  python experiments/MOPRA_Count/build_scouse_cache.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPT = Path(__file__).resolve()
_REPO = _SCRIPT.parents[2]
sys.path.insert(0, str(_REPO / "src"))

from spectackle.data.mopra_scouse_labels import build_scouse_labeled_cache  ### noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Scouse .dat labeled spectrum cache.")
    parser.add_argument(
        "--dat",
        type=Path,
        default=_REPO / "data" / "final_fits_updated.dat",
    )
    parser.add_argument(
        "--cube",
        type=Path,
        default=_REPO / "data" / "CMZ_3mm_HNCO_60.fits",
        help="Scouse-input smoothed cube (must match .dat WCS).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=_SCRIPT.parent / "cache" / "scouse_labeled_smooth60.npz",
    )
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--cell-deg", type=float, default=0.08, help="Spatial split cell size (deg).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-row-count", action="store_true", help="K = n rows (default: col0 ncomps).")
    args = parser.parse_args()

    meta = build_scouse_labeled_cache(
        dat_path=args.dat,
        cube_path=args.cube,
        out_path=args.out,
        val_frac=args.val_frac,
        cell_deg=args.cell_deg,
        seed=args.seed,
        use_row_count=args.use_row_count,
    )
    print(f"Wrote {meta['cache_path']}")
    print(f"  n_train={meta['n_train']}  n_val={meta['n_val']}  n_ch={meta['n_channels']}")
    print(f"  K_hist={meta['K_hist']}")


if __name__ == "__main__":
    main()
