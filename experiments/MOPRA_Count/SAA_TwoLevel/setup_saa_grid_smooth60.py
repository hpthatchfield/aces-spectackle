#!/usr/bin/env python
"""Build Scouse-style SAA grid on smooth60 cube (stage-1 coverage, no human input)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_SCRIPT = Path(__file__).resolve()
_REPO = _SCRIPT.parents[3]
_OUT = _SCRIPT.parent / "runs" / "saa_grid_smooth60"


def main() -> None:
    cmd = [
        sys.executable,
        str(_SCRIPT.parents[1] / "setup_saa_grid.py"),
        "--cube",
        str(_REPO / "data" / "CMZ_3mm_HNCO_60.fits"),
        "--out",
        str(_OUT),
    ]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
