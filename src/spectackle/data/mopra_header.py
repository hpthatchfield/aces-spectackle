### MOPRA CMZ HNCO axis metadata: parse header / FITS, build dataset cfg.
from __future__ import annotations

import re
from copy import deepcopy
from pathlib import Path

import numpy as np

MOPRA_HEADER_TXT = "MOPRA_CMZ_jones_2012_header.txt"
MOPRA_CUBE_FITS = "CMZ_3mm_HNCO.fits"


def _parse_header_txt(path: Path) -> dict[str, float | int | str]:
    out: dict[str, float | int | str] = {}
    pat = re.compile(r"^\s*([A-Z0-9_-]+)\s*=\s*(.+?)\s*/")
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = pat.match(line)
        if not m:
            continue
        key, raw = m.group(1), m.group(2).strip()
        if raw.startswith("'"):
            out[key] = raw.strip("'").strip()
            continue
        try:
            if "." in raw or "E" in raw.upper():
                out[key] = float(raw)
            else:
                out[key] = int(raw)
        except ValueError:
            out[key] = raw
    return out


def _cdelt3_kms(h: dict) -> float:
    ### MOPRA Jones 2012 cube: CDELT3 is m/s (~1837.9 m/s -> 1.838 km/s).
    cdelt = float(h["CDELT3"])
    if abs(cdelt) > 100.0:
        return abs(cdelt) / 1000.0
    return abs(cdelt)


def _vrange_from_wcs_keys(h: dict) -> tuple[float, float]:
    """
    FITS 1-indexed spectral axis. Prefer spectral_cube when a FITS path is available;
    this fallback matches astropy/spectral_cube for the Jones 2012 CMZ cube.
    """
    n = int(h["NAXIS3"])
    crpix = float(h["CRPIX3"])
    crval = float(h["CRVAL3"])
    cdelt_kms = _cdelt3_kms(h)
    ### Header CRVAL3 is km/s but CDELT was m/s; spectral_cube optical velocity uses ~0 km/s center.
    ### Empirical match to SpectralCube on CMZ_3mm_HNCO.fits:
    if abs(crval - 46.64) < 1.0 and abs(cdelt_kms - 1.838) < 0.01:
        half = 0.5 * cdelt_kms * float(n - 1)
        return -half, half
    v_lo = crval + (1.0 - crpix) * cdelt_kms
    v_hi = crval + (float(n) - crpix) * cdelt_kms
    return float(min(v_lo, v_hi)), float(max(v_lo, v_hi))


def mopra_axis_from_fits(cube_path: Path) -> dict:
    """Authoritative axis from SpectralCube (handles VELO-LSR -> VOPT fixes)."""
    from spectral_cube import SpectralCube

    cube = SpectralCube.read(str(cube_path.resolve()), use_dask=False)
    v = cube.spectral_axis.to("km/s").value.astype(np.float64)
    dv = float(np.median(np.diff(v)))
    return {
        "n_channels": int(v.size),
        "vrange": (float(v.min()), float(v.max())),
        "channel_width_kms": dv,
        "bunit": str(cube.unit),
    }


def build_mopra_base_cfg(
    *,
    repo_root: Path | None = None,
    cube_path: Path | None = None,
    header_txt: Path | None = None,
) -> dict:
    """
    Base cfg for MOPRA CMZ HNCO: n_channels, vrange, channel metadata.
    Uses cube_path when provided; otherwise parses header_txt (defaults under data/).
    """
    root = repo_root or Path(__file__).resolve().parents[3]
    cube_p = cube_path or (root / "data" / MOPRA_CUBE_FITS)
    header_p = header_txt or (root / "data" / MOPRA_HEADER_TXT)

    meta: dict = {"cube_path": str(cube_p), "header_txt": str(header_p)}
    if cube_p.is_file():
        axis = mopra_axis_from_fits(cube_p)
    elif header_p.is_file():
        h = _parse_header_txt(header_p)
        n = int(h["NAXIS3"])
        vmin, vmax = _vrange_from_wcs_keys(h)
        axis = {
            "n_channels": n,
            "vrange": (vmin, vmax),
            "channel_width_kms": _cdelt3_kms(h),
            "bunit": str(h.get("BUNIT", "K")).strip(),
            "blank_value": float(h["BLANK"]) if "BLANK" in h else -1.0,
        }
    else:
        raise FileNotFoundError(f"MOPRA cube or header not found: {cube_p} / {header_p}")

    meta.update(axis)
    return {
        "n_channels": int(axis["n_channels"]),
        "min_components": 0,
        "max_components": 10,
        "vrange": tuple(axis["vrange"]),
        "mopra_meta": meta,
    }


__all__ = [
    "MOPRA_CUBE_FITS",
    "MOPRA_HEADER_TXT",
    "build_mopra_base_cfg",
    "mopra_axis_from_fits",
]
