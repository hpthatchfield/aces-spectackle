### Helpers to align cutout maps with a reference subcube via WCS.
from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

_HEADER_CUBE = "NLWCUBE"
_HEADER_SUBCUBE = "NLWSUBCB"
_HEADER_Y0 = "NLWY0"
_HEADER_Y1 = "NLWY1"
_HEADER_X0 = "NLWX0"
_HEADER_X1 = "NLWX1"


def spatial_bounds_in_mosaic(
    mosaic_wcs: WCS,
    subcube_wcs: WCS,
    *,
    sub_nx: int,
    sub_ny: int,
) -> tuple[int, int, int, int]:
    """
    Map a subcube's sky footprint to 0-based [y0, y1) x [x0, x1) indices in the mosaic.

    Uses subcube corner pixels (inclusive) snapped outward to integer mosaic pixels.
    Returns (y0, y1, x0, x1) for numpy / spectral-cube slicing: cube[:, y0:y1, x0:x1].
    """
    if sub_nx <= 0 or sub_ny <= 0:
        raise ValueError(f"Subcube must have positive spatial shape, got ({sub_ny}, {sub_nx}).")

    xs = np.array([0.0, sub_nx - 1.0, 0.0, sub_nx - 1.0], dtype=np.float64)
    ys = np.array([0.0, 0.0, sub_ny - 1.0, sub_ny - 1.0], dtype=np.float64)
    lon, lat = subcube_wcs.all_pix2world(xs, ys, 0)
    mx, my = mosaic_wcs.all_world2pix(lon, lat, 0)

    x0 = int(np.floor(mx.min()))
    x1 = int(np.ceil(mx.max())) + 1
    y0 = int(np.floor(my.min()))
    y1 = int(np.ceil(my.max())) + 1
    return y0, y1, x0, x1


def bounds_from_subcube_ref(mosaic_path: Path, subcube_path: Path) -> tuple[int, int, int, int]:
    """Read FITS headers and return mosaic slice bounds covering the subcube footprint."""
    mosaic_path = mosaic_path.resolve()
    subcube_path = subcube_path.resolve()
    m_hdr = fits.getheader(mosaic_path)
    s_hdr = fits.getheader(subcube_path)
    sub_nx = int(s_hdr["NAXIS1"])
    sub_ny = int(s_hdr["NAXIS2"])
    m_wcs = WCS(m_hdr).celestial
    s_wcs = WCS(s_hdr).celestial
    y0, y1, x0, x1 = spatial_bounds_in_mosaic(m_wcs, s_wcs, sub_nx=sub_nx, sub_ny=sub_ny)
    ny_m = int(m_hdr["NAXIS2"])
    nx_m = int(m_hdr["NAXIS1"])
    y0 = max(0, y0)
    x0 = max(0, x0)
    y1 = min(ny_m, y1)
    x1 = min(nx_m, x1)
    if y1 <= y0 or x1 <= x0:
        raise ValueError(
            f"Subcube {subcube_path.name} footprint does not overlap mosaic {mosaic_path.name} "
            f"(computed y=[{y0},{y1}) x=[{x0},{x1}))."
        )
    return y0, y1, x0, x1


def resolve_spatial_bounds(
    *,
    mosaic_ny: int,
    mosaic_nx: int,
    y0: int | None,
    y1: int | None,
    x0: int | None,
    x1: int | None,
    subcube_ref: Path | None,
    mosaic_path: Path,
) -> tuple[int, int, int, int]:
    """
    Choose cutout bounds: explicit CLI wins; else derive from --subcube-ref; else full mosaic.
    """
    manual = [y0, y1, x0, x1]
    if any(v is not None for v in manual):
        if subcube_ref is not None and any(v is not None for v in manual):
            print(
                "NOTE: using explicit --y0/--y1/--x0/--x1; ignoring --subcube-ref for bounds.",
                flush=True,
            )
        y0_i = 0 if y0 is None else int(y0)
        y1_i = mosaic_ny if y1 is None else int(y1)
        x0_i = 0 if x0 is None else int(x0)
        x1_i = mosaic_nx if x1 is None else int(x1)
    elif subcube_ref is not None:
        y0_i, y1_i, x0_i, x1_i = bounds_from_subcube_ref(mosaic_path, subcube_ref)
        print(
            f"Cutout from subcube WCS ({subcube_ref.name}): "
            f"y=[{y0_i},{y1_i}) x=[{x0_i},{x1_i}) -> {y1_i - y0_i}x{x1_i - x0_i}",
            flush=True,
        )
    else:
        y0_i, y1_i, x0_i, x1_i = 0, mosaic_ny, 0, mosaic_nx

    y0_i, y1_i = max(0, y0_i), min(mosaic_ny, y1_i)
    x0_i, x1_i = max(0, x0_i), min(mosaic_nx, x1_i)
    if y1_i <= y0_i or x1_i <= x0_i:
        raise ValueError(f"Empty cutout: y=[{y0_i},{y1_i}) x=[{x0_i},{x1_i})")
    return y0_i, y1_i, x0_i, x1_i


def write_prob_map_provenance(
    header,
    *,
    y0: int,
    y1: int,
    x0: int,
    x1: int,
    cube_path: Path,
    subcube_ref: Path | None = None,
) -> None:
    header[_HEADER_Y0] = int(y0)
    header[_HEADER_Y1] = int(y1)
    header[_HEADER_X0] = int(x0)
    header[_HEADER_X1] = int(x1)
    header[_HEADER_CUBE] = str(cube_path.resolve())[:68]
    if subcube_ref is not None:
        header[_HEADER_SUBCUBE] = str(subcube_ref.resolve())[:68]
    header["COMMENT"] = "NLW map provenance: NLWY0/X0 = 0-based mosaic origin (inclusive)"


def read_prob_map_offsets(header) -> tuple[int, int, int, int, Path | None]:
    """
    Return (y0, y1, x0, x1, cube_path) from a probability-map FITS header.
    Raises KeyError if provenance keys are missing.
    """
    y0 = int(header[_HEADER_Y0])
    y1 = int(header[_HEADER_Y1])
    x0 = int(header[_HEADER_X0])
    x1 = int(header[_HEADER_X1])
    cube_raw = header.get(_HEADER_CUBE)
    cube_path = Path(cube_raw) if cube_raw else None
    return y0, y1, x0, x1, cube_path


def repair_prob_map_wcs(
    prob_map_path: Path,
    mosaic_path: Path,
    *,
    out: Path | None = None,
) -> Path:
    """
    Rewrite spatial WCS on an NLW prob map using NLWY0/X0 provenance.

    For maps written before the [y0:y1, x0:x1] slice fix (bad CRPIX / wrong sky).
    Pixel values are unchanged; only the header WCS is updated.
    """
    from spectackle.wcs_plot import merge_wcs_header, wcs_celestial, wcs_header_for_array_cutout

    prob_map_path = prob_map_path.resolve()
    mosaic_path = mosaic_path.resolve()
    data = fits.getdata(prob_map_path)
    base = fits.getheader(prob_map_path)
    y0, y1, x0, x1, _ = read_prob_map_offsets(base)
    w_parent = wcs_celestial(fits.getheader(mosaic_path))
    wcs_hdr = wcs_header_for_array_cutout(w_parent, y0=y0, y1=y1, x0=x0, x1=x1)
    new_hdr = merge_wcs_header(base, wcs_header=wcs_hdr)
    out_path = prob_map_path if out is None else Path(out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fits.PrimaryHDU(data=data, header=new_hdr).writeto(str(out_path), overwrite=True)
    return out_path


def resolve_spectrum_offsets(
    *,
    prob_header,
    cube_path: Path,
    y0: int | None,
    x0: int | None,
    subcube_ref: Path | None,
    default_mosaic: Path,
) -> tuple[int, int, Path]:
    """
    Pick (y0, x0, cube_path) for mapping prob-map pixels to mosaic spectra.

    Priority: explicit CLI > FITS provenance > --subcube-ref WCS > (0, 0).
    """
    if y0 is not None or x0 is not None:
        if subcube_ref is not None:
            print("NOTE: using explicit --y0/--x0; ignoring --subcube-ref for offsets.", flush=True)
        return int(y0 or 0), int(x0 or 0), cube_path.resolve()

    try:
        hy0, _, hx0, _, hcube = read_prob_map_offsets(prob_header)
        out_cube = hcube if hcube is not None and hcube.exists() else cube_path.resolve()
        print(
            f"Offsets from prob-map header: y0={hy0} x0={hx0}  cube={out_cube.name}",
            flush=True,
        )
        return hy0, hx0, out_cube
    except KeyError:
        pass

    if subcube_ref is not None:
        hy0, _, hx0, _ = bounds_from_subcube_ref(default_mosaic.resolve(), subcube_ref.resolve())
        print(
            f"Offsets from subcube WCS ({subcube_ref.name}): y0={hy0} x0={hx0}",
            flush=True,
        )
        return hy0, hx0, default_mosaic.resolve()

    return 0, 0, cube_path.resolve()
