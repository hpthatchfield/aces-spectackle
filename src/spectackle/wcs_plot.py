### Shared WCSAxes styling for Galactic sky maps (l, b in decimal deg, l wrapped at +/-180).
from __future__ import annotations

import contextlib
import warnings

import numpy as np
from astropy import units as u
from astropy.io.fits import Header
from astropy.wcs import WCS


def wcs_celestial(header) -> WCS:
    ### Drop spectral/stokes axes; use the 2D celestial slice only.
    return WCS(header).celestial


def wcs_header_for_array_cutout(
    wcs: WCS,
    *,
    y0: int,
    y1: int,
    x0: int,
    x1: int,
) -> Header:
    """
    FITS header for a 2D array cut as parent[y0:y1, x0:x1] (numpy row=y, col=x).

    Astropy celestial WCS slice order is **[y0:y1, x0:x1]** - matches numpy.
    Using [x0:x1, y0:y1] swaps axes and yields bad CRPIX (often negative / off-image).
    """
    ny = int(y1 - y0)
    nx = int(x1 - x0)
    if ny <= 0 or nx <= 0:
        raise ValueError(f"Empty cutout: y=[{y0},{y1}) x=[{x0},{x1})")
    w_cut = wcs[y0:y1, x0:x1].celestial
    ### Standalone cutout FITS: put CRPIX near the image center (slice keeps mosaic-scale CRPIX).
    ix, iy = (nx - 1) / 2.0, (ny - 1) / 2.0
    center_world = w_cut.pixel_to_world(ix, iy)
    w_out = w_cut.deepcopy()
    w_out.wcs.crval = [float(center_world.l.deg), float(center_world.b.deg)]
    w_out.wcs.crpix = [ix + 1.0, iy + 1.0]
    _validate_cutout_wcs(w_out, nx=nx, ny=ny)
    hdr = w_out.to_header()
    hdr["NAXIS"] = 2
    hdr["NAXIS1"] = nx
    hdr["NAXIS2"] = ny
    return hdr


def _validate_cutout_wcs(w_cut: WCS, *, nx: int, ny: int) -> None:
    ### Sanity check: CRPIX should sit near the cutout, not far outside it.
    crpix = np.asarray(w_cut.wcs.crpix, dtype=np.float64)
    if crpix.shape[0] < 2:
        return
    cx, cy = float(crpix[0]), float(crpix[1])
    if not (-0.5 * nx <= cx <= 1.5 * nx and -0.5 * ny <= cy <= 1.5 * ny):
        raise ValueError(
            f"Cutout WCS CRPIX=({cx:.1f}, {cy:.1f}) looks wrong for shape ({ny}, {nx}); "
            "use wcs[y0:y1, x0:x1], not wcs[x0:x1, y0:y1]."
        )


def merge_wcs_header(
    base: Header,
    *,
    wcs_header: Header,
    preserve_keys: tuple[str, ...] = (),
) -> Header:
    ### Replace celestial WCS keys but keep provenance / BUNIT / COMMENT extras.
    out = Header(wcs_header)
    for key in preserve_keys:
        if key in base:
            out[key] = base[key]
    for key, val in base.items():
        if key in ("COMMENT", "HISTORY"):
            continue
        if key in out:
            continue
        if key.startswith(("CTYPE", "CRVAL", "CRPIX", "CDELT", "CROTA", "PC", "CD", "PV")):
            continue
        if key in ("NAXIS", "NAXIS1", "NAXIS2"):
            continue
        try:
            out[key] = val
        except Exception:
            pass
    return out


def _world_extent_deg(wcs: WCS, shape_yx: tuple[int, int]) -> tuple[float, float, float, float]:
    ny, nx = shape_yx
    world = wcs.pixel_to_world([0, nx - 1, 0, nx - 1], [0, 0, ny - 1, ny - 1])
    l = np.asarray(world.l.deg, dtype=np.float64)
    b = np.asarray(world.b.deg, dtype=np.float64)
    return float(l.min()), float(l.max()), float(b.min()), float(b.max())


def _lon_span_deg(l_deg: np.ndarray) -> float:
    ### Short arc on the circle (handles l~0/360 crossings in CMZ cutouts).
    lmin, lmax = float(np.min(l_deg)), float(np.max(l_deg))
    raw = lmax - lmin
    if raw > 180.0:
        return 360.0 - raw
    return raw


def _field_spans_deg(wcs: WCS, shape_yx: tuple[int, int]) -> tuple[float, float]:
    ny, nx = shape_yx
    world = wcs.pixel_to_world([0, nx - 1, 0, nx - 1], [0, 0, ny - 1, ny - 1])
    l = np.asarray(world.l.deg, dtype=np.float64)
    b = np.asarray(world.b.deg, dtype=np.float64)
    return _lon_span_deg(l), float(b.max() - b.min())


def _nice_tick_step(span_deg: float, n_target: int = 5) -> float:
    ### Pick a 1/2/5 * 10^n step so the field shows ~n_target major ticks.
    if span_deg <= 0:
        return 0.1
    raw = span_deg / n_target
    exp = int(np.floor(np.log10(max(raw, 1e-12))))
    base = 10.0**exp
    for mult in (1.0, 2.0, 5.0, 10.0):
        step = mult * base
        if span_deg / step <= n_target + 1:
            return float(step)
    return float(raw)


def _degree_formatter_for_step(step_deg: float) -> str:
    ### Match tick label precision to spacing (d.d cannot label 0.02 deg steps).
    if step_deg >= 0.1:
        return "d.d"
    if step_deg >= 0.01:
        return "d.dd"
    return "d.ddd"


def style_galactic_wcs_axes(
    ax,
    *,
    wcs: WCS | None = None,
    shape_yx: tuple[int, int] | None = None,
    n_major: int = 5,
    lon_spacing_deg: float | None = None,
    lat_spacing_deg: float | None = None,
    lon_minpad: float = 0.5,
    lat_minpad: float = 0.5,
    minor_frequency: int = 4,
    grid: bool = True,
    grid_color: str = "white",
    grid_alpha: float = 0.45,
) -> None:
    """
    Galactic l,b ticks: wrap longitude at +/-180 deg, decimal degrees (negative l OK).
    Use ax.coords labels - avoid ax.set_xlabel/set_ylabel on WCSAxes (retriggers ticks).

    When wcs + shape_yx are given, major-tick spacing is chosen from the field extent
    (fixed 0.2 deg/0.1 deg is too coarse for small ACES cutouts).
    """
    ax.coords[0].set_axislabel(r"Galactic longitude, $l$ (deg)", minpad=lon_minpad)
    ax.coords[1].set_axislabel(r"Galactic latitude, $b$ (deg)", minpad=lat_minpad)
    lon = ax.coords[0]
    lat = ax.coords[1]
    lon.set_coord_type("longitude", coord_wrap=180 * u.deg)

    if lon_spacing_deg is None or lat_spacing_deg is None:
        if wcs is not None and shape_yx is not None:
            l_span, b_span = _field_spans_deg(wcs, shape_yx)
            if lon_spacing_deg is None:
                lon_spacing_deg = _nice_tick_step(l_span, n_major)
            if lat_spacing_deg is None:
                lat_spacing_deg = _nice_tick_step(b_span, n_major)
        else:
            lon.set_ticks(number=n_major)
            lat.set_ticks(number=n_major)

    if lon_spacing_deg is not None:
        lon.set_ticks(spacing=float(lon_spacing_deg) * u.deg)
    if lat_spacing_deg is not None:
        lat.set_ticks(spacing=float(lat_spacing_deg) * u.deg)

    lon_fmt = _degree_formatter_for_step(float(lon_spacing_deg or 0.1))
    lat_fmt = _degree_formatter_for_step(float(lat_spacing_deg or 0.1))
    lon.set_major_formatter(lon_fmt)
    lon.set_format_unit(u.deg, decimal=True, show_decimal_unit=False)
    lon.set_ticklabel(exclude_overlapping=False)
    lat.set_major_formatter(lat_fmt)
    lat.set_format_unit(u.deg, decimal=True, show_decimal_unit=False)
    lat.set_ticklabel(exclude_overlapping=False)
    lon.display_minor_ticks(True)
    lat.display_minor_ticks(True)
    lon.set_minor_frequency(int(minor_frequency))
    lat.set_minor_frequency(int(minor_frequency))
    lon.set_ticks_position("b")
    lat.set_ticks_position("l")
    if grid:
        ax.grid(color=grid_color, ls=":", lw=0.5, alpha=grid_alpha)


@contextlib.contextmanager
def suppress_wcsaxes_format_warnings():
    ### WCSAxes may try to format NaN overlay ticks on SIN cutouts (harmless).
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="invalid value encountered in do_format",
            category=RuntimeWarning,
        )
        yield
