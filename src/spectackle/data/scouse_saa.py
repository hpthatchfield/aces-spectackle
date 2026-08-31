### SCouse-style spectral averaging areas (SAA) - grid geometry aligned with scousepy stage 1.
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any
import warnings

import numpy as np

### CMZ HNCO defaults from Henshaw+2016 / Jones CMZ mosaic (see experiments/MOPRA_Count).
CMZ_R_SAA_DEG = 0.05  ### square SAA side length in deg (Henshaw/Jones CMZ)
CMZ_ITOL_SIGMA = 3.0  ### moment mask: retain emission above this x sigma_rms
CMZ_FILLFACTOR = 0.5  ### min fraction of unmasked pixels inside each SAA square
CMZ_SCOUSE_SPACING = "nyquist"  ### centre spacing = wsaa / 2
CMZ_SCOUSE_COVMETHOD = "regular"

### Stage-3 tolerances (stored for ScousePy handoff; not used in grid setup).
### Order matches scousepy.config tol = [dK, SNR, v_res_mult, dV_width, dV_cent, sep_frac].
CMZ_SCOUSE_TOL = [2.0, 3.0, 1.0, 4.0, 1.0, 0.5]


@dataclass(frozen=True)
class ScouseSaaConfig:
    """Coverage parameters for one SAA size (single refinement level)."""

    r_saa_deg: float = CMZ_R_SAA_DEG
    ### If False (CMZ default), r_saa_deg is the square side; wsaa = width / d-theta.
    r_is_radius: bool = False
    wsaa_pix: int | None = None  ### override computed width (pixels)
    itol_sigma: float = CMZ_ITOL_SIGMA
    fillfactor: float = CMZ_FILLFACTOR
    spacing: str = CMZ_SCOUSE_SPACING
    covmethod: str = CMZ_SCOUSE_COVMETHOD
    tol: tuple[float, ...] = field(default_factory=lambda: tuple(CMZ_SCOUSE_TOL))
    mask_below: float | None = None  ### absolute K; default itol_sigma x cube_rms
    xmin: int = 0
    ymin: int = 0
    xmax: int | None = None
    ymax: int | None = None
    vel_min: float | None = None
    vel_max: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def pixel_scale_deg(wcs_celestial) -> tuple[float, float]:
    cdelt = wcs_celestial.wcs.cdelt
    if cdelt is None or len(cdelt) < 2:
        raise ValueError("WCS missing CDELT for celestial axes.")
    return abs(float(cdelt[0])), abs(float(cdelt[1]))


def wsaa_from_r_saa(
    r_saa_deg: float,
    pixel_scale_x_deg: float,
    *,
    r_is_radius: bool = True,
) -> int:
    """
  Convert angular SAA size to scousepy wsaa (square width in pixels).

  SCouse uses a square macropixel of side wsaa. For R_SAA defined as a radius
  (CMZ default), we set wsaa ~ 2 x R / d-theta so the square spans the diameter.
  """
    if pixel_scale_x_deg <= 0:
        raise ValueError(f"Non-positive pixel scale: {pixel_scale_x_deg}")
    width_deg = 2.0 * r_saa_deg if r_is_radius else r_saa_deg
    wsaa = int(round(width_deg / pixel_scale_x_deg))
    return max(3, wsaa)


def spacing_from_wsaa(wsaa: int, *, spacing: str = CMZ_SCOUSE_SPACING) -> float:
    ### scousepy: nyquist -> centre spacing = wsaa / 2
    if spacing == "nyquist":
        return float(wsaa) / 2.0
    if spacing == "regular":
        return float(wsaa)
    raise ValueError(f"Unknown spacing {spacing!r}; use 'nyquist' or 'regular'.")


def estimate_spectrum_rms(spec: np.ndarray) -> float:
    """
    scousepy.stage_1.calc_rms logic - robust sigma from negative / low channels.
    spec: 1D float array (K).
    """
    from astropy.stats import median_absolute_deviation

    spec = np.asarray(spec, dtype=np.float64)
    finite = np.isfinite(spec)
    if not np.any(finite):
        return np.nan
    s = spec[finite]
    negative = s < 0.0
    reflected = np.concatenate((s[negative], np.abs(s[negative])))
    if reflected.size == 0:
        mad = median_absolute_deviation(s)
        noise = s
    else:
        mad = median_absolute_deviation(reflected)
        if negative.sum() < 0.47 * s.size:
            cap = 3.5 * mad
        else:
            cap = 4.0 * mad
        noise = s[s < abs(cap)]
    if noise.size == 0:
        return float(np.std(s))
    return float(np.sqrt(np.mean(noise**2)))


def estimate_cube_rms(
    cube_data_vyx: np.ndarray,
    *,
    sample_pixels: int = 512,
    seed: int = 0,
) -> float:
    """Median per-spectrum RMS over a random pixel sample (v,y,x)."""
    nv, ny, nx = cube_data_vyx.shape
    n_pix = ny * nx
    rng = np.random.default_rng(seed)
    idx = rng.choice(n_pix, size=min(sample_pixels, n_pix), replace=False)
    rms_vals = []
    for flat in idx:
        y, x = divmod(int(flat), nx)
        r = estimate_spectrum_rms(cube_data_vyx[:, y, x])
        if np.isfinite(r) and r > 0:
            rms_vals.append(r)
    if not rms_vals:
        return float(np.nanstd(cube_data_vyx))
    return float(np.median(rms_vals))


def get_coverage_grid(shape_yx: tuple[int, int], spacing: float) -> np.ndarray:
    """
    scousepy.scousecoverage.get_coverage - SAA centre coordinates (x, y, include).

  Returns (N, 3) with columns [x_center, y_center, to_fit_bool].
  """
    ny, nx = int(shape_yx[0]), int(shape_yx[1])
    y = np.arange(ny)
    x = np.arange(nx)
    rangex = [np.min(x), np.max(x)]
    sizex = abs(rangex[0] - rangex[1])
    rangey = [np.min(y), np.max(y)]
    sizey = abs(rangey[0] - rangey[1])
    nposx = int(sizex / spacing + 1.0)
    nposy = int(sizey / spacing + 1.0)
    cov_x = np.max(rangex) - spacing * np.arange(nposx)
    cov_y = np.min(rangey) + spacing * np.arange(nposy)
    cov_xx, cov_yy = np.meshgrid(cov_x, cov_y)
    cov_xx = np.flip(cov_xx, axis=1)
    include = np.zeros(cov_xx.size, dtype=bool)
    return np.column_stack((cov_xx.ravel(), cov_yy.ravel(), include))


def _mask_square(
    mask_yx: np.ndarray,
    centre_yx: tuple[float, float],
    width: float,
) -> np.ndarray:
    """Boolean square mask centred on (y, x) with side width (scousepy mask_img)."""
    ny, nx = mask_yx.shape
    cy, cx = int(centre_yx[0]), int(centre_yx[1])
    w = int(width)
    half = w // 2
    yn = max(cy - half, 0)
    yp = min(cy + half + (w % 2), ny)
    xn = max(cx - half, 0)
    xp = min(cx + half + (w % 2), nx)
    out = np.zeros((ny, nx), dtype=bool)
    out[yn:yp, xn:xp] = True
    return out


def filter_coverage_by_moment_mask(
    coverage: np.ndarray,
    moment_mask_yx: np.ndarray,
    wsaa: int,
    fillfactor: float,
) -> np.ndarray:
    """
    scousepy check_against_mask - keep SAAs with enough significant pixels.
    Modifies coverage[:, 2] in place.
    """
    cov = coverage.copy()
    for i in range(cov.shape[0]):
        cx, cy = cov[i, 0], cov[i, 1]
        local = _mask_square(np.ones_like(moment_mask_yx, dtype=bool), (cy, cx), wsaa)
        maxpix = int(local.sum())
        sigpix = int((local & moment_mask_yx).sum())
        frac = 0.0 if maxpix == 0 else float(sigpix) / float(maxpix)
        cov[i, 2] = frac >= float(fillfactor)
    return cov


def moment0_mask_from_cube(
    cube_vyx: np.ndarray,
    mask_below: float,
    *,
    vel_slice: slice | None = None,
    blank_value: float = -1.0,
) -> np.ndarray:
    """
    Boolean (y,x) mask matching scousepy stage-1 logic:

    1. Per-channel mask: T > mask_below (I_tol x sigma_rms in K).
    2. Moment-0 on masked cube over vel_slice (or full axis).
    3. Pixel is valid where moment-0 is finite (>=1 channel passed mask).
    """
    data = np.asarray(cube_vyx, dtype=np.float64)
    if vel_slice is not None:
        data = data[vel_slice]
    data = np.where(data == blank_value, np.nan, data)
    chan_ok = data > float(mask_below)
    masked = np.where(chan_ok, data, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        with np.errstate(all="ignore"):
            mom0 = np.nanmean(masked, axis=0)
    return np.isfinite(mom0)


def saa_pixel_mask(
    centre_x: float,
    centre_y: float,
    wsaa: int,
    shape_yx: tuple[int, int],
) -> np.ndarray:
    """Square SAA footprint (y,x) matching scousepy generate_saamask rectangle."""
    ny, nx = shape_yx
    bl_x = centre_x - wsaa / 2.0
    bl_y = centre_y - wsaa / 2.0
    yy, xx = np.mgrid[0:ny, 0:nx]
    inside = (
        (xx >= bl_x)
        & (xx < bl_x + wsaa)
        & (yy >= bl_y)
        & (yy < bl_y + wsaa)
    )
    return inside


def average_spectrum_in_mask(
    cube_vyx: np.ndarray,
    pixel_mask_yx: np.ndarray,
    moment_mask_yx: np.ndarray,
    *,
    blank_value: float = -1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Mean spectrum over pixels in SAA  intersect  moment mask.
    Returns (spec_mean, (y_indices, x_indices) for included pixels).
    """
    use = pixel_mask_yx & moment_mask_yx
    ys, xs = np.where(use)
    if ys.size == 0:
        return np.full(cube_vyx.shape[0], np.nan, dtype=np.float32), (ys, xs)
    sub = cube_vyx[:, ys, xs].astype(np.float64)
    sub = np.where(sub == blank_value, np.nan, sub)
    spec = np.nanmean(sub, axis=1).astype(np.float32)
    return spec, (ys, xs)


def build_saa_grid(
    cube_vyx: np.ndarray,
    cfg: ScouseSaaConfig,
    *,
    pixel_scale_x_deg: float,
    cube_rms: float | None = None,
    blank_value: float = -1.0,
) -> dict[str, Any]:
    """
    Full stage-1 style SAA grid for one wsaa level.

    Returns dict with coverage, wsaa, spacing, mask_below, saa_list, meta.
    """
    nv, ny, nx = cube_vyx.shape
    ymax = ny if cfg.ymax is None else int(cfg.ymax)
    xmax = nx if cfg.xmax is None else int(cfg.xmax)
    ymin, xmin = int(cfg.ymin), int(cfg.xmin)
    sub_shape = (ymax - ymin, xmax - xmin)
    sub_cube = cube_vyx[:, ymin:ymax, xmin:xmax]

    wsaa = (
        int(cfg.wsaa_pix)
        if cfg.wsaa_pix is not None
        else wsaa_from_r_saa(cfg.r_saa_deg, pixel_scale_x_deg, r_is_radius=cfg.r_is_radius)
    )
    spacing = spacing_from_wsaa(wsaa, spacing=cfg.spacing)

    if cube_rms is None:
        cube_rms = estimate_cube_rms(sub_cube)
    mask_below = float(cfg.mask_below) if cfg.mask_below is not None else float(cfg.itol_sigma) * cube_rms

    vel_sl = None
    if cfg.vel_min is not None or cfg.vel_max is not None:
        ### Caller should pass vel_slice via trimmed cube; placeholder for API symmetry.
        pass
    moment_mask = moment0_mask_from_cube(sub_cube, mask_below, blank_value=blank_value)

    coverage = get_coverage_grid(sub_shape, spacing)
    coverage = filter_coverage_by_moment_mask(coverage, moment_mask, wsaa, cfg.fillfactor)

    saa_list: list[dict[str, Any]] = []
    saa_id = 0
    for row in coverage:
        cx, cy, keep = float(row[0]), float(row[1]), bool(row[2])
        if not keep:
            continue
        pix_mask = saa_pixel_mask(cx, cy, wsaa, sub_shape)
        spec, (ys, xs) = average_spectrum_in_mask(
            sub_cube, pix_mask, moment_mask, blank_value=blank_value
        )
        saa_list.append(
            {
                "saa_id": saa_id,
                "center_x": cx + xmin,
                "center_y": cy + ymin,
                "center_x_trim": cx,
                "center_y_trim": cy,
                "wsaa": wsaa,
                "n_pixels": int(ys.size),
                "to_fit": True,
                "spectrum": spec,
                "pixel_y": (ys + ymin).astype(np.int32),
                "pixel_x": (xs + xmin).astype(np.int32),
            }
        )
        saa_id += 1

    return {
        "cfg": cfg.to_dict(),
        "wsaa": wsaa,
        "spacing": spacing,
        "mask_below": mask_below,
        "cube_rms": cube_rms,
        "n_saa_total": int(coverage.shape[0]),
        "n_saa_kept": len(saa_list),
        "coverage": coverage,
        "moment_mask": moment_mask,
        "saa_list": saa_list,
        "trim": {"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax},
    }


__all__ = [
    "CMZ_FILLFACTOR",
    "CMZ_ITOL_SIGMA",
    "CMZ_R_SAA_DEG",
    "CMZ_SCOUSE_TOL",
    "ScouseSaaConfig",
    "build_saa_grid",
    "estimate_cube_rms",
    "get_coverage_grid",
    "moment0_mask_from_cube",
    "spacing_from_wsaa",
    "wsaa_from_r_saa",
]
