### ScousePy-style multi-Gaussian fit handoff from ML K_pred.
###
### Uses scipy (not pyspeckit/scousepy) so the MVP runs in the existing env.
### Output columns match ScousePy ascii with (l,b) instead of (x,y) pixels,
### same layout as Henshaw final_fits_updated.dat:
###   ncomps, l, b, amp, amp_err, v, v_err, sigma, sigma_err,
###   rms, resid_std, chi2, dof, redchi, aic
### Width is Gaussian dispersion sigma (km/s), as in ScousePy (not FWHM).
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.signal import find_peaks

from .mopra_preprocess import MOPRA_BLANK_VALUE, valid_mask_mopra
from .scouse_saa import estimate_spectrum_rms

_FWHM_TO_SIGMA = 2.0 * np.sqrt(2.0 * np.log(2.0))  ### ~ 2.355

### Column names for the 15-column handoff table (Henshaw / ScousePy style).
HANDOFF_COLUMNS = (
    "ncomps",
    "l",
    "b",
    "amp",
    "amp_err",
    "v",
    "v_err",
    "sigma",
    "sigma_err",
    "rms",
    "resid_std",
    "chi2",
    "dof",
    "redchi",
    "aic",
)


@dataclass(frozen=True)
class FitResult:
    """One pixel's multi-Gaussian fit (or empty if K=0 / failed)."""

    k: int
    amps: np.ndarray
    vs: np.ndarray
    sigmas: np.ndarray
    amp_errs: np.ndarray
    v_errs: np.ndarray
    sigma_errs: np.ndarray
    rms: float
    resid_std: float
    chi2: float
    dof: float
    redchi: float
    aic: float
    success: bool
    message: str = ""


def _empty_fit(*, rms: float = np.nan, message: str = "") -> FitResult:
    z = np.zeros(0, dtype=np.float64)
    return FitResult(
        k=0,
        amps=z,
        vs=z,
        sigmas=z,
        amp_errs=z,
        v_errs=z,
        sigma_errs=z,
        rms=float(rms),
        resid_std=float("nan"),
        chi2=float("nan"),
        dof=float("nan"),
        redchi=float("nan"),
        aic=float("nan"),
        success=True,
        message=message,
    )


def _gaussian_sum(v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """params: [amp0, v0, sig0, amp1, v1, sig1, ...]."""
    k = params.size // 3
    y = np.zeros_like(v, dtype=np.float64)
    for i in range(k):
        amp, mu, sig = params[3 * i : 3 * i + 3]
        if sig <= 0.0:
            continue
        y += amp * np.exp(-0.5 * ((v - mu) / sig) ** 2)
    return y


def seed_components(
    spec: np.ndarray,
    vel: np.ndarray,
    k: int,
    *,
    rms: float,
    min_sep_kms: float = 4.0,
    snr_floor: float = 3.0,
) -> np.ndarray:
    """
    Initial guesses: top-K prominence peaks; pad by splitting the strongest if needed.

    Returns flat params [amp, v, sigma] * k.
    """
    k = int(k)
    if k <= 0:
        return np.zeros(0, dtype=np.float64)

    y = np.asarray(spec, dtype=np.float64) - float(np.median(spec))
    dv = float(np.median(np.diff(vel))) if vel.size > 1 else 2.0
    min_dist = max(1, int(round(min_sep_kms / max(abs(dv), 1e-6))))
    height = max(snr_floor * float(rms), 1e-6)
    peaks, props = find_peaks(y, height=height, prominence=height, distance=min_dist)
    if peaks.size == 0:
        ### Fallback: single bump at global max.
        i0 = int(np.argmax(y))
        amp0 = max(float(y[i0]), height)
        params = np.array([amp0, float(vel[i0]), max(abs(dv) * 2.0, 2.0)], dtype=np.float64)
        peaks = np.array([i0])
        prom = np.array([amp0])
    else:
        prom = np.asarray(props["prominences"], dtype=np.float64)
        order = np.argsort(-prom)
        peaks = peaks[order]
        prom = prom[order]
        params_list = []
        for i, p in enumerate(peaks[:k]):
            amp = max(float(y[p]), height)
            ### Rough sigma from prominence width if available, else 2 channels.
            sig = max(abs(dv) * 2.0, 2.0)
            params_list.extend([amp, float(vel[p]), sig])
        params = np.asarray(params_list, dtype=np.float64)

    ### Pad to K by splitting the brightest component in velocity.
    while params.size // 3 < k:
        amps = params[0::3]
        i_max = int(np.argmax(amps))
        amp, mu, sig = params[3 * i_max : 3 * i_max + 3]
        split = max(sig * 0.75, abs(dv))
        params[3 * i_max : 3 * i_max + 3] = [amp * 0.6, mu - split, sig]
        params = np.concatenate(
            [params, np.array([amp * 0.6, mu + split, sig], dtype=np.float64)]
        )
    return params[: 3 * k]


def fit_spectrum_gaussians(
    spec: np.ndarray,
    vel: np.ndarray,
    k: int,
    *,
    blank_value: float = MOPRA_BLANK_VALUE,
    vel_range: tuple[float, float] | None = (40.0, 140.0),
    min_sep_kms: float = 4.0,
    snr_floor: float = 3.0,
    sigma_min_kms: float = 1.0,
    sigma_max_kms: float = 80.0,
) -> FitResult:
    """
    Fit K Gaussians to a 1D spectrum in an optional velocity window.

    K comes from the ML count model (rounded). Returns empty fit for K<=0.
    """
    spec = np.asarray(spec, dtype=np.float64).reshape(-1)
    vel = np.asarray(vel, dtype=np.float64).reshape(-1)
    if spec.size != vel.size:
        raise ValueError(f"spec length {spec.size} != vel length {vel.size}")

    valid = valid_mask_mopra(spec, blank_value=blank_value)
    if vel_range is not None:
        vlo, vhi = float(vel_range[0]), float(vel_range[1])
        valid &= (vel >= vlo) & (vel <= vhi)
    if valid.sum() < 8:
        return _empty_fit(message="too_few_channels")

    v = vel[valid]
    y = spec[valid]
    med = float(np.median(y))
    y0 = y - med
    rms = float(estimate_spectrum_rms(y0))
    if not np.isfinite(rms) or rms <= 0.0:
        rms = float(np.std(y0) + 1e-12)

    k = int(k)
    if k <= 0:
        resid = y0
        resid_std = float(np.std(resid))
        return FitResult(
            k=0,
            amps=np.zeros(0),
            vs=np.zeros(0),
            sigmas=np.zeros(0),
            amp_errs=np.zeros(0),
            v_errs=np.zeros(0),
            sigma_errs=np.zeros(0),
            rms=rms,
            resid_std=resid_std,
            chi2=float(np.sum((resid / rms) ** 2)),
            dof=float(max(y0.size - 1, 1)),
            redchi=float("nan"),
            aic=float("nan"),
            success=True,
            message="k0",
        )

    x0 = seed_components(y, v, k, rms=rms, min_sep_kms=min_sep_kms, snr_floor=snr_floor)
    v_lo, v_hi = float(v.min()), float(v.max())
    amp_hi = max(float(np.max(np.abs(y0))) * 2.0, snr_floor * rms)
    lo = []
    hi = []
    for i in range(k):
        lo.extend([0.0, v_lo, float(sigma_min_kms)])
        hi.extend([amp_hi, v_hi, float(sigma_max_kms)])
    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)
    x0 = np.clip(x0, lo, hi)

    def residual(p: np.ndarray) -> np.ndarray:
        return (_gaussian_sum(v, p) - y0) / rms

    try:
        sol = least_squares(
            residual,
            x0,
            bounds=(lo, hi),
            method="trf",
            max_nfev=200 * max(k, 1),
        )
    except Exception as exc:  ### noqa: BLE001
        return _empty_fit(rms=rms, message=f"fit_exception:{exc}")

    p = sol.x
    model = _gaussian_sum(v, p)
    resid = y0 - model
    n = int(y0.size)
    npar = 3 * k
    dof = float(max(n - npar, 1))
    chi2 = float(np.sum((resid / rms) ** 2))
    redchi = chi2 / dof
    ssr = float(np.sum(resid**2))
    ### ScousePy-style AIC for Gaussian residuals (Henshaw tutorial).
    aic = float(n * np.log(max(ssr / n, 1e-30)) + 2 * npar)
    if n < 40:
        aic += float(2 * npar * (npar + 1) / max(n - npar - 1, 1))

    ### Rough 1sigma errors from approximate Hessian (J^T J).
    amp_errs = np.full(k, np.nan, dtype=np.float64)
    v_errs = np.full(k, np.nan, dtype=np.float64)
    sigma_errs = np.full(k, np.nan, dtype=np.float64)
    try:
        jac = sol.jac
        jtj = jac.T @ jac
        cov = np.linalg.inv(jtj) * max(redchi, 1.0)
        err = np.sqrt(np.clip(np.diag(cov), 0.0, None))
        for i in range(k):
            amp_errs[i] = float(err[3 * i])
            v_errs[i] = float(err[3 * i + 1])
            sigma_errs[i] = float(err[3 * i + 2])
    except Exception:  ### noqa: BLE001
        pass

    amps = p[0::3].copy()
    vs = p[1::3].copy()
    sigmas = p[2::3].copy()
    ### Sort components by increasing velocity (stable, Scouse-like).
    order = np.argsort(vs)
    return FitResult(
        k=k,
        amps=amps[order],
        vs=vs[order],
        sigmas=sigmas[order],
        amp_errs=amp_errs[order],
        v_errs=v_errs[order],
        sigma_errs=sigma_errs[order],
        rms=rms,
        resid_std=float(np.std(resid)),
        chi2=chi2,
        dof=dof,
        redchi=float(redchi),
        aic=aic,
        success=bool(sol.success),
        message=str(sol.message),
    )


def fit_result_to_rows(
    fit: FitResult,
    *,
    l: float,
    b: float,
) -> np.ndarray:
    """Stack FitResult into (K, 15) rows. Empty (0, 15) if K=0."""
    if fit.k <= 0:
        return np.zeros((0, 15), dtype=np.float64)
    rows = np.zeros((fit.k, 15), dtype=np.float64)
    for i in range(fit.k):
        rows[i] = [
            float(fit.k),
            float(l),
            float(b),
            float(fit.amps[i]),
            float(fit.amp_errs[i]) if np.isfinite(fit.amp_errs[i]) else 0.0,
            float(fit.vs[i]),
            float(fit.v_errs[i]) if np.isfinite(fit.v_errs[i]) else 0.0,
            float(fit.sigmas[i]),
            float(fit.sigma_errs[i]) if np.isfinite(fit.sigma_errs[i]) else 0.0,
            float(fit.rms),
            float(fit.resid_std),
            float(fit.chi2),
            float(fit.dof),
            float(fit.redchi),
            float(fit.aic),
        ]
    return rows


def write_handoff_dat(path, rows: np.ndarray, *, header_comment: str | None = None) -> None:
    """Write 15-column ascii table (one row per component)."""
    path = __import__("pathlib").Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    comment = header_comment or (
        "ncomps l b amp amp_err v v_err sigma sigma_err "
        "rms resid_std chi2 dof redchi aic  "
        "(sigma = Gaussian dispersion km/s)"
    )
    if rows.size == 0:
        path.write_text(f"# {comment}\n", encoding="utf-8")
        return
    np.savetxt(
        path,
        rows,
        fmt=[
            "%8.0f",
            "%12.5f",
            "%12.5f",
            "%10.4f",
            "%10.4f",
            "%12.3f",
            "%10.3f",
            "%10.3f",
            "%10.3f",
            "%10.4f",
            "%12.4f",
            "%12.4f",
            "%10.1f",
            "%10.4f",
            "%12.4f",
        ],
        header=comment,
        comments="# ",
    )


def parse_handoff_dat(path) -> dict[tuple[float, float], np.ndarray]:
    """Group handoff/Henshaw .dat rows by (l, b)."""
    from collections import defaultdict
    from pathlib import Path

    arr = np.loadtxt(Path(path))
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    by_pos: dict[tuple[float, float], list[np.ndarray]] = defaultdict(list)
    for row in arr:
        key = (round(float(row[1]), 5), round(float(row[2]), 5))
        by_pos[key].append(row)
    return {k: np.vstack(v) for k, v in by_pos.items()}


def fwhm_to_sigma(fwhm_kms: np.ndarray | float) -> np.ndarray:
    return np.asarray(fwhm_kms, dtype=np.float64) / _FWHM_TO_SIGMA


def sigma_to_fwhm(sigma_kms: np.ndarray | float) -> np.ndarray:
    return np.asarray(sigma_kms, dtype=np.float64) * _FWHM_TO_SIGMA
