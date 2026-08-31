### MOPRA-specific preprocessing: BLANK=-1 and optional NaN handling.
from __future__ import annotations

import numpy as np

from spectackle.data.preprocess import prepare_spectrum_input as _prepare_base
from spectackle.data.scouse_saa import estimate_spectrum_rms

MOPRA_BLANK_VALUE = -1.0
### zscore: (T - mean) / std over valid channels (legacy).
### rms: (T - median) / Scouse-style sigma_rms; preserves physical SNR in the input.
NORM_MODES = ("zscore", "rms")


def valid_mask_mopra(spec_raw: np.ndarray, *, blank_value: float | None = MOPRA_BLANK_VALUE) -> np.ndarray:
    """True = real channel. Treats FITS BLANK, NaN, and exact-zero as invalid."""
    a = np.asarray(spec_raw)
    ok = np.isfinite(a)
    if blank_value is not None:
        ok &= a != float(blank_value)
    ok &= a != 0.0
    return ok


def prepare_mopra_input(
    spec_raw: np.ndarray,
    *,
    blank_value: float | None = MOPRA_BLANK_VALUE,
    norm_mode: str = "zscore",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Normalize a spectrum (or batch) for MOPRA training / cube inference.

    Accepts (C,) or (B, C). Returns (x, mask) same shape:
      x    : float32 normalized values; invalid channels 0
      mask : float32 1 on valid channels

    norm_mode:
      zscore : (T - mean_valid) / std_valid  (legacy; erases absolute SNR)
      rms    : (T - median_valid) / sigma_rms     (Scouse-style; peak ~ SNR units)
    """
    mode = str(norm_mode).lower().strip()
    if mode not in NORM_MODES:
        raise ValueError(f"Unknown norm_mode {norm_mode!r}; use one of {NORM_MODES}")

    a = np.asarray(spec_raw, dtype=np.float64)
    one_d = a.ndim == 1
    if one_d:
        a = a[None, :]

    valid = valid_mask_mopra(a, blank_value=blank_value)
    m = valid.astype(np.float32)

    if mode == "zscore":
        a_filled = np.where(valid, a, 0.0)
        cnt = valid.sum(axis=1, keepdims=True).clip(min=1)
        mu = a_filled.sum(axis=1, keepdims=True) / cnt
        var = np.where(valid, (a - mu) ** 2, 0.0).sum(axis=1, keepdims=True) / cnt
        sd = np.sqrt(var) + 1e-6
        x = np.where(valid, (a - mu) / sd, 0.0).astype(np.float32)
    else:
        ### Per-spectrum robust baseline + Scouse sigma_rms (loop; B is modest at train/infer).
        x = np.zeros_like(a, dtype=np.float32)
        for i in range(a.shape[0]):
            ok = valid[i]
            if not np.any(ok):
                continue
            s = a[i, ok]
            med = float(np.median(s))
            rms = estimate_spectrum_rms(s)
            if not np.isfinite(rms) or rms <= 0.0:
                rms = float(np.std(s)) if s.size > 1 else 1e-6
            rms = max(float(rms), 1e-6)
            x[i, ok] = ((s - med) / rms).astype(np.float32)

    if one_d:
        return x[0], m[0]
    return x, m


def prepare_mopra_input_from_base(
    spec_raw: np.ndarray,
    *,
    norm_mode: str = "zscore",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Synth / already-stripped spectra: no FITS BLANK sentinel, but still drop exact 0 pads.
    """
    if str(norm_mode).lower().strip() == "zscore":
        ### Keep legacy path for bit-stability with older zscore runs.
        return _prepare_base(spec_raw)
    return prepare_mopra_input(spec_raw, blank_value=None, norm_mode=norm_mode)


def _channel_mask_mopra(
    spec_raw: np.ndarray,
    *,
    blank_value: float | None = MOPRA_BLANK_VALUE,
    vel_kms: np.ndarray | None = None,
    vel_range: tuple[float, float] | None = None,
) -> np.ndarray:
    """Valid-channel mask, optionally restricted to a velocity window."""
    valid = valid_mask_mopra(spec_raw, blank_value=blank_value)
    if vel_kms is not None and vel_range is not None:
        vlo, vhi = float(vel_range[0]), float(vel_range[1])
        in_vel = (vel_kms >= vlo) & (vel_kms <= vhi)
        if spec_raw.ndim == 1:
            valid &= in_vel
        else:
            valid &= in_vel[None, :]
    return valid


def snr_peak_scouse_mopra(
    spec_raw: np.ndarray,
    *,
    blank_value: float | None = MOPRA_BLANK_VALUE,
    vel_kms: np.ndarray | None = None,
    vel_range: tuple[float, float] | None = None,
    min_channels: int = 10,
) -> np.ndarray:
    """
    ScousePy-style peak SNR: (max T - median T) / sigma_rms in the evaluation window.

    sigma_rms follows scousepy.stage_1.calc_rms (negative-channel MAD). Use vel_range
    to avoid counting high-SNR lines outside the science window as "detectable".

    spec_raw: (C,) or (B, C). Returns scalar or (B,) float64.
    """
    a = np.asarray(spec_raw, dtype=np.float64)
    one_d = a.ndim == 1
    if one_d:
        a = a[None, :]

    valid = _channel_mask_mopra(
        a, blank_value=blank_value, vel_kms=vel_kms, vel_range=vel_range
    )
    snr = np.zeros(a.shape[0], dtype=np.float64)
    for i in range(a.shape[0]):
        s = a[i, valid[i]]
        if s.size < min_channels:
            snr[i] = 0.0
            continue
        rms = estimate_spectrum_rms(s)
        if not np.isfinite(rms) or rms <= 0.0:
            snr[i] = 0.0
            continue
        snr[i] = float(np.max(s) - np.median(s)) / rms
    return snr[0] if one_d else snr


def snr_peak_rms_mopra(
    spec_raw: np.ndarray,
    *,
    blank_value: float | None = MOPRA_BLANK_VALUE,
) -> np.ndarray:
    """
    Legacy peak SNR: max|T| / RMS over valid channels (after mean removal).

    Prefer snr_peak_scouse_mopra for CMZ inference gating; this metric can be
    inflated when the global peak sits outside the line-rich velocity window.

    spec_raw: (C,) or (B, C). Returns scalar or (B,) float64.
    """
    a = np.asarray(spec_raw, dtype=np.float64)
    one_d = a.ndim == 1
    if one_d:
        a = a[None, :]

    valid = valid_mask_mopra(a, blank_value=blank_value)
    filled = np.where(valid, a, 0.0)
    cnt = valid.sum(axis=1).clip(min=1)
    mu = filled.sum(axis=1) / cnt
    var = np.where(valid, (a - mu[:, None]) ** 2, 0.0).sum(axis=1) / cnt
    rms = np.sqrt(var) + 1e-6
    spec_for_peak = np.where(valid, a, np.nan)
    with np.errstate(all="ignore"):
        peak = np.nanmax(np.abs(spec_for_peak), axis=1)
    peak = np.where(np.isfinite(peak), peak, 0.0)
    snr = peak / rms
    return snr[0] if one_d else snr


__all__ = [
    "MOPRA_BLANK_VALUE",
    "NORM_MODES",
    "valid_mask_mopra",
    "prepare_mopra_input",
    "prepare_mopra_input_from_base",
    "snr_peak_scouse_mopra",
    "snr_peak_rms_mopra",
]
