### Scouse-style and glance-visible component acceptance for synthetic K labels.
from __future__ import annotations

import numpy as np

from .generator import channel_width_kms as _channel_width_kms
from .resolvable_peaks import count_resolvable_peaks
from .scouse_saa import estimate_spectrum_rms

### FWHM = _FWHM_OVER_SIGMA * Gaussian sigma (km/s).
_FWHM_OVER_SIGMA = 2.3548


def matched_filter_snr(
    amp: float,
    noise_std: float,
    fwhm_kms: float,
    delta_v_kms: float,
) -> float:
    """
    Matched-filter SNR for a Gaussian line (training labels; uses true A, W).

    SNR_matched = (A/sigma) * pi^{1/4} * sqrt(W / (2.3548 * dv))
    """
    if noise_std <= 0.0 or not np.isfinite(noise_std):
        return float("inf") if float(amp) > 0 else 0.0
    w = max(float(fwhm_kms), 1e-6)
    dv = max(abs(float(delta_v_kms)), 1e-6)
    return float(amp) / float(noise_std) * (np.pi**0.25) * np.sqrt(w / (_FWHM_OVER_SIGMA * dv))


def min_sep_keep_indices(
    amps: np.ndarray,
    mus: np.ndarray,
    *,
    min_sep_kms: float = 4.0,
) -> list[int]:
    """
    Hard merge floor: greedy keep brightest components with |dv| >= min_sep_kms.

    Closer pairs collapse to the stronger component (resolution degeneracy).
    """
    amps = np.asarray(amps, dtype=np.float64).reshape(-1)
    mus = np.asarray(mus, dtype=np.float64).reshape(-1)
    n = int(amps.size)
    if n == 0:
        return []
    order = np.argsort(-amps)
    kept: list[int] = []
    sep = float(min_sep_kms)
    for idx in order.tolist():
        ok = True
        for j in kept:
            if abs(float(mus[idx]) - float(mus[j])) < sep:
                ok = False
                break
        if ok:
            kept.append(int(idx))
    kept.sort()
    return kept


def matched_filter_keep_indices(
    amps: np.ndarray,
    mus: np.ndarray,
    sigs: np.ndarray,
    noise_std: float,
    delta_v_kms: float,
    *,
    snr_matched_tol: float = 3.0,
) -> list[int]:
    """
    Credit components with matched-filter SNR >= tol (true A, W; others conceptually removed).

    For each survivor, SNR_matched uses that component's amplitude and FWHM.
    """
    amps = np.asarray(amps, dtype=np.float64).reshape(-1)
    sigs = np.asarray(sigs, dtype=np.float64).reshape(-1)
    n = int(amps.size)
    keep: list[int] = []
    for i in range(n):
        fwhm = float(sigs[i]) * _FWHM_OVER_SIGMA
        snr_m = matched_filter_snr(float(amps[i]), float(noise_std), fwhm, float(delta_v_kms))
        if snr_m >= float(snr_matched_tol):
            keep.append(i)
    return keep


def _gauss1d(v: np.ndarray, amp: float, mu: float, sig: float) -> np.ndarray:
    dv = (v - mu) / (sig + 1e-6)
    return amp * np.exp(-0.5 * dv * dv)


def _gauss_from_positive_flux(
    y: np.ndarray,
    v: np.ndarray,
    noise_std: float,
    *,
    snr_tol: float = 3.0,
    sigma_min_kms: float = 1.0,
    sigma_max_kms: float = 80.0,
) -> tuple[float, float, float] | None:
    """
    Fast single-Gaussian estimate seeded at the brightest channel.

    Width from half-max around that peak (local), not the full-spectrum flux
    centroid, so a well-separated secondary is not absorbed into one wide peel.
    Shoulders without a second local max still leave residual flux after subtract.
    """
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    if y.size < 8 or v.size != y.size:
        return None
    sigma = float(noise_std)
    if not np.isfinite(sigma) or sigma <= 0.0:
        return None
    peak_i = int(np.argmax(y))
    amp = float(y[peak_i])
    if not np.isfinite(amp) or amp < float(snr_tol) * sigma:
        return None
    mu = float(v[peak_i])
    dv = float(np.median(np.diff(v))) if v.size > 1 else 2.0
    half = 0.5 * amp
    ### Expand from peak until below half-max (local FWHM proxy).
    left = peak_i
    while left > 0 and float(y[left]) >= half:
        left -= 1
    right = peak_i
    while right < y.size - 1 and float(y[right]) >= half:
        right += 1
    fwhm = max(abs(dv), abs(float(v[right]) - float(v[left])))
    sig = fwhm / 2.355
    sig = max(float(sigma_min_kms), min(float(sigma_max_kms), float(sig)))
    return float(amp), float(mu), float(sig)


def residual_flux_significant(
    resid: np.ndarray,
    v: np.ndarray,
    noise_std: float,
    *,
    integ_snr_tol: float = 5.0,
    window_half_kms: float = 30.0,
) -> bool:
    """
    Flux/energy test: positive integrated residual in a local window vs noise.

    Uses sum(max(r,0)) / (sigma sqrtn_win) around the strongest positive residual channel.
    Not a local-maximum / prominence peak finder on the original spectrum.
    """
    r = np.asarray(resid, dtype=np.float64).reshape(-1)
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    if r.size == 0 or v.size != r.size:
        return False
    sigma = float(noise_std)
    if not np.isfinite(sigma) or sigma <= 0.0:
        sigma = float(estimate_spectrum_rms(r))
    if not np.isfinite(sigma) or sigma <= 0.0:
        return False

    peak_i = int(np.argmax(r))
    if float(r[peak_i]) <= 0.0:
        return False
    v0 = float(v[peak_i])
    win = (v >= (v0 - float(window_half_kms))) & (v <= (v0 + float(window_half_kms)))
    if not np.any(win):
        win = np.ones(r.size, dtype=bool)
    r_win = r[win]
    n_win = int(r_win.size)
    if n_win < 3:
        return False
    pos_flux = float(np.sum(np.clip(r_win, 0.0, None)))
    integ_snr = pos_flux / (sigma * np.sqrt(float(n_win)))
    return bool(integ_snr >= float(integ_snr_tol))


def count_residual_credited_components(
    spec: np.ndarray,
    v_axis: np.ndarray,
    noise_std: float,
    k_max: int,
    *,
    snr_tol: float = 3.0,
    integ_snr_tol: float = 5.0,
    window_half_kms: float = 30.0,
    primary_amp: float | None = None,
    primary_mu: float | None = None,
    primary_sig: float | None = None,
) -> int:
    """
    Iteratively credit components by Gaussian peel + residual flux check.

    1) Subtract known primary (or a flux-based Gaussian estimate); credit 1.
    2) If residual positive integrated flux is significant vs noise, credit +1
       even without a second local maximum; peel an approximate Gaussian and repeat.
    3) Stop when residual is noise-like or k_max is reached.

    Returns n_credited in 0..k_max.
    """
    k_max = int(max(0, k_max))
    if k_max <= 0:
        return 0

    spec = np.asarray(spec, dtype=np.float64).reshape(-1)
    v = np.asarray(v_axis, dtype=np.float64).reshape(-1)
    valid = np.isfinite(spec) & np.isfinite(v)
    if valid.sum() < 8:
        return 0
    y = spec[valid].copy()
    vv = v[valid]
    med = float(np.median(y))
    working = y - med
    sigma = float(noise_std)
    if not np.isfinite(sigma) or sigma <= 0.0:
        sigma = float(estimate_spectrum_rms(working))
    if not np.isfinite(sigma) or sigma <= 0.0:
        sigma = float(np.std(working) + 1e-12)

    ### First component: prefer known primary; else flux-mass estimate.
    if primary_amp is not None and primary_mu is not None and primary_sig is not None:
        if float(primary_amp) < float(snr_tol) * sigma:
            return 0
        fit = (float(primary_amp), float(primary_mu), float(primary_sig))
    else:
        fit = _gauss_from_positive_flux(working, vv, sigma, snr_tol=snr_tol)
        if fit is None:
            return 0
    working = working - _gauss1d(vv, fit[0], fit[1], fit[2])
    n_cred = 1

    while n_cred < k_max:
        if not residual_flux_significant(
            working,
            vv,
            sigma,
            integ_snr_tol=integ_snr_tol,
            window_half_kms=window_half_kms,
        ):
            break
        ### Residual flux is real -> credit another component (no local-max required).
        n_cred += 1
        if n_cred >= k_max:
            break
        peel = _gauss_from_positive_flux(working, vv, sigma, snr_tol=snr_tol)
        if peel is None:
            break
        working = working - _gauss1d(vv, peel[0], peel[1], peel[2])
    return int(n_cred)


def scouse_accept_indices(
    amps: np.ndarray,
    mus: np.ndarray,
    sigs: np.ndarray,
    noise_std: float,
    *,
    snr_tol: float = 3.0,
    min_sep_factor: float | None = 3.0,
    min_sep_channels: float | None = 5.0,
    channel_width_kms: float | None = None,
    cfg: dict | None = None,
) -> list[int]:
    """
    Greedy deblend: keep strongest peaks that pass amp/sigma_rms >= snr_tol and separation rules.

    amps: peak amplitudes at component centers (same units as spec_clean).
    mus, sigs: centers and Gaussian sigma in km/s.
    Returns sorted indices of accepted components.
    """
    n = int(len(amps))
    if n == 0:
        return []
    if noise_std <= 0.0 or not np.isfinite(noise_std):
        return list(range(n))

    if channel_width_kms is None and cfg is not None:
        channel_width_kms = float(_channel_width_kms(cfg))

    order = np.argsort(-np.asarray(amps, dtype=np.float64))
    kept: list[int] = []
    for idx in order.tolist():
        snr = float(amps[idx]) / float(noise_std)
        if snr < float(snr_tol):
            continue
        ok = True
        for j in kept:
            min_sep = 0.0
            if min_sep_factor is not None:
                min_sep = max(min_sep, float(min_sep_factor) * (float(sigs[idx]) + float(sigs[j])))
            if min_sep_channels is not None and channel_width_kms is not None:
                min_sep = max(min_sep, float(min_sep_channels) * float(channel_width_kms))
            if abs(float(mus[idx]) - float(mus[j])) < min_sep:
                ok = False
                break
        if ok:
            kept.append(int(idx))
    kept.sort()
    return kept


def _rebuild_ex_from_keep(ex: dict, keep: list[int]) -> dict:
    """Drop rejected components from clean/noisy spectrum; preserve baseline+noise residual."""
    k = int(ex["k"])
    amps = ex["component_amp"][:k]
    mus = ex["component_v_kms"][:k]
    sigs = ex["component_sigma"][:k]
    k_new = len(keep)
    ex["k_drawn"] = k
    ex["k"] = k_new
    if k_new == k:
        return ex

    v = ex["v_axis"]
    Kmax = int(len(ex["component_amp"]))
    spec_clean_new = np.zeros_like(ex["spec_clean"])
    A = np.zeros(Kmax, dtype=np.float32)
    mu = np.zeros(Kmax, dtype=np.float32)
    sig = np.ones(Kmax, dtype=np.float32)
    for j, i in enumerate(keep):
        dv = (v - mus[i]) / (sigs[i] + 1e-6)
        spec_clean_new += amps[i] * np.exp(-0.5 * dv * dv).astype(np.float32)
        A[j] = amps[i]
        mu[j] = mus[i]
        sig[j] = sigs[i]

    residual = ex["spec"] - ex["spec_clean"]
    ex["spec_clean"] = spec_clean_new.astype(np.float32)
    ex["spec"] = (spec_clean_new + residual).astype(np.float32)
    ex["component_amp"] = A
    ex["component_v_kms"] = mu
    ex["component_sigma"] = sig
    return ex


def apply_scouse_label_filter(ex: dict, cfg: dict) -> dict:
    """
    Re-label k and rebuild spec/spec_clean after Scouse acceptance on drawn components.

    Preserves baseline + noise draw: spec_new = spec_clean_new + (spec - spec_clean).
    """
    gen = cfg.get("gen", {})
    if not gen.get("scouse_label_k"):
        return ex

    k = int(ex["k"])
    if k <= 0:
        return ex

    noise_std = float(ex["noise_std"][0])
    amps = ex["component_amp"][:k]
    mus = ex["component_v_kms"][:k]
    sigs = ex["component_sigma"][:k]

    sep_factor = gen.get("scouse_min_sep_factor", gen.get("min_component_separation", 3.0))
    min_sep_ch = gen.get("scouse_min_sep_channels", gen.get("min_sep_channels"))
    snr_tol = float(gen.get("scouse_snr_tol", 3.0))

    keep = scouse_accept_indices(
        amps,
        mus,
        sigs,
        noise_std,
        snr_tol=snr_tol,
        min_sep_factor=sep_factor,
        min_sep_channels=min_sep_ch,
        cfg=cfg,
    )
    return _rebuild_ex_from_keep(ex, keep)


def apply_glance_visible_label(ex: dict, cfg: dict) -> dict:
    """
    Label K by what is distinguishable by eye / above noise (not Scouse sep-merge).

    1) Drop components with amp/sigma_rms < snr_tol (noise-like; removed from spectrum).
    2) Keep remaining Gaussians in the spectrum (blends stay morphologically).
    3) Cap K with glance_cap_mode:
         - "resolvable" (default): min(k_snr, n_resolvable_peaks) peak-finder bump-cap
         - "residual": min(k_snr, n residual-credited via iterative Gaussian subtract
           + positive integrated residual flux vs noise)
         - "none": K = k_snr (no bump-cap; same as glance_cap_resolvable=False)

    Resolvable peaks use prominence >= max(prominence_sigma * sigma_rms, peak_frac * peak).
    Residual mode can credit unresolved secondaries without a second local maximum.
    """
    gen = cfg.get("gen", {})
    if not gen.get("glance_label_k"):
        return ex

    k = int(ex["k"])
    if k <= 0:
        ex["k_drawn"] = 0
        return ex

    noise_std = float(ex["noise_std"][0])
    snr_tol = float(gen.get("glance_snr_tol", gen.get("scouse_snr_tol", 3.0)))
    amps = np.asarray(ex["component_amp"][:k], dtype=np.float64)
    keep_snr = [i for i in range(k) if noise_std <= 0 or (amps[i] / noise_std) >= snr_tol]
    ex = _rebuild_ex_from_keep(ex, keep_snr)
    k_snr = int(ex["k"])
    ex["k_snr"] = k_snr
    if k_snr <= 0:
        return ex

    ### Cap mode: prefer glance_cap_mode; glance_cap_resolvable=False -> "none".
    cap_mode = str(gen.get("glance_cap_mode", "resolvable")).lower()
    if not bool(gen.get("glance_cap_resolvable", True)):
        cap_mode = "none"

    if cap_mode == "none":
        ex["k_resolvable"] = -1
        ex["k_residual"] = -1
        ex["k_matched"] = -1
        ex["k"] = k_snr
        return ex

    if cap_mode == "matched":
        ### Hard merge floor, then matched-filter SNR on survivors (true A, W).
        min_sep_kms = float(gen.get("glance_min_sep_kms", 4.0))
        snr_m_tol = float(gen.get("glance_matched_snr_tol", snr_tol))
        amps_k = np.asarray(ex["component_amp"][:k_snr], dtype=np.float64)
        mus_k = np.asarray(ex["component_v_kms"][:k_snr], dtype=np.float64)
        keep_sep = min_sep_keep_indices(amps_k, mus_k, min_sep_kms=min_sep_kms)
        ex = _rebuild_ex_from_keep(ex, keep_sep)
        k_sep = int(ex["k"])
        ex["k_sep"] = k_sep
        if k_sep <= 0:
            ex["k_resolvable"] = -1
            ex["k_residual"] = -1
            ex["k_matched"] = 0
            return ex
        amps_s = np.asarray(ex["component_amp"][:k_sep], dtype=np.float64)
        mus_s = np.asarray(ex["component_v_kms"][:k_sep], dtype=np.float64)
        sigs_s = np.asarray(ex["component_sigma"][:k_sep], dtype=np.float64)
        delta_v = float(_channel_width_kms(cfg))
        ### Conceptual: subtract other survivors from noisy spec, then matched-filter
        ### SNR for this component. With true params that equals the closed-form below.
        keep_m = matched_filter_keep_indices(
            amps_s,
            mus_s,
            sigs_s,
            noise_std,
            delta_v,
            snr_matched_tol=snr_m_tol,
        )
        ex = _rebuild_ex_from_keep(ex, keep_m)
        k_label = int(ex["k"])
        ex["k_resolvable"] = -1
        ex["k_residual"] = -1
        ex["k_matched"] = k_label
        return ex

    if cap_mode == "residual":
        n_cred = count_residual_credited_components(
            ex["spec"],
            ex["v_axis"],
            noise_std,
            k_snr,
            snr_tol=snr_tol,
            integ_snr_tol=float(gen.get("glance_residual_integ_snr", 5.0)),
            window_half_kms=float(gen.get("glance_residual_window_kms", 30.0)),
        )
        k_label = int(min(k_snr, max(0, n_cred)))
        ex["k_resolvable"] = -1
        ex["k_residual"] = int(n_cred)
        ex["k_matched"] = -1
        ex["k"] = k_label
    else:
        ### Default: resolvable-peak bump-cap.
        prom_sigma = float(gen.get("glance_prominence_sigma", 3.0))
        prom_mode = str(gen.get("glance_prominence_mode", "adaptive"))
        peak_frac = float(gen.get("glance_peak_frac", 0.15))
        min_sep_kms = float(gen.get("glance_min_sep_kms", 4.0))

        n_res, _ = count_resolvable_peaks(
            ex["spec"],
            ex["v_axis"],
            blank_value=None,
            vel_range=None,
            prominence_sigma=prom_sigma,
            min_sep_kms=min_sep_kms,
            prominence_mode=prom_mode,
            peak_frac=peak_frac,
        )
        ### Cap by physical SNR-passers: noise spikes cannot invent extra components.
        k_label = int(min(k_snr, max(0, n_res)))
        ex["k_resolvable"] = int(n_res)
        ex["k_residual"] = -1
        ex["k_matched"] = -1
        ex["k"] = k_label

    ### Spectrum keeps all SNR-passers (blends stay). Slot metadata for D-lite must
    ### match K: brightest k_label centers, then velocity-sorted (Scheme D convention).
    if 0 < k_label < k_snr:
        amps_k = np.asarray(ex["component_amp"][:k_snr], dtype=np.float64)
        mus_k = np.asarray(ex["component_v_kms"][:k_snr], dtype=np.float64)
        sigs_k = np.asarray(ex["component_sigma"][:k_snr], dtype=np.float64)
        top = np.argsort(-amps_k)[:k_label]
        top = top[np.argsort(mus_k[top])]
        Kmax = int(len(ex["component_amp"]))
        A = np.zeros(Kmax, dtype=np.float32)
        mu = np.zeros(Kmax, dtype=np.float32)
        sig = np.ones(Kmax, dtype=np.float32)
        for j, i in enumerate(top):
            A[j] = float(amps_k[i])
            mu[j] = float(mus_k[i])
            sig[j] = float(sigs_k[i])
        ex["component_amp"] = A
        ex["component_v_kms"] = mu
        ex["component_sigma"] = sig
    return ex


def apply_snr_component_label(ex: dict, cfg: dict) -> dict:
    """
    Label K = number of components with amp/sigma_rms >= snr_tol.

    Close / blended secondaries count if they clear the SNR floor. No resolvable-peak
    cap and no Scouse separation merge. Noise-like draws are removed from the spectrum.
    """
    gen = cfg.get("gen", {})
    if not gen.get("snr_label_k"):
        return ex

    k = int(ex["k"])
    if k <= 0:
        ex["k_drawn"] = 0
        return ex

    noise_std = float(ex["noise_std"][0])
    snr_tol = float(gen.get("snr_label_tol", gen.get("glance_snr_tol", gen.get("scouse_snr_tol", 3.0))))
    amps = np.asarray(ex["component_amp"][:k], dtype=np.float64)
    keep_snr = [i for i in range(k) if noise_std <= 0 or (amps[i] / noise_std) >= snr_tol]
    ex = _rebuild_ex_from_keep(ex, keep_snr)
    k_snr = int(ex["k"])
    ex["k_snr"] = k_snr
    ex["k_resolvable"] = -1
    ex["k"] = k_snr
    return ex


__all__ = [
    "apply_glance_visible_label",
    "apply_scouse_label_filter",
    "apply_snr_component_label",
    "count_residual_credited_components",
    "matched_filter_keep_indices",
    "matched_filter_snr",
    "min_sep_keep_indices",
    "residual_flux_significant",
    "scouse_accept_indices",
]
