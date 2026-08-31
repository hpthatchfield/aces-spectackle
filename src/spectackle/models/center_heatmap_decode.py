### Decode component count and centers from a center heatmap (Scheme C / heatmap path).
from __future__ import annotations

import numpy as np
import torch
from scipy.signal import find_peaks


def decode_centers_from_heatmap(
    prob: np.ndarray,
    vel_kms: np.ndarray | None = None,
    *,
    valid_mask: np.ndarray | None = None,
    height: float = 0.35,
    prominence: float = 0.15,
    min_sep_kms: float = 4.0,
    Kmax: int | None = None,
) -> tuple[int, np.ndarray]:
    """
    Peak-pick on a per-channel center probability map.

    Returns (k, peak_indices) where peak_indices are channel indices into the full axis.
    height/prominence are on the [0, 1] heatmap scale (post-sigmoid).
    """
    p = np.asarray(prob, dtype=np.float64).reshape(-1)
    n = p.size
    valid = np.ones(n, dtype=bool) if valid_mask is None else np.asarray(valid_mask, dtype=bool).reshape(-1)
    valid &= np.isfinite(p)
    if not np.any(valid):
        return 0, np.zeros(0, dtype=np.int64)

    y = p[valid]
    idx_map = np.flatnonzero(valid)
    if vel_kms is not None:
        vel = np.asarray(vel_kms, dtype=np.float64).reshape(-1)
        v = vel[valid]
        dv = float(np.median(np.diff(v))) if v.size > 1 else 2.0
    else:
        dv = 2.0
    min_dist = max(1, int(round(float(min_sep_kms) / max(abs(dv), 1e-6))))

    peak_amp = float(np.max(y)) if y.size else 0.0
    h = max(float(height), 0.25 * peak_amp)
    prom = max(float(prominence), 0.5 * h)

    peaks, _ = find_peaks(y, height=h, prominence=prom, distance=min_dist)
    if peaks.size == 0:
        return 0, np.zeros(0, dtype=np.int64)
    peak_idx = idx_map[peaks]
    if Kmax is not None:
        if peak_idx.size > int(Kmax):
            ### Keep the Kmax strongest heatmap peaks.
            order = np.argsort(p[peak_idx])[::-1][: int(Kmax)]
            peak_idx = np.sort(peak_idx[order])
    return int(peak_idx.size), peak_idx.astype(np.int64)


def decode_k_batch_from_heatmap(
    prob: np.ndarray,
    vel_kms: np.ndarray | None = None,
    *,
    valid_mask: np.ndarray | None = None,
    height: float = 0.35,
    prominence: float = 0.15,
    min_sep_kms: float = 4.0,
    Kmax: int = 10,
) -> np.ndarray:
    """prob: (B, C) -> k_hat (B,) int."""
    prob = np.asarray(prob, dtype=np.float64)
    if prob.ndim == 1:
        k, _ = decode_centers_from_heatmap(
            prob, vel_kms, valid_mask=valid_mask, height=height, prominence=prominence,
            min_sep_kms=min_sep_kms, Kmax=Kmax,
        )
        return np.array([k], dtype=np.int64)
    B = prob.shape[0]
    out = np.zeros(B, dtype=np.int64)
    vm = valid_mask
    for i in range(B):
        m = None if vm is None else vm[i]
        k, _ = decode_centers_from_heatmap(
            prob[i], vel_kms, valid_mask=m, height=height, prominence=prominence,
            min_sep_kms=min_sep_kms, Kmax=Kmax,
        )
        out[i] = k
    return out


def decode_centers_batch_from_heatmap(
    prob: np.ndarray,
    vel_kms: np.ndarray | None = None,
    *,
    valid_mask: np.ndarray | None = None,
    height: float = 0.35,
    prominence: float = 0.15,
    min_sep_kms: float = 4.0,
    Kmax: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Batch decode: returns
      k_hat: (B,) int
      v_slots: (B, Kmax) float, NaN-padded velocity centers (km/s)
      p_slots: (B, Kmax) float, NaN-padded peak probabilities
    """
    prob = np.asarray(prob, dtype=np.float64)
    if prob.ndim == 1:
        prob = prob[None, :]
        if valid_mask is not None:
            valid_mask = np.asarray(valid_mask)[None, :]
    B, _C = prob.shape
    Kmax = int(Kmax)
    k_out = np.zeros(B, dtype=np.int64)
    v_slots = np.full((B, Kmax), np.nan, dtype=np.float32)
    p_slots = np.full((B, Kmax), np.nan, dtype=np.float32)
    vel = None if vel_kms is None else np.asarray(vel_kms, dtype=np.float64).reshape(-1)
    for i in range(B):
        m = None if valid_mask is None else valid_mask[i]
        k, peak_idx = decode_centers_from_heatmap(
            prob[i],
            vel_kms,
            valid_mask=m,
            height=height,
            prominence=prominence,
            min_sep_kms=min_sep_kms,
            Kmax=Kmax,
        )
        k_out[i] = k
        if k <= 0:
            continue
        ### Peak indices are already sorted by channel (velocity order).
        for j, ix in enumerate(peak_idx.tolist()):
            if j >= Kmax:
                break
            p_slots[i, j] = float(prob[i, int(ix)])
            if vel is not None:
                v_slots[i, j] = float(vel[int(ix)])
            else:
                v_slots[i, j] = float(ix)
    return k_out, v_slots, p_slots


@torch.no_grad()
def eval_center_heatmap_k_decode(
    model,
    val_loader,
    v_axis,
    *,
    device: str = "cpu",
    label_sigma_kms: float = 4.0,
    height: float = 0.35,
    prominence: float = 0.15,
    min_sep_kms: float = 4.0,
    Kmax: int = 10,
) -> dict[str, float]:
    """Decode K from heatmap peaks; report MAE vs true component count."""
    from spectackle.training import batch_model_input

    model.eval()
    v_np = np.asarray(v_axis, dtype=np.float64)
    abs_err: list[float] = []
    exact = 0
    n = 0
    by_k: dict[int, list[float]] = {}
    for batch in val_loader:
        x, mask = batch_model_input(batch, device)
        logits = model(x, mask)
        prob = torch.sigmoid(logits).cpu().numpy()
        k_true = batch["K_true"].cpu().numpy().reshape(-1).astype(int)
        vm = mask.cpu().numpy() if mask is not None else None
        k_pred = decode_k_batch_from_heatmap(
            prob, v_np, valid_mask=vm, height=height, prominence=prominence,
            min_sep_kms=min_sep_kms, Kmax=Kmax,
        )
        for kt, kp in zip(k_true.tolist(), k_pred.tolist()):
            err = abs(int(kp) - int(kt))
            abs_err.append(float(err))
            exact += int(kp == kt)
            n += 1
            by_k.setdefault(int(kt), []).append(float(err))
    mae_by_k = {str(k): float(np.mean(v)) for k, v in sorted(by_k.items())}
    return {
        "k_mae": float(np.mean(abs_err)) if abs_err else float("nan"),
        "k_exact_frac": float(exact / max(1, n)),
        "decode_height": float(height),
        "decode_prominence": float(prominence),
        "decode_min_sep_kms": float(min_sep_kms),
        "mae_by_k": mae_by_k,
    }


def _k_mae_from_cached(
    prob: np.ndarray,
    k_true: np.ndarray,
    v_np: np.ndarray,
    valid_mask: np.ndarray | None,
    *,
    height: float,
    prominence: float,
    min_sep_kms: float,
    Kmax: int,
) -> dict[str, float]:
    """K MAE / exact on cached (B, C) heatmap probs (no model re-forward)."""
    k_pred = decode_k_batch_from_heatmap(
        prob, v_np, valid_mask=valid_mask, height=height, prominence=prominence,
        min_sep_kms=min_sep_kms, Kmax=Kmax,
    )
    abs_err = np.abs(k_pred.astype(np.int64) - k_true.astype(np.int64)).astype(np.float64)
    exact = int(np.sum(k_pred == k_true))
    n = int(k_true.size)
    by_k: dict[int, list[float]] = {}
    for kt, err in zip(k_true.tolist(), abs_err.tolist()):
        by_k.setdefault(int(kt), []).append(float(err))
    mae_by_k = {str(k): float(np.mean(v)) for k, v in sorted(by_k.items())}
    return {
        "k_mae": float(np.mean(abs_err)) if n else float("nan"),
        "k_exact_frac": float(exact / max(1, n)),
        "decode_height": float(height),
        "decode_prominence": float(prominence),
        "decode_min_sep_kms": float(min_sep_kms),
        "mae_by_k": mae_by_k,
    }


@torch.no_grad()
def tune_heatmap_decode_thresholds(
    model,
    val_loader,
    v_axis,
    *,
    device: str = "cpu",
    Kmax: int = 10,
    heights: tuple[float, ...] = (0.25, 0.35, 0.45, 0.55),
    prominences: tuple[float, ...] = (0.08, 0.12, 0.18, 0.25),
    min_sep_kms: float = 4.0,
) -> dict:
    """
    Grid search decode thresholds on val set (small K MAE).

    Caches heatmap probs once, then only re-runs peak decode (critical for long ACES axes).
    """
    from spectackle.training import batch_model_input

    model.eval()
    v_np = np.asarray(v_axis, dtype=np.float64)
    probs: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    k_trues: list[np.ndarray] = []
    for batch in val_loader:
        x, mask = batch_model_input(batch, device)
        logits = model(x, mask)
        probs.append(torch.sigmoid(logits).cpu().numpy())
        if mask is not None:
            masks.append(mask.cpu().numpy())
        k_trues.append(batch["K_true"].cpu().numpy().reshape(-1).astype(int))
    if not probs:
        return {}
    prob = np.concatenate(probs, axis=0)
    k_true = np.concatenate(k_trues, axis=0)
    vm = np.concatenate(masks, axis=0) if masks else None

    best = None
    for h in heights:
        for p in prominences:
            m = _k_mae_from_cached(
                prob, k_true, v_np, vm,
                height=h, prominence=p, min_sep_kms=min_sep_kms, Kmax=Kmax,
            )
            if best is None or m["k_mae"] < best["metrics"]["k_mae"]:
                best = {"height": h, "prominence": p, "metrics": m}
    return best or {}


__all__ = [
    "decode_centers_from_heatmap",
    "decode_k_batch_from_heatmap",
    "decode_centers_batch_from_heatmap",
    "eval_center_heatmap_k_decode",
    "tune_heatmap_decode_thresholds",
]
