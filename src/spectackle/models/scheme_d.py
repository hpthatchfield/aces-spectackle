### Scheme D: two-stage count + Gaussian params (Phase 1: oracle-K param head).
from __future__ import annotations

import math

import torch
import torch.nn as nn

from spectackle.models.pooling import masked_global_mean_pool


def sigma_bounds_from_cfg(cfg: dict) -> tuple[float, float]:
    """Read Gaussian sigma bounds (km/s) from a dataset cfg gen block."""
    gen = cfg.get("gen", {})
    return float(gen.get("sigma_min", 0.1)), float(gen.get("sigma_max", 10.0))


def synthesize_gaussian_stack(
    v_axis: torch.Tensor,
    mu_kms: torch.Tensor,
    sigma_kms: torch.Tensor,
    amp_norm: torch.Tensor,
    *,
    slot_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Sum of 1D Gaussians on a velocity grid in normalized amplitude units.

    v_axis: (C,) km/s
    mu_kms, sigma_kms, amp_norm: (B, Kmax)
    slot_mask: optional (B, Kmax) float 0/1 - zero inactive slots
    Returns (B, C).
    """
    if v_axis.dim() != 1:
        raise ValueError("v_axis must be 1D")
    sig = sigma_kms.clamp(min=1e-3)
    dv = (v_axis[None, None, :] - mu_kms[:, :, None]) / sig[:, :, None]
    g = amp_norm[:, :, None] * torch.exp(-0.5 * dv * dv)
    if slot_mask is not None:
        g = g * slot_mask[:, :, None].to(dtype=g.dtype)
    return g.sum(dim=1)


class OracleParamNet1DDeep(nn.Module):
    """
    Phase 1 Scheme D: shared conv encoder + param slots (v, log_sigma, amp_norm).

    K_true / component_valid select active slots at train time; no Stage-1 head yet.
    """

    def __init__(
        self,
        Kmax: int,
        width: int = 96,
        n_blocks: int = 6,
        *,
        sigma_min_kms: float = 0.1,
        sigma_max_kms: float = 10.0,
    ):
        super().__init__()
        self.Kmax = int(Kmax)
        if float(sigma_max_kms) <= float(sigma_min_kms):
            raise ValueError("sigma_max_kms must exceed sigma_min_kms")
        log_lo = math.log(float(sigma_min_kms))
        log_hi = math.log(float(sigma_max_kms))
        self.register_buffer("log_sig_min", torch.tensor(log_lo, dtype=torch.float32))
        self.register_buffer("log_sig_max", torch.tensor(log_hi, dtype=torch.float32))
        blocks: list[nn.Module] = []
        for i in range(int(n_blocks)):
            in_ch = 1 if i == 0 else width
            blocks.extend(
                [
                    nn.Conv1d(in_ch, width, 9, padding=4),
                    nn.BatchNorm1d(width),
                    nn.ReLU(),
                ]
            )
        self.conv = nn.Sequential(*blocks)
        n_out = self.Kmax * 3
        self.param_head = nn.Sequential(
            nn.Linear(width, width // 2),
            nn.ReLU(),
            nn.Linear(width // 2, n_out),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        x: (B, C) normalized spectrum.
        Returns (B, Kmax, 3): [v_kms, log_sigma_kms, amp_norm].
        """
        h = self.conv(x.unsqueeze(1))
        h = masked_global_mean_pool(h, mask)
        raw = self.param_head(h).view(-1, self.Kmax, 3)
        ### Bounded log sigma via sigmoid -> [log(sigma_min), log(sigma_max)].
        out = raw.clone()
        out[:, :, 1] = self.log_sig_min + torch.sigmoid(raw[:, :, 1]) * (self.log_sig_max - self.log_sig_min)
        return out


__all__ = ["OracleParamNet1DDeep", "sigma_bounds_from_cfg", "synthesize_gaussian_stack"]
