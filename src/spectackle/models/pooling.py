### Masked global mean pool over the velocity axis (B, W, C) -> (B, W).
from __future__ import annotations

import torch


def masked_global_mean_pool(h: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """
    h: (B, W, C) conv features. mask: optional (B, C) with 1=valid channel, 0=pad.
  When mask is None, plain mean over C (legacy Scheme B/C path).
    """
    if mask is None:
        return h.mean(dim=-1)
    m = mask.unsqueeze(1).to(dtype=h.dtype)
    return (h * m).sum(dim=-1) / m.sum(dim=-1).clamp(min=1.0)


__all__ = ["masked_global_mean_pool"]
