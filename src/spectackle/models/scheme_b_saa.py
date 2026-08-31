### Scheme B with parent SAA spectrum + K_parent conditioning (two-level Scouse-style).
from __future__ import annotations

import torch
import torch.nn as nn

from spectackle.models.pooling import masked_global_mean_pool
from spectackle.models.scheme_d_lite import _build_conv_encoder


class CountNet1DDeepSaaCond(nn.Module):
    """
    Stage-2 pixel K regressor conditioned on parent SAA context.

    Inputs:
      x_pixel: (B, C) normalized pixel spectrum
      parent_spec: (B, C) normalized parent SAA mean spectrum
      k_parent: (B,) integer parent K from stage 1 (0..Kmax)
      mask: optional valid channel mask

    The conv encoder sees two channels [pixel, parent]. k_parent is embedded and added
    to the pooled features before the regression head.
    """

    def __init__(self, *, width: int = 96, n_blocks: int = 6, Kmax: int = 10):
        super().__init__()
        self.Kmax = int(Kmax)
        self.conv = _build_conv_encoder(width, n_blocks, in_channels=2)
        self.k_embed = nn.Embedding(self.Kmax + 1, width)
        self.head = nn.Sequential(
            nn.Linear(width, width // 2),
            nn.ReLU(),
            nn.Linear(width // 2, 1),
        )

    def forward(
        self,
        x_pixel: torch.Tensor,
        parent_spec: torch.Tensor,
        k_parent: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h_in = torch.stack([x_pixel, parent_spec], dim=1)
        h = self.conv(h_in)
        h = masked_global_mean_pool(h, mask)
        kp = k_parent.long().clamp(0, self.Kmax)
        h = h + self.k_embed(kp)
        return self.head(h).squeeze(-1)


__all__ = ["CountNet1DDeepSaaCond"]
