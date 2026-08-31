### Stage-2 K head on a Stage-1 center heatmap (heatmap -> learned count).
from __future__ import annotations

import torch
import torch.nn as nn

from spectackle.models.center_heatmap import CenterHeatmapNet1DDeep
from spectackle.models.pooling import masked_global_mean_pool
from spectackle.models.scheme_d_lite import _build_conv_encoder


class HeatmapCountNet(nn.Module):
    """
    Frozen (or tunable) heatmap + scalar K head.

    ``k_input="p"`` uses P(center) only; ``k_input="spec_p"`` uses [spectrum; P].
    Train with SmoothL1 on K. Peak-decode on the heatmap is for comparison only.
    """

    def __init__(
        self,
        heatmap: CenterHeatmapNet1DDeep,
        *,
        width: int = 96,
        n_blocks: int = 6,
        k_input: str = "spec_p",
        freeze_heatmap: bool = True,
        kernel_size: int = 9,
    ):
        super().__init__()
        if k_input not in ("p", "spec_p"):
            raise ValueError(f"k_input must be 'p' or 'spec_p', got {k_input!r}")
        self.heatmap = heatmap
        self.k_input = str(k_input)
        self.freeze_heatmap = bool(freeze_heatmap)
        self.kernel_size = int(kernel_size)
        in_ch = 1 if self.k_input == "p" else 2
        self.count_conv = _build_conv_encoder(
            width, n_blocks, in_channels=in_ch, kernel_size=self.kernel_size
        )
        self.k_head = nn.Sequential(
            nn.Linear(width, width // 2),
            nn.ReLU(),
            nn.Linear(width // 2, 1),
        )
        self.set_freeze_heatmap(self.freeze_heatmap)

    def set_freeze_heatmap(self, freeze: bool) -> None:
        """Freeze or unfreeze Stage-1 heatmap parameters."""
        self.freeze_heatmap = bool(freeze)
        for p in self.heatmap.parameters():
            p.requires_grad = not self.freeze_heatmap

    def heatmap_logits(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Stage-1 center logits (B, C). Uses no_grad when heatmap is frozen."""
        if self.freeze_heatmap:
            with torch.no_grad():
                return self.heatmap(x, mask)
        return self.heatmap(x, mask)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        x: (B, C) normalized spectrum. mask: optional (B, C).
        Returns scalar K_hat (B,).
        """
        logits = self.heatmap_logits(x, mask)
        p = torch.sigmoid(logits)
        if self.k_input == "p":
            h_in = p.unsqueeze(1)
        else:
            h_in = torch.stack([x, p], dim=1)
        h = self.count_conv(h_in)
        h = masked_global_mean_pool(h, mask)
        return self.k_head(h).squeeze(-1)


__all__ = ["HeatmapCountNet"]
