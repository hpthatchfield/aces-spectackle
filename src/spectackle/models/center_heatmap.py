### Center heatmap: per-channel P(component center here) over the velocity axis.
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from spectackle.models.scheme_d_lite import _build_conv_encoder

### Fill value for invalid channels so sigmoid(logit) -> 0 there (masked in loss and decode).
_MASK_FILL = -30.0


class CenterHeatmapNet1DDeep(nn.Module):
    """
    Conv encoder -> per-channel center logits (length-C heatmap).

    Optional ``coord`` adds a normalized velocity channel. Peak count/centers are
    decoded downstream (find_peaks or a K head), not inside this module.
    """

    def __init__(
        self,
        width: int = 96,
        n_blocks: int = 6,
        coord: torch.Tensor | np.ndarray | None = None,
        *,
        kernel_size: int = 9,
    ):
        super().__init__()
        self.use_coord = coord is not None
        self.kernel_size = int(kernel_size)
        in_channels = 2 if self.use_coord else 1
        self.conv = _build_conv_encoder(
            width, n_blocks, in_channels=in_channels, kernel_size=self.kernel_size
        )
        if self.use_coord:
            coord_t = torch.as_tensor(np.asarray(coord), dtype=torch.float32).reshape(-1)
            self.register_buffer("coord", coord_t)
        ### 1x1 conv over channels: (B, W, C) -> (B, 1, C) per-channel center logit.
        self.head = nn.Conv1d(width, 1, kernel_size=1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        x: (B, C) normalized spectrum. mask: optional (B, C), 1=valid channel.
        Returns per-channel center logits (B, C). Invalid channels are forced to a large
        negative so their probability is ~0.
        """
        h_in = x.unsqueeze(1)
        if self.use_coord:
            c = self.coord.to(x.dtype).view(1, 1, -1).expand(x.size(0), 1, -1)
            h_in = torch.cat([h_in, c], dim=1)
        h = self.conv(h_in)
        logits = self.head(h).squeeze(1)
        if mask is not None:
            logits = logits.masked_fill(mask < 0.5, _MASK_FILL)
        return logits


__all__ = ["CenterHeatmapNet1DDeep"]
