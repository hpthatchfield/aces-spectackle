### Scheme D-lite: velocity centers (oracle-K and two-stage K + centers).
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from spectackle.models.pooling import masked_global_mean_pool


def _build_conv_encoder(
    width: int,
    n_blocks: int,
    in_channels: int = 1,
    *,
    kernel_size: int = 9,
) -> nn.Sequential:
    """Stacked Conv1d-BN-ReLU. kernel_size odd recommended; padding keeps length C."""
    k = int(kernel_size)
    if k < 1:
        raise ValueError(f"kernel_size must be >= 1, got {k}")
    pad = k // 2
    blocks: list[nn.Module] = []
    for i in range(int(n_blocks)):
        in_ch = int(in_channels) if i == 0 else width
        blocks.extend(
            [
                nn.Conv1d(in_ch, width, k, padding=pad),
                nn.BatchNorm1d(width),
                nn.ReLU(),
            ]
        )
    return nn.Sequential(*blocks)


class OracleCenterNet1DDeep(nn.Module):
    """
    D-lite Phase 1: shared conv encoder + Kmax velocity-center slots.

    Predicts v (km/s) per slot only. K_true / component_valid mask active slots at train time.
    """

    def __init__(self, Kmax: int, width: int = 96, n_blocks: int = 6):
        super().__init__()
        self.Kmax = int(Kmax)
        self.conv = _build_conv_encoder(width, n_blocks)
        self.v_head = nn.Sequential(
            nn.Linear(width, width // 2),
            nn.ReLU(),
            nn.Linear(width // 2, self.Kmax),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        x: (B, C) normalized spectrum.
        Returns (B, Kmax) velocity centers in km/s.
        """
        h = self.conv(x.unsqueeze(1))
        h = masked_global_mean_pool(h, mask)
        return self.v_head(h)


class CenterNet1DDeep(nn.Module):
    """
    D-lite two-stage (single encoder): K head + Kmax velocity-center slots.

    k_mode:
      - "ce"  (default): Scheme C style logits over {0..Kmax}
      - "reg": Scheme B style scalar K (SmoothL1); matches CountNet1DDeep on MOPRA

    Stage 2: v (km/s) per v-ordered slot. v loss uses ground-truth slot mask at train time;
    K is not given to the model.

    Optional coord channel: pass ``coord`` (a length-C normalized velocity axis) to feed
    km/s-per-channel alongside the intensity spectrum. Global mean pooling discards channel
    position, so on a wide velocity axis the v head cannot localize from intensity alone; the
    coord channel lets the pooled feature encode an intensity-weighted velocity (centroid).
    Default off, so Scheme D easy-window runs and prior checkpoints are unchanged.
    """

    def __init__(
        self,
        Kmax: int,
        width: int = 96,
        n_blocks: int = 6,
        coord: torch.Tensor | np.ndarray | None = None,
        k_mode: str = "ce",
    ):
        super().__init__()
        self.Kmax = int(Kmax)
        if k_mode not in ("ce", "reg"):
            raise ValueError(f"k_mode must be 'ce' or 'reg', got {k_mode!r}")
        self.k_mode = str(k_mode)
        self.use_coord = coord is not None
        in_channels = 2 if self.use_coord else 1
        self.conv = _build_conv_encoder(width, n_blocks, in_channels=in_channels)
        if self.use_coord:
            coord_t = torch.as_tensor(np.asarray(coord), dtype=torch.float32).reshape(-1)
            ### Fixed per-run velocity axis (normalized km/s); rides along as input channel 2.
            self.register_buffer("coord", coord_t)
        k_out = 1 if self.k_mode == "reg" else (self.Kmax + 1)
        self.k_head = nn.Sequential(
            nn.Linear(width, width // 2),
            nn.ReLU(),
            nn.Linear(width // 2, k_out),
        )
        self.v_head = nn.Sequential(
            nn.Linear(width, width // 2),
            nn.ReLU(),
            nn.Linear(width // 2, self.Kmax),
        )

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (k_out, v_kms).
        k_out is (B, Kmax+1) logits when k_mode='ce', or (B,) scalar when k_mode='reg'.
        """
        h_in = x.unsqueeze(1)
        if self.use_coord:
            c = self.coord.to(x.dtype).view(1, 1, -1).expand(x.size(0), 1, -1)
            h_in = torch.cat([h_in, c], dim=1)
        h = self.conv(h_in)
        h = masked_global_mean_pool(h, mask)
        k_out = self.k_head(h)
        if self.k_mode == "reg":
            k_out = k_out.squeeze(-1)
        return k_out, self.v_head(h)


__all__ = ["CenterNet1DDeep", "OracleCenterNet1DDeep"]
