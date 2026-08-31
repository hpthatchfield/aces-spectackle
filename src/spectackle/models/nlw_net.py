### NLW Finder - Scheme-B-style backbone, single logit for BCEWithLogitsLoss
import torch.nn as nn

from spectackle.models.pooling import masked_global_mean_pool


class NLWNet1DDeep(nn.Module):
    """Same conv stack pattern as CountNet1DDeep; head outputs one logit per spectrum."""

    def __init__(self, width: int = 96, n_blocks: int = 6):
        super().__init__()
        blocks = []
        for i in range(n_blocks):
            in_ch = 1 if i == 0 else width
            blocks.extend(
                [
                    nn.Conv1d(in_ch, width, 9, padding=4),
                    nn.BatchNorm1d(width),
                    nn.ReLU(),
                ]
            )
        self.conv = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.Linear(width, width // 2),
            nn.ReLU(),
            nn.Linear(width // 2, 1),
        )

    def forward(self, x, mask=None):
        ### x: (B, C). Optional mask: (B, C) with 1=valid channel, 0=padded (NaN/zero).
        ### Masked global average pool -> pooled feature is invariant to how much of the axis
        ### is padded (and where), so the model judges only the real channels.
        h = self.conv(x.unsqueeze(1))
        h = masked_global_mean_pool(h, mask)
        return self.head(h).squeeze(-1)
