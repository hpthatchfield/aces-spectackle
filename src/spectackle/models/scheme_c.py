### CountNet1D_Classify - Scheme C classification model
import torch.nn as nn

from spectackle.models.pooling import masked_global_mean_pool


class CountNet1DDeep_Classify(nn.Module):
    """
    Scheme C with same conv backbone as CountNet1DDeep (Scheme B).
    Use for parity sweeps vs deep B on identical data.
    """

    def __init__(self, Kmax: int, width: int = 96, n_blocks: int = 6):
        super().__init__()
        self.Kmax = int(Kmax)
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
            nn.Linear(width // 2, self.Kmax + 1),
        )

    def forward(self, x, mask=None):
        h = self.conv(x.unsqueeze(1))
        h = masked_global_mean_pool(h, mask)
        return self.head(h)


class CountNet1D_Classify(nn.Module):
    """Scheme C: classification head -> logits for 0..Kmax. Conv blocks use BatchNorm."""

    def __init__(self, Kmax: int, width: int = 64):
        super().__init__()
        self.Kmax = int(Kmax)
        self.conv = nn.Sequential(
            nn.Conv1d(1, width, 9, padding=4),
            nn.BatchNorm1d(width),
            nn.ReLU(),
            nn.Conv1d(width, width, 9, padding=4),
            nn.BatchNorm1d(width),
            nn.ReLU(),
            nn.Conv1d(width, width, 9, padding=4),
            nn.BatchNorm1d(width),
            nn.ReLU(),
        )
        self.head = nn.Linear(width, self.Kmax + 1)

    def forward(self, x, mask=None):
        h = self.conv(x.unsqueeze(1))
        h = masked_global_mean_pool(h, mask)
        return self.head(h)
