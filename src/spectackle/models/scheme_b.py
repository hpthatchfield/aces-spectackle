### CountNet1D – Scheme B regression model
import torch.nn as nn


class CountNet1D(nn.Module):
    """Scheme B: regression head → scalar K."""

    def __init__(self, width=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, width, 9, padding=4),
            nn.ReLU(),
            nn.Conv1d(width, width, 9, padding=4),
            nn.ReLU(),
            nn.Conv1d(width, width, 9, padding=4),
            nn.ReLU(),
        )
        self.head = nn.Linear(width, 1)

    def forward(self, x):
        h = self.conv(x.unsqueeze(1))
        h = h.mean(dim=-1)
        return self.head(h).squeeze(-1)


class CountNet1DDeep(nn.Module):
    """Deeper Scheme B variant: more blocks, BatchNorm, 2-layer head."""

    def __init__(self, width=96, n_blocks=6):
        super().__init__()
        blocks = []
        for i in range(n_blocks):
            in_ch = 1 if i == 0 else width
            blocks.extend([
                nn.Conv1d(in_ch, width, 9, padding=4),
                nn.BatchNorm1d(width),
                nn.ReLU(),
            ])
        self.conv = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.Linear(width, width // 2),
            nn.ReLU(),
            nn.Linear(width // 2, 1),
        )

    def forward(self, x):
        h = self.conv(x.unsqueeze(1))
        h = h.mean(dim=-1)
        return self.head(h).squeeze(-1)
