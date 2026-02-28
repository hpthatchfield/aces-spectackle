### CountNet1D_Classify – Scheme C classification model
import torch.nn as nn


class CountNet1D_Classify(nn.Module):
    """Scheme C: classification head → logits for 0..Kmax."""

    def __init__(self, Kmax: int, width: int = 64):
        super().__init__()
        self.Kmax = int(Kmax)
        self.conv = nn.Sequential(
            nn.Conv1d(1, width, 9, padding=4),
            nn.ReLU(),
            nn.Conv1d(width, width, 9, padding=4),
            nn.ReLU(),
            nn.Conv1d(width, width, 9, padding=4),
            nn.ReLU(),
        )
        self.head = nn.Linear(width, self.Kmax + 1)

    def forward(self, x):
        h = self.conv(x.unsqueeze(1))
        h = h.mean(dim=-1)
        return self.head(h)
