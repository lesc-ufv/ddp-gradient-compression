from __future__ import annotations

import torch
from torch import nn


class MNISTMLP(nn.Module):
    """MLP intentionally large enough to create multiple DDP buckets with small caps."""

    def __init__(self, hidden_size: int = 512) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def build_model(hidden_size: int) -> nn.Module:
    return MNISTMLP(hidden_size=hidden_size)
