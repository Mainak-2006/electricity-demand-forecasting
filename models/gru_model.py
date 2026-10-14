"""GRU forecasting model (primary architecture).

Input (30-day window)
    -> GRU (64)
    -> Dropout (0.2)
    -> GRU (32)
    -> Dense (32, ReLU)
    -> Dense (7)          -> 7-day forecast
"""

from __future__ import annotations

import torch
import torch.nn as nn


class GRUForecaster(nn.Module):
    def __init__(
        self,
        input_size: int = 1,
        hidden1: int = 64,
        hidden2: int = 32,
        dropout: float = 0.2,
        dense: int = 32,
        output_size: int = 7,
    ) -> None:
        super().__init__()
        self.gru1 = nn.GRU(input_size, hidden1, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.gru2 = nn.GRU(hidden1, hidden2, batch_first=True)
        self.fc1 = nn.Linear(hidden2, dense)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(dense, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_size)
        out, _ = self.gru1(x)
        out = self.dropout(out)
        out, _ = self.gru2(out)
        last = out[:, -1, :]  # final time step
        hidden = self.relu(self.fc1(last))
        return self.fc2(hidden)  # (batch, output_size)
