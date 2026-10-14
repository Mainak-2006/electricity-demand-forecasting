"""LSTM forecasting model (comparison baseline to the GRU).

Identical topology to GRUForecaster but with LSTM cells, so the two can be
compared fairly under the same training protocol.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LSTMForecaster(nn.Module):
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
        self.lstm1 = nn.LSTM(input_size, hidden1, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(hidden1, hidden2, batch_first=True)
        self.fc1 = nn.Linear(hidden2, dense)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(dense, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm1(x)
        out = self.dropout(out)
        out, _ = self.lstm2(out)
        last = out[:, -1, :]
        hidden = self.relu(self.fc1(last))
        return self.fc2(hidden)
