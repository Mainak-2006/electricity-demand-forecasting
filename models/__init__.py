"""Model registry for BijliPredict AI."""

from __future__ import annotations

from .gru_model import GRUForecaster
from .lstm_model import LSTMForecaster

MODEL_REGISTRY = {
    "gru": GRUForecaster,
    "lstm": LSTMForecaster,
}


def get_model(name: str, **kwargs):
    """Instantiate a forecasting model by name ('gru' or 'lstm')."""
    key = name.lower().strip()
    if key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Choose from {sorted(MODEL_REGISTRY)}.")
    return MODEL_REGISTRY[key](**kwargs)


__all__ = [
    "GRUForecaster",
    "LSTMForecaster",
    "MODEL_REGISTRY",
    "get_model",
]
