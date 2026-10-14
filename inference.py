"""BijliPredict AI - shared model loading, device selection and batching.

Keeps `train.py`, `evaluate.py` and `predict.py` from each re-implementing the
same inference helpers.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from features import FeatureConfig, stack_window
from models import get_model


def resolve_device(requested: str = "auto") -> torch.device:
    """Turn a CLI device string into a torch device.

    Raises a clear error when CUDA is requested but unavailable, instead of
    silently falling back to CPU.
    """
    requested = (requested or "auto").lower().strip()
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("--device cuda requested but CUDA is not available.")
        return torch.device("cuda")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raise SystemExit(f"Unknown device '{requested}' (choose auto/cpu/cuda).")


def load_model(model_path: str | Path, device: torch.device) -> dict:
    """Load a training checkpoint into an eval-mode model.

    `weights_only=True` restricts unpickling to tensors and plain containers,
    avoiding arbitrary code execution from a tampered artifact.
    """
    ckpt = torch.load(model_path, map_location=device, weights_only=True)
    model = get_model(ckpt["model_type"], **ckpt["model_config"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return {"ckpt": ckpt, "model": model}


def predict(model, X: np.ndarray, device: torch.device, batch_size: int = 256) -> np.ndarray:
    """Batched forward pass over `X`, returning a numpy array on CPU."""
    outs = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[i : i + batch_size]).to(device)
            outs.append(model(xb).cpu().numpy())
    return np.concatenate(outs, axis=0) if outs else np.empty((0,))


def feature_info(ckpt: dict) -> tuple[FeatureConfig, list[str], str | None]:
    """Recover the exogenous-feature metadata persisted in a checkpoint.

    Older checkpoints predate features, so every field defaults to a no-op
    config, an empty column list and no feature scaler.
    """
    config = FeatureConfig.from_dict(ckpt.get("feature_config"))
    columns = list(ckpt.get("feature_columns") or [])
    scaler_path = ckpt.get("feature_scaler_path")
    return config, columns, scaler_path


def make_model_input(
    scaler,
    feature_scaler,
    consumptions: np.ndarray,
    feature_window: np.ndarray | None = None,
) -> np.ndarray:
    """Build a single ``(1, input_len, 1 + n_exog)`` model input tensor.

    `consumptions` are raw kWh (scaled here with the consumption `scaler`);
    `feature_window` is the *raw* exogenous block for the same days (scaled here
    with `feature_scaler`). This is the one place train/evaluate/predict share,
    so the channel layout can never drift between them.
    """
    cons = np.asarray(consumptions, dtype=float).reshape(-1, 1)
    scaled = scaler.transform(cons)
    if feature_window is None or feature_scaler is None:
        feats = None
    else:
        feats = feature_scaler.transform(np.asarray(feature_window, dtype=float))
    return stack_window(scaled, feats)[None].astype(np.float32)
