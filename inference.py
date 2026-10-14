"""BijliPredict AI - shared model loading, device selection and batching.

Keeps `train.py`, `evaluate.py` and `predict.py` from each re-implementing the
same inference helpers.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

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
