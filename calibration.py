"""BijliPredict AI - conformal prediction intervals.

The 7-day model is a direct (fixed-horizon) multi-output forecaster, so we
calibrate one interval width **per horizon step** using residuals from the
held-out **validation** fold:

    s_{i,h} = |y_{i,h} - yhat_{i,h}| / sigma(yhat_{i,h})   (i = window, h = 1..H)

For a nominal level ``1 - alpha`` the split-conformal quantile is the
``ceil((n + 1) * (1 - alpha))``-th smallest score (finite-sample valid under
exchangeability). This needs no distributional assumption and no retraining.

The normalizer ``sigma`` controls how the interval half-width scales with the
forecast level:

* ``absolute`` -- ``sigma = 1``; a constant half-width (classic split conformal).
* ``power``    -- ``sigma = yhat ** gamma``; household load error grows
  **sub-linearly** with the level, so ``gamma = 0.5`` (variance-stabilising)
  keeps interval coverage stable across low- and high-consumption seasons. This
  is the default.

Beyond the calibrated horizon H (the 90-day rollout), the per-step quantile is
extrapolated as ``q_H * sqrt(h / H)`` because the variance of an h-step-ahead
error grows roughly linearly in h.

Everything is stored as plain JSON (``artifacts/calibration.json``) so
`evaluate.py` (writes it) and `predict.py` / `report.py` (read it) never import
sklearn objects.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

DEFAULT_LEVELS = (0.80, 0.90)
DEFAULT_METHOD = "split-conformal-normalized"
DEFAULT_SCALE = "power"
DEFAULT_GAMMA = 0.5
DEFAULT_CALIBRATION_PATH = "artifacts/calibration.json"
_MIN_SCALE = 1e-6


def _normalizer(pred, scale: str, gamma: float):
    """Per-observation scale ``sigma(yhat)`` used to normalize residuals."""
    pred = np.asarray(pred, dtype=float)
    if scale == "absolute":
        return np.ones_like(pred)
    if scale == "power":
        return np.maximum(pred, _MIN_SCALE) ** float(gamma)
    raise ValueError(f"unknown scale {scale!r}; expected 'absolute' or 'power'")


def conformal_quantile(residuals, level: float) -> float:
    """Finite-sample split-conformal quantile of residual scores."""
    resid = np.asarray(residuals, dtype=float).reshape(-1)
    resid = resid[~np.isnan(resid)]
    n = resid.size
    if n == 0:
        raise ValueError("No calibration residuals provided.")
    level = float(level)
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must be in (0, 1), got {level}")
    # Conservative order statistic: ceil((n + 1) * level), clamped to n.
    k = max(1, min(int(np.ceil((n + 1) * level)), int(n)))
    return float(np.sort(resid)[k - 1])


def fit(
    y_true,
    y_pred,
    levels=DEFAULT_LEVELS,
    method: str = DEFAULT_METHOD,
    scale: str = DEFAULT_SCALE,
    gamma: float = DEFAULT_GAMMA,
) -> dict:
    """Fit per-horizon normalized-residual quantiles.

    ``y_true`` and ``y_pred`` are matching ``(n_windows, horizon)`` arrays in
    kWh. Scores are ``|y_true - y_pred| / sigma(y_pred)`` (see module docstring).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.ndim != 2 or y_true.shape != y_pred.shape or y_true.shape[0] == 0:
        raise ValueError("y_true and y_pred must be matching non-empty 2-D arrays")
    if scale not in ("absolute", "power"):
        raise ValueError(f"unknown scale {scale!r}; expected 'absolute' or 'power'")
    scores = np.abs(y_true - y_pred) / _normalizer(y_pred, scale, gamma)
    horizon = int(y_true.shape[1])
    levels = tuple(float(level) for level in levels)
    return {
        "method": method,
        "scale": scale,
        "gamma": float(gamma),
        "horizon": horizon,
        "n_cal": int(y_true.shape[0]),
        "levels": {
            f"{level:.2f}": [conformal_quantile(scores[:, h], level) for h in range(horizon)]
            for level in levels
        },
    }


def widths(calib: dict, level: float, steps: int) -> np.ndarray:
    """Per-step quantile ``q_h`` (before applying the ``sigma`` normalizer).

    Steps within the calibrated horizon use the fitted per-step quantile; later
    steps grow from the last fitted quantile as ``sqrt(step / horizon)``.
    """
    key = f"{float(level):.2f}"
    if key not in calib.get("levels", {}):
        raise KeyError(
            f"Calibration has no level {key}; available: {list(calib.get('levels', {}))}"
        )
    per = np.asarray(calib["levels"][key], dtype=float)
    horizon = per.size
    steps = int(steps)
    out = np.empty(steps, dtype=float)
    take = min(steps, horizon)
    out[:take] = per[:take]
    if steps > horizon:
        last = per[-1]
        h = np.arange(horizon + 1, steps + 1, dtype=float)  # 1-indexed days ahead
        out[horizon:] = last * np.sqrt(h / horizon)
    return out


def apply(pred, calib: dict, level: float, steps: int | None = None):
    """Return ``(lower, upper)`` arrays for ``pred`` at the given level.

    ``pred`` may be 1-D (a single rolling projection) or 2-D (n windows x H).
    The half-width is ``q_h * sigma(pred)``; lower bounds are clipped at 0
    because consumption is non-negative.
    """
    pred = np.asarray(pred, dtype=float)
    if pred.ndim == 0:
        pred = pred.reshape(1)
    if steps is None:
        steps = pred.shape[-1]
    w = widths(calib, level, steps)
    norm = _normalizer(pred, calib.get("scale", "absolute"), calib.get("gamma", 0.0))
    if pred.ndim == 1:
        w = w[: pred.shape[0]]
    half = w * norm
    lower = np.clip(pred - half, 0.0, None)
    upper = pred + half
    return lower, upper


def empirical_coverage(y_true, lower, upper) -> float:
    """Fraction of observations inside ``[lower, upper]`` (element-wise)."""
    y_true = np.asarray(y_true, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    inside = (y_true >= lower) & (y_true <= upper)
    return float(np.mean(inside))


def coverage_report(calib: dict, y_true, y_pred) -> dict:
    """Empirical vs. nominal coverage for every calibrated level (2-D inputs)."""
    y_true = np.asarray(y_true, dtype=float)
    horizon = y_true.shape[-1]
    report: dict[str, dict] = {}
    for key in calib.get("levels", {}):
        level = float(key)
        lower, upper = apply(y_pred, calib, level, steps=horizon)
        report[key] = {
            "target": level,
            "empirical": empirical_coverage(y_true, lower, upper),
            "mean_width": float(np.mean(np.asarray(upper) - np.asarray(lower))),
        }
    return report


def save(calib: dict, path: str | Path = DEFAULT_CALIBRATION_PATH) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(calib, indent=2))
    return path


def load(path: str | Path = DEFAULT_CALIBRATION_PATH) -> dict:
    return json.loads(Path(path).read_text())
