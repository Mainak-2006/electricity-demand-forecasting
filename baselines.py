"""BijliPredict AI - classical statistical baselines (SARIMAX and ETS).

These complement the naive baselines already in `evaluate.py` with two standard
univariate time-series models. `statsmodels` is an **optional** dependency
(install with ``pip install .[stats]``); importing this module never fails, and
the forecast helpers raise a clear message if statsmodels is missing.

Public API:
    statsmodels_available()  -> bool
    sarimax_forecast(history, horizon, ...)  -> np.ndarray
    ets_forecast(history, horizon, ...)      -> np.ndarray
    forecast_block(history, horizon, method) -> np.ndarray
    rolling_block(history, test, input_len, horizon, method) -> np.ndarray
"""

from __future__ import annotations

import importlib

import numpy as np

try:  # optional dependency
    importlib.import_module("statsmodels")

    HAVE_STATSMODELS = True
except Exception:  # pragma: no cover - exercised only without the extra
    HAVE_STATSMODELS = False

DEFAULT_SARIMAX_ORDER = (1, 1, 1)
DEFAULT_SARIMAX_SEASONAL = (1, 0, 0, 7)
DEFAULT_SEASONAL_PERIODS = 7


def statsmodels_available() -> bool:
    """True when the optional `stats` extra (statsmodels) is importable."""
    return HAVE_STATSMODELS


def _require_statsmodels() -> None:
    if not HAVE_STATSMODELS:
        raise ImportError(
            "statsmodels is required for the SARIMAX/ETS baselines; install it "
            "with `pip install .[stats]` (or `uv pip install statsmodels`)."
        )


def _clip_nonneg(values: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=float), 0.0, None)


def sarimax_forecast(
    history,
    horizon: int,
    order: tuple[int, int, int] = DEFAULT_SARIMAX_ORDER,
    seasonal_order: tuple[int, int, int, int] = DEFAULT_SARIMAX_SEASONAL,
) -> np.ndarray:
    """Fit SARIMAX on `history` and forecast `horizon` steps ahead.

    Uses a weekly seasonal term by default and falls back to a non-seasonal
    model when the history is too short to support it. Consumption is clipped at
    zero.
    """
    _require_statsmodels()
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    y = np.asarray(history, dtype=float).reshape(-1)
    if len(y) < 2:
        raise ValueError("Need at least 2 observations to fit SARIMAX.")
    if len(y) < 2 * seasonal_order[3] + 2:
        seasonal_order = (0, 0, 0, 0)
    model = SARIMAX(
        y,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fit = model.fit(disp=False, maxiter=100)
    return _clip_nonneg(fit.get_forecast(steps=horizon).predicted_mean)


def ets_forecast(
    history,
    horizon: int,
    seasonal_periods: int = DEFAULT_SEASONAL_PERIODS,
    trend: str | None = None,
) -> np.ndarray:
    """Fit Holt-Winters (ETS) on `history` and forecast `horizon` steps ahead.

    Adds a seasonal component when the history is long enough, otherwise fits a
    plain (optionally trended) exponential smoothing model.
    """
    _require_statsmodels()
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    y = np.asarray(history, dtype=float).reshape(-1)
    if len(y) < 2:
        raise ValueError("Need at least 2 observations to fit ETS.")
    seasonal = "add" if len(y) >= 2 * seasonal_periods else None
    model = ExponentialSmoothing(
        y,
        trend=trend,
        seasonal=seasonal,
        seasonal_periods=seasonal_periods if seasonal else None,
        initialization_method="estimated",
    )
    fit = model.fit()
    return _clip_nonneg(fit.forecast(horizon))


_BASELINES = {"sarimax": sarimax_forecast, "ets": ets_forecast}


def forecast_block(history, horizon: int, method: str) -> np.ndarray:
    """Dispatch to a named statistical baseline (``"sarimax"`` or ``"ets"``)."""
    if method not in _BASELINES:
        raise ValueError(f"Unknown baseline '{method}' (choose from {sorted(_BASELINES)}).")
    return _BASELINES[method](history, horizon)


def _n_windows(n_test: int, input_len: int, horizon: int) -> int:
    n = n_test - input_len - horizon + 1
    if n <= 0:
        raise ValueError(
            f"Test series too short ({n_test}) for input_len={input_len} + horizon={horizon}."
        )
    return n


def sarimax_rolling_forecast(
    history,
    test_values,
    input_len: int,
    horizon: int,
    order: tuple[int, int, int] = DEFAULT_SARIMAX_ORDER,
    seasonal_order: tuple[int, int, int, int] = DEFAULT_SARIMAX_SEASONAL,
) -> np.ndarray:
    """Rolling ``horizon``-step SARIMAX forecasts for every test window.

    The model is fit once on `history`; its state is then advanced with each
    observed test value (``refit=False``) so window ``i`` forecasts exactly the
    days the neural model's window ``i`` is scored on. Returns ``(n_windows,
    horizon)``.
    """
    _require_statsmodels()
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    hist = np.asarray(history, dtype=float).reshape(-1)
    test = np.asarray(test_values, dtype=float).reshape(-1)
    if len(hist) < 2:
        raise ValueError("Need at least 2 observations of history to fit SARIMAX.")
    if len(hist) < 2 * seasonal_order[3] + 2:
        seasonal_order = (0, 0, 0, 0)
    n_windows = _n_windows(len(test), input_len, horizon)
    result = SARIMAX(
        hist,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False, maxiter=100)

    result = result.append(test[:input_len], refit=False)
    out = np.empty((n_windows, horizon), dtype=float)
    for i in range(n_windows):
        out[i] = result.get_forecast(steps=horizon).predicted_mean
        nxt = input_len + i
        if nxt < len(test):
            result = result.append(test[nxt : nxt + 1], refit=False)
    return _clip_nonneg(out)


def ets_rolling_forecast(
    history,
    test_values,
    input_len: int,
    horizon: int,
    seasonal_periods: int = DEFAULT_SEASONAL_PERIODS,
    trend: str | None = None,
) -> np.ndarray:
    """Rolling ``horizon``-step Holt-Winters (ETS) forecasts for every window.

    ETS results do not support incremental updating, so the model is refit on
    each window's context (history plus the observations seen so far). Fitting is
    cheap (~20ms), so this stays well under a second for a few hundred windows.
    Returns ``(n_windows, horizon)``.
    """
    _require_statsmodels()
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    hist = np.asarray(history, dtype=float).reshape(-1)
    test = np.asarray(test_values, dtype=float).reshape(-1)
    if len(hist) < 2:
        raise ValueError("Need at least 2 observations of history to fit ETS.")
    n_windows = _n_windows(len(test), input_len, horizon)
    out = np.empty((n_windows, horizon), dtype=float)
    for i in range(n_windows):
        y = np.concatenate([hist, test[: input_len + i]])
        seasonal = "add" if len(y) >= 2 * seasonal_periods else None
        model = ExponentialSmoothing(
            y,
            trend=trend,
            seasonal=seasonal,
            seasonal_periods=seasonal_periods if seasonal else None,
            initialization_method="estimated",
        )
        out[i] = model.fit().forecast(horizon)
    return _clip_nonneg(out)


_ROLLING = {"sarimax": sarimax_rolling_forecast, "ets": ets_rolling_forecast}


def rolling_block(history, test_values, input_len: int, horizon: int, method: str) -> np.ndarray:
    """Dispatch to a rolling statistical baseline (``"sarimax"`` or ``"ets"``)."""
    if method not in _ROLLING:
        raise ValueError(f"Unknown baseline '{method}' (choose from {sorted(_ROLLING)}).")
    return _ROLLING[method](history, test_values, input_len, horizon)
