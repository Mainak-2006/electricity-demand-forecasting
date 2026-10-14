"""Tests for the optional SARIMAX / ETS statistical baselines."""

from __future__ import annotations

import numpy as np
import pytest

import baselines

requires_stats = pytest.mark.skipif(
    not baselines.statsmodels_available(), reason="statsmodels (stats extra) not installed"
)
# SARIMAX can fail to fully converge on tiny synthetic series; that is expected.
pytestmark = pytest.mark.filterwarnings(
    "ignore:Maximum Likelihood optimization failed to converge:Warning"
)


def _seasonal_series(n: int = 200, period: int = 7) -> np.ndarray:
    t = np.arange(n, dtype=float)
    return 10.0 + 2.0 * np.sin(2.0 * np.pi * t / period) + 0.1 * t


def test_available_is_bool():
    assert isinstance(baselines.statsmodels_available(), bool)


def test_forecast_block_unknown_method():
    with pytest.raises(ValueError):
        baselines.forecast_block(np.arange(20.0), 7, "no-such-model")


@requires_stats
def test_sarimax_forecast_shape_and_nonneg():
    fc = baselines.sarimax_forecast(_seasonal_series(), 7)
    assert fc.shape == (7,)
    assert np.all(fc >= 0.0)
    assert np.all(np.isfinite(fc))


@requires_stats
def test_ets_forecast_shape_and_nonneg():
    fc = baselines.ets_forecast(_seasonal_series(), 7)
    assert fc.shape == (7,)
    assert np.all(fc >= 0.0)


@requires_stats
def test_ets_short_series_falls_back_to_non_seasonal():
    fc = baselines.ets_forecast(_seasonal_series(n=8), 3)
    assert fc.shape == (3,)


@requires_stats
def test_forecast_block_dispatches():
    fc = baselines.forecast_block(_seasonal_series(), 5, "ets")
    assert fc.shape == (5,)


def test_rolling_block_unknown_method():
    with pytest.raises(ValueError):
        baselines.rolling_block(np.arange(40.0), np.arange(10.0), 5, 2, "nope")


@requires_stats
def test_sarimax_rolling_shape():
    history = _seasonal_series(120)
    test = _seasonal_series(40)
    out = baselines.rolling_block(history, test, input_len=10, horizon=7, method="sarimax")
    assert out.shape == (40 - 10 - 7 + 1, 7)
    assert np.all(out >= 0.0)


@requires_stats
def test_ets_rolling_shape():
    history = _seasonal_series(120)
    test = _seasonal_series(40)
    out = baselines.rolling_block(history, test, input_len=10, horizon=7, method="ets")
    assert out.shape == (40 - 10 - 7 + 1, 7)
    assert np.all(out >= 0.0)


@requires_stats
def test_rolling_short_test_raises():
    with pytest.raises(ValueError):
        baselines.rolling_block(_seasonal_series(120), np.arange(5.0), 10, 7, "ets")
