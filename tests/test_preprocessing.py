"""Tests for loading, cleaning, splitting and windowing."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from preprocessing import (
    chronological_split,
    clean_series,
    load_and_prepare,
    load_consumption,
    make_windows,
)


def _write_series(path, n=400, start="2023-01-01", seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=n, freq="D")
    vals = np.round(3.0 + rng.normal(0, 0.2, n), 3)
    pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "consumption_kwh": vals}).to_csv(
        path, index=False
    )
    return dates, vals


def test_make_windows_shapes():
    X, y = make_windows(np.arange(10, dtype=float), input_len=3, horizon=2)
    assert X.shape == (6, 3, 1)
    assert y.shape == (6, 2)
    np.testing.assert_allclose(X[0, :, 0], [0, 1, 2])
    np.testing.assert_allclose(y[0], [3, 4])


def test_make_windows_too_short():
    with pytest.raises(ValueError):
        make_windows(np.arange(3, dtype=float), input_len=5, horizon=7)


def test_chronological_split_sizes_and_order():
    idx = pd.date_range("2023-01-01", periods=100)
    df = pd.DataFrame({"consumption_kwh": np.arange(100.0)}, index=idx)
    frames = chronological_split(df)
    assert len(frames["train"]) == 70
    assert len(frames["val"]) == 15
    assert len(frames["test"]) == 15
    assert frames["train"].index.max() < frames["val"].index.min()
    assert frames["val"].index.max() < frames["test"].index.min()


def test_chronological_split_bad_ratios():
    with pytest.raises(ValueError):
        chronological_split(pd.DataFrame({"consumption_kwh": [1.0] * 10}), 0.5, 0.2, 0.2)


def test_clean_series_fills_gap():
    idx = pd.date_range("2023-01-01", periods=5)
    df = pd.DataFrame({"consumption_kwh": [1.0, np.nan, 3.0, 4.0, 5.0]}, index=idx)
    out = clean_series(df)
    assert len(out) == 5
    assert not out["consumption_kwh"].isna().any()
    assert out["consumption_kwh"].iloc[1] == pytest.approx(2.0)


def test_clean_series_dedupes_and_drops_negative():
    idx = pd.to_datetime(["2023-01-01", "2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04"])
    df = pd.DataFrame({"consumption_kwh": [2.0, 4.0, -1.0, 5.0, 6.0]}, index=idx)
    out = clean_series(df)
    # Duplicate timestamp averaged, negative row dropped, gap interpolated.
    assert out.loc["2023-01-01", "consumption_kwh"] == pytest.approx(3.0)
    assert out.loc["2023-01-02", "consumption_kwh"] == pytest.approx(4.0)


def test_scaler_fit_on_train_only(tmp_path):
    path = tmp_path / "series.csv"
    _, vals = _write_series(path, n=400, seed=1)
    prepared = load_and_prepare(path, input_len=30, horizon=7)
    n_train = int(np.floor(400 * 0.70))
    # The scaler must see only the training segment -> classic leakage guard.
    assert prepared["scaler"].mean_[0] == pytest.approx(vals[:n_train].mean())
    assert len(prepared["datasets"]["test"]["X"]) > 0
    assert prepared["frames"]["train"].index.max() < prepared["frames"]["test"].index.min()


def test_load_consumption_validates_by_default(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame(
        {"date": ["2023-01-01", "oops", "2023-01-03"], "consumption_kwh": [3.0, 3.1, 2.9]}
    ).to_csv(path, index=False)
    with pytest.raises(ValueError, match="unparseable date"):
        load_consumption(path, validate=True)
    # validate=False preserves the old permissive behaviour (bad date dropped).
    out = load_consumption(path, validate=False)
    assert len(out) == 2


def test_load_consumption_attaches_report(tmp_path):
    path = tmp_path / "warn.csv"
    pd.DataFrame(
        {"date": ["2023-01-01", "2023-01-02", "2023-01-03"], "consumption_kwh": [3.0, -1.0, 2.9]}
    ).to_csv(path, index=False)
    out = load_consumption(path, validate=True)
    report = out.attrs["validation"]
    assert report["ok"] is True
    assert report["n_negative"] == 1
