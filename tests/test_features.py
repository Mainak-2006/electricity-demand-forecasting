"""Tests for feature engineering: calendar, holidays, temperature, config."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from features import (
    CALENDAR_COLUMNS,
    FeatureConfig,
    build_feature_matrix,
    calendar_features,
    holiday_features,
    stack_window,
    temperature_features,
    temperature_for_dates,
)


def _temperature(dates, base=25.0) -> pd.DataFrame:
    idx = pd.DatetimeIndex(pd.to_datetime(list(dates)))
    mean = base + 10.0 * np.sin(2.0 * np.pi * idx.dayofyear.to_numpy() / 365.25)
    return pd.DataFrame(
        {"temp_mean": mean, "temp_max": mean + 4.0, "temp_min": mean - 4.0}, index=idx
    )


def test_calendar_features_columns_and_weekend():
    dates = pd.date_range("2024-01-01", periods=7, freq="D")  # Mon..Sun
    df = calendar_features(dates)
    assert list(df.columns) == CALENDAR_COLUMNS
    assert df["is_weekend"].tolist() == [0, 0, 0, 0, 0, 1, 1]
    assert df["dow_sin"].between(-1.0, 1.0).all()
    assert df["month_sin"].between(-1.0, 1.0).all()


def test_holiday_features_flags_indian_holidays():
    dates = pd.to_datetime(["2024-01-26", "2024-01-27"])  # Republic Day, then normal
    df = holiday_features(dates, country="IN")
    assert df["is_holiday"].tolist() == [1.0, 0.0]


def test_temperature_features_lags_roll_and_cdd():
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    temp = _temperature(dates, base=25.0)
    cfg = FeatureConfig(temperature=True, temp_lags=(1, 2), temp_roll=3, cdd_base=24.0)
    df = temperature_features(temp, cfg)
    assert not df.isna().any().any()
    # lag1 at day t equals mean at t-1
    assert df["temp_lag1"].iloc[3] == pytest.approx(temp["temp_mean"].iloc[2])
    assert df["temp_lag2"].iloc[3] == pytest.approx(temp["temp_mean"].iloc[1])
    assert df["cdd"].iloc[0] == pytest.approx(max(0.0, temp["temp_mean"].iloc[0] - 24.0))


def test_config_round_trip_and_buffer():
    cfg = FeatureConfig(
        calendar=True, holidays=True, temperature=True, city="mumbai", temp_lags=(1, 2)
    )
    restored = FeatureConfig.from_dict(cfg.to_dict())
    assert restored == cfg
    assert cfg.buffer_days == 7
    assert FeatureConfig().from_dict({}) == FeatureConfig()


def test_n_exog_matches_built_columns():
    dates = pd.date_range("2024-03-01", periods=40, freq="D")
    temp = _temperature(dates)
    for cfg in (
        FeatureConfig(calendar=True),
        FeatureConfig(calendar=True, holidays=True),
        FeatureConfig(calendar=True, holidays=True, temperature=True),
    ):
        matrix, cols = build_feature_matrix(dates, temperature=temp, config=cfg)
        assert len(cols) == cfg.n_exog()
        assert matrix.shape == (len(dates), cfg.n_exog())
        assert list(matrix.index) == list(dates)


def test_disabled_config_yields_no_features():
    dates = pd.date_range("2024-03-01", periods=5, freq="D")
    matrix, cols = build_feature_matrix(dates, config=FeatureConfig())
    assert cols == []
    assert matrix.empty


def test_temperature_requested_without_data_raises():
    dates = pd.date_range("2024-03-01", periods=5, freq="D")
    with pytest.raises(ValueError):
        build_feature_matrix(dates, config=FeatureConfig(temperature=True))


def test_temperature_not_covering_dates_raises():
    dates = pd.date_range("2024-03-01", periods=5, freq="D")
    short = _temperature(dates[:2])
    with pytest.raises(ValueError):
        build_feature_matrix(dates, temperature=short, config=FeatureConfig(temperature=True))


def test_stack_window_layout_and_channel_zero():
    cons = np.arange(30, dtype=float)
    feats = np.ones((30, 4), dtype=float)
    x = stack_window(cons, feats)
    assert x.shape == (30, 5)
    np.testing.assert_allclose(x[:, 0], cons)
    np.testing.assert_allclose(x[:, 1:], feats)
    # feature-free path keeps the legacy (input_len, 1) layout
    assert stack_window(cons, None).shape == (30, 1)


def _write_consumption(path, n_days=220, start="2024-01-01"):
    dates = pd.date_range(start, periods=n_days, freq="D")
    kwh = 12.0 + 4.0 * np.sin(2.0 * np.pi * np.arange(n_days) / 365.0) + 0.1 * np.arange(n_days)
    pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "consumption_kwh": kwh}).to_csv(
        path, index=False
    )
    return dates


def _write_weather(path, dates):
    idx = pd.date_range(dates.min() - pd.Timedelta(days=10), dates.max(), freq="D")
    mean = 26.0 + 8.0 * np.sin(2.0 * np.pi * idx.dayofyear.to_numpy() / 365.25)
    pd.DataFrame(
        {
            "date": idx.strftime("%Y-%m-%d"),
            "temp_mean": mean,
            "temp_max": mean + 4.0,
            "temp_min": mean - 4.0,
        }
    ).to_csv(path, index=False)


@pytest.mark.parametrize("with_temperature", [False, True])
def test_train_eval_predict_feature_parity(tmp_path, with_temperature):
    """The predict-side input builder must reproduce the evaluate-side X exactly."""
    import torch

    from inference import make_model_input, predict
    from models import get_model
    from preprocessing import load_and_prepare

    data_path = tmp_path / "cons.csv"
    dates = _write_consumption(data_path)
    cfg = FeatureConfig(calendar=True, holidays=True, temperature=with_temperature, city="delhi")
    temperature = None
    if with_temperature:
        weather_path = tmp_path / "weather.csv"
        _write_weather(weather_path, dates)
        temperature = temperature_for_dates(cfg, dates, weather_file=weather_path)

    input_len, horizon = 14, 7
    prepared = load_and_prepare(
        data_path,
        input_len=input_len,
        horizon=horizon,
        splits=(0.7, 0.15, 0.15),
        save_scaler=False,
        feature_config=cfg,
        temperature=temperature,
    )
    X_test = prepared["datasets"]["test"]["X"]
    assert len(X_test) > 0
    assert X_test.shape[2] == 1 + cfg.n_exog()

    torch.manual_seed(0)
    model = get_model(
        "gru",
        input_size=1 + cfg.n_exog(),
        hidden1=8,
        hidden2=4,
        dropout=0.0,
        dense=4,
        output_size=horizon,
    )
    device = torch.device("cpu")

    frames = prepared["frames"]["test"]
    for i in range(len(X_test)):
        raw_cons = frames["consumption_kwh"].to_numpy()[i : i + input_len]
        window_dates = frames.index[i : i + input_len]
        raw_feats = prepared["features"].loc[window_dates].to_numpy()
        x = make_model_input(prepared["scaler"], prepared["feature_scaler"], raw_cons, raw_feats)
        # 1) input parity: predict path == evaluate path
        np.testing.assert_allclose(x[0], X_test[i], rtol=1e-5, atol=1e-5)
        # 2) output parity follows from the shared forward pass
        np.testing.assert_allclose(
            predict(model, x, device),
            predict(model, X_test[i : i + 1], device),
            rtol=1e-5,
            atol=1e-6,
        )
