"""Tests for the synthetic data generator (weather coupling + households)."""

from __future__ import annotations

import pandas as pd
import pytest

from generate_data import generate_households, generate_series, generate_temperature


def test_temperature_shape_and_coverage():
    temp = generate_temperature(start="2023-01-01", days=365, seed=42)
    assert list(temp.columns) == ["temp_mean", "temp_max", "temp_min"]
    assert temp.index.name == "date"
    assert len(temp) == 365
    assert (temp["temp_max"] > temp["temp_min"]).all()
    assert temp["temp_mean"].between(-10, 50).all()


def test_temperature_peaks_in_summer():
    temp = generate_temperature(start="2023-01-01", days=365, seed=42)
    summer = temp.loc["2023-06-01":"2023-07-31", "temp_mean"].mean()
    winter = temp.loc["2023-12-01":"2023-12-31", "temp_mean"].mean()
    assert summer > winter + 10.0


def test_coupled_load_rises_in_summer():
    temp = generate_temperature(start="2023-01-01", days=3 * 365, seed=7)
    series = generate_series(start="2023-01-01", days=3 * 365, seed=7, temperature=temp)
    series = series.assign(date=pd.to_datetime(series["date"])).set_index("date")
    summer = series.loc["2023-06-01":"2023-07-31", "consumption_kwh"].mean()
    winter = series.loc["2023-01-01":"2023-01-31", "consumption_kwh"].mean()
    assert summer > winter


def test_uncoupled_load_still_positive():
    series = generate_series(start="2023-01-01", days=200, seed=3, couple_weather=False)
    assert (series["consumption_kwh"] > 0).all()


def test_coupling_requires_full_temperature():
    temp = generate_temperature(start="2023-01-01", days=100, seed=1)
    with pytest.raises(ValueError, match="temperature does not cover"):
        generate_series(start="2023-01-01", days=200, seed=1, temperature=temp)


def test_single_household_is_wide():
    df = generate_households(days=200, n_households=1, seed=5)
    assert list(df.columns) == ["date", "consumption_kwh"]
    assert "household" not in df.columns


def test_multiple_households_are_long():
    df = generate_households(days=200, n_households=3, seed=5)
    assert list(df.columns) == ["date", "household", "consumption_kwh"]
    assert sorted(df["household"].unique()) == ["h01", "h02", "h03"]
    per_home = df.groupby("household").size()
    assert per_home.nunique() == 1
    assert per_home.iloc[0] == 200


def test_rejects_zero_households():
    with pytest.raises(ValueError, match="n_households"):
        generate_households(days=30, n_households=0)
