"""Tests for the Open-Meteo weather client (no network: a fake fetcher is used)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import weather


def make_fake_fetcher(today="2024-06-15"):
    """Return (fetcher, calls). The fetcher serves both archive and forecast."""
    calls: list[tuple[str, dict]] = []
    today_ts = pd.Timestamp(today)

    def fetcher(url, params):
        calls.append((url, dict(params)))
        if url == weather.ARCHIVE_URL:
            times = pd.date_range(params["start_date"], params["end_date"], freq="D")
        else:
            past = int(params.get("past_days", 0))
            ahead = int(params.get("forecast_days", 16))
            times = pd.date_range(
                today_ts - pd.Timedelta(days=past), periods=past + ahead, freq="D"
            )
        mean = [20.0 + 10.0 * np.sin(2.0 * np.pi * d.dayofyear / 365.25) for d in times]
        return {
            "daily": {
                "time": [str(t.date()) for t in times],
                "temperature_2m_mean": mean,
                "temperature_2m_max": [m + 4.0 for m in mean],
                "temperature_2m_min": [m - 4.0 for m in mean],
            }
        }

    return fetcher, calls


def test_resolve_city_canonical_and_alias():
    assert weather.resolve_city("Delhi")[0] == "delhi"
    assert weather.resolve_city("new delhi")[0] == "delhi"
    assert weather.resolve_city("bangalore")[0] == "bengaluru"
    lat, lon = weather.resolve_city("mumbai")[1:]
    assert 18.0 < lat < 20.0 and 72.0 < lon < 74.0


def test_resolve_city_unknown_raises():
    with pytest.raises(ValueError):
        weather.resolve_city("atlantis")


def test_parse_daily_coerces_and_indexes():
    payload = {
        "daily": {
            "time": ["2024-01-01", "2024-01-02"],
            "temperature_2m_mean": [12.0, 13.0],
            "temperature_2m_max": [16.0, 17.0],
            "temperature_2m_min": [8.0, 9.0],
        }
    }
    df = weather._parse_daily(payload)
    assert list(df.index) == [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")]
    assert list(df.columns) == weather.TEMP_COLUMNS


def test_fetch_uses_cache_on_second_call(tmp_path):
    fetcher, calls = make_fake_fetcher()
    df1 = weather.fetch(
        "delhi", "2024-06-01", "2024-06-10", cache_dir=tmp_path, fetcher=fetcher, today="2024-06-15"
    )
    n_after_first = len(calls)
    assert len(df1.dropna()) == 10
    assert n_after_first >= 1  # archive (+ forecast) fetched

    df2 = weather.fetch(
        "delhi", "2024-06-01", "2024-06-10", cache_dir=tmp_path, fetcher=fetcher, today="2024-06-15"
    )
    assert len(calls) == n_after_first  # fully served from cache
    pd.testing.assert_frame_equal(df1, df2)
    assert (tmp_path / "delhi.csv").exists()


def test_get_temperature_fills_far_future_from_climatology(tmp_path):
    fetcher, _ = make_fake_fetcher()
    dates = pd.date_range("2024-06-15", "2024-09-01", freq="D")
    temps = weather.get_temperature(
        "delhi", dates, cache_dir=tmp_path, fetcher=fetcher, today="2024-06-15"
    )
    assert list(temps.index) == list(dates)
    assert not temps["temp_mean"].isna().any()
    # The forecast can only reach ~16 days out, so late dates came from climatology.
    assert temps.loc["2024-09-01", "temp_mean"] > -50.0


def test_weather_file_bypasses_network(tmp_path):
    csv = tmp_path / "fixed.csv"
    pd.DataFrame(
        {
            "date": pd.date_range("2024-06-01", periods=5, freq="D").strftime("%Y-%m-%d"),
            "temp-mean": [30.0, 31.0, 32.0, 33.0, 34.0],
        }
    ).to_csv(csv, index=False)

    def boom(*_a, **_k):
        raise AssertionError("network must not be used when weather_file is given")

    dates = pd.date_range("2024-06-01", periods=5, freq="D")
    temps = weather.get_temperature("delhi", dates, weather_file=csv, fetcher=boom)
    assert temps["temp_mean"].tolist() == [30.0, 31.0, 32.0, 33.0, 34.0]
    # max/min fall back to mean when only temp-mean is provided.
    assert temps["temp_max"].tolist() == temps["temp_mean"].tolist()


def test_load_weather_derives_mean_from_min_max(tmp_path):
    csv = tmp_path / "mm.csv"
    pd.DataFrame(
        {
            "date": ["2024-06-01", "2024-06-02"],
            "temp_max": [34.0, 36.0],
            "temp_min": [24.0, 26.0],
        }
    ).to_csv(csv, index=False)
    df = weather.load_weather(csv)
    assert df["temp_mean"].tolist() == [29.0, 31.0]


def test_climatology_is_day_of_year_mean(tmp_path):
    fetcher, _ = make_fake_fetcher()
    clim = weather.get_climatology(
        "delhi", cache_dir=tmp_path, fetcher=fetcher, years=2, today="2024-06-15"
    )
    assert clim.index.name == "doy"
    assert set(weather.TEMP_COLUMNS).issubset(clim.columns)
    assert len(clim) >= 365
    assert clim["temp_mean"].between(-60.0, 60.0).all()
