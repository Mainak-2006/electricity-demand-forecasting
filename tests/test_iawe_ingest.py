"""Tests for the iAWE ingest helper (single- and multi-household)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from iawe_ingest import _is_main_meter, ingest


def _write_meter(directory, name="1.csv", days=3, watts=100.0):
    directory.mkdir(parents=True, exist_ok=True)
    start = pd.Timestamp("2023-01-01", tz="Asia/Kolkata")
    stamps = []
    for d in range(days):
        base = start + pd.Timedelta(days=d)
        for h in range(24):
            stamps.append(int((base + pd.Timedelta(hours=h)).timestamp()))
    pd.DataFrame({"timestamp": stamps, "W": np.full(len(stamps), watts)}).to_csv(
        directory / name, index=False
    )


def test_is_main_meter_detection():
    assert _is_main_meter("1.csv")
    assert _is_main_meter("main_meter.csv")
    assert not _is_main_meter("fridge.csv")
    assert not _is_main_meter("air_condition.csv")


def test_ingest_single_directory_is_wide(tmp_path):
    _write_meter(tmp_path)
    df = ingest([tmp_path])
    assert list(df.columns) == ["date", "consumption_kwh"]
    assert len(df) == 3
    assert (df["consumption_kwh"] >= 0).all()


def test_ingest_multiple_directories_is_long(tmp_path):
    _write_meter(tmp_path / "home_a", watts=100.0)
    _write_meter(tmp_path / "home_b", watts=200.0)
    df = ingest([tmp_path / "home_a", tmp_path / "home_b"])
    assert list(df.columns) == ["date", "household", "consumption_kwh"]
    assert set(df["household"]) == {"home_a", "home_b"}
    # Each day appears once per household.
    assert (df.groupby("date").size() == 2).all()
    # The higher-wattage home consumes more.
    means = df.groupby("household")["consumption_kwh"].mean()
    assert means["home_b"] > means["home_a"]


def test_ingest_requires_a_source(tmp_path):
    with pytest.raises(ValueError, match="at least one"):
        ingest([])
