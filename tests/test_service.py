"""Tests for the shared, framework-free helpers in `service.py`."""

from __future__ import annotations

import pandas as pd

from service import chart_columns


def test_chart_columns_without_calibration():
    frame = pd.DataFrame({"date": ["2026-01-01"], "predicted_kwh": [5.0]})
    assert chart_columns(frame) == ["predicted_kwh"]


def test_chart_columns_includes_present_bands():
    frame = pd.DataFrame(
        {
            "date": ["2026-01-01"],
            "predicted_kwh": [5.0],
            "lower_80": [4.0],
            "upper_80": [6.0],
            "lower_90": [3.5],
            "upper_90": [6.5],
        }
    )
    cols = chart_columns(frame)
    assert cols[0] == "predicted_kwh"
    assert set(cols[1:]) == {"lower_80", "upper_80", "lower_90", "upper_90"}


def test_chart_columns_partial_bands():
    frame = pd.DataFrame(
        {"predicted_kwh": [5.0], "lower_90": [3.5], "upper_90": [6.5]}
    )
    assert chart_columns(frame) == ["predicted_kwh", "lower_90", "upper_90"]
