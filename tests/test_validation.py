"""Tests for consumption-data validation (validation.py)."""

from __future__ import annotations

import pandas as pd
import pytest

from validation import (
    pandera_available,
    pandera_validate,
    validate_consumption_frame,
    validation_to_dataframe,
)


def _frame(**cols):
    base = {
        "date": ["2023-01-01", "2023-01-02", "2023-01-03"],
        "consumption_kwh": [3.0, 3.2, 2.9],
    }
    base.update(cols)
    return pd.DataFrame(base)


def test_clean_frame_has_no_errors():
    report = validate_consumption_frame(_frame(), "date", "consumption_kwh")
    assert report.ok
    assert report.n_rows == 3
    assert report.n_bad_dates == 0
    assert report.n_bad_values == 0
    assert report.n_negative == 0


def test_unparseable_dates_are_errors():
    df = _frame(date=["2023-01-01", "not-a-date", "2023-01-03"])
    report = validate_consumption_frame(df, "date", "consumption_kwh", strict=False)
    assert not report.ok
    assert report.n_bad_dates == 1


def test_non_numeric_values_are_errors():
    df = _frame(consumption_kwh=[3.0, "oops", 2.9])
    report = validate_consumption_frame(df, "date", "consumption_kwh", strict=False)
    assert not report.ok
    assert report.n_bad_values == 1


def test_negative_values_are_warnings():
    df = _frame(consumption_kwh=[3.0, -1.0, 2.9])
    report = validate_consumption_frame(df, "date", "consumption_kwh")
    assert report.ok  # warnings do not fail the run
    assert report.n_negative == 1
    assert report.warnings


def test_missing_values_are_warnings():
    df = _frame(consumption_kwh=[3.0, None, 2.9])
    report = validate_consumption_frame(df, "date", "consumption_kwh")
    assert report.ok
    assert report.n_missing_values == 1
    assert report.n_bad_values == 0


def test_too_few_rows_is_error():
    df = _frame(date=["2023-01-01"], consumption_kwh=[3.0])
    report = validate_consumption_frame(df, "date", "consumption_kwh", strict=False)
    assert not report.ok


def test_report_as_dataframe_roundtrip():
    df = _frame(consumption_kwh=[3.0, -1.0, None])
    report = validate_consumption_frame(df, "date", "consumption_kwh")
    out = validation_to_dataframe(report)
    assert list(out.columns) == ["check", "status", "count"]
    assert set(out["check"]) == {
        "rows",
        "bad_dates",
        "bad_values",
        "missing_dates",
        "missing_values",
        "negative_values",
        "duplicate_dates",
    }
    rows = dict(zip(out["check"], out["count"], strict=False))
    assert rows["negative_values"] == 1
    assert rows["missing_values"] == 1


def test_absence_of_consumption_column_raises():
    df = pd.DataFrame({"date": ["2023-01-01"], "kwh": [1.0]})
    with pytest.raises(KeyError):
        validate_consumption_frame(df, "date", "consumption_kwh")


@pytest.mark.skipif(not pandera_available(), reason="pandera not installed")
def test_pandera_validate_accepts_clean_frame():
    pandera_validate(_frame(), "date", "consumption_kwh")


@pytest.mark.skipif(not pandera_available(), reason="pandera not installed")
def test_pandera_validate_allows_negative():
    # Negatives are a soft warning everywhere, so the schema must not reject them.
    pandera_validate(_frame(consumption_kwh=[3.0, -5.0, 2.9]), "date", "consumption_kwh")


@pytest.mark.skipif(not pandera_available(), reason="pandera not installed")
def test_pandera_validate_rejects_non_numeric():
    with pytest.raises(ValueError):
        pandera_validate(_frame(consumption_kwh=[3.0, "oops", 2.9]), "date", "consumption_kwh")


@pytest.mark.skipif(not pandera_available(), reason="pandera not installed")
def test_pandera_validate_rejects_bad_date():
    with pytest.raises(ValueError):
        pandera_validate(
            _frame(date=["2023-01-01", "nope", "2023-01-03"]), "date", "consumption_kwh"
        )
