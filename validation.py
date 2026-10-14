"""Input-data validation for consumption CSVs.

`load_consumption` runs these checks so a malformed file fails early with an
actionable message instead of a confusing error deep inside the pipeline:

  * the date and consumption columns must exist (alias detection lives in
    `preprocessing.py`);
  * every date must be present and parseable;
  * every consumption value must be numeric;
  * duplicate timestamps, missing values and negative values are reported as
    *warnings* because `clean_series` already averages duplicates, interpolates
    gaps and drops negatives.

When the optional ``validate`` extra (pandera) is installed, a declarative
pandera schema is applied on top of the dependency-free checks. Without pandera
the checks below still run, so validation is always available — the extra only
adds the schema layer.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

MIN_ROWS = 3
DEFAULT_MAX_EXAMPLES = 3

DATE_ALIASES = ("date", "datetime", "timestamp", "time", "ds")
VALUE_ALIASES = ("consumption_kwh", "consumption", "kwh", "energy_kwh", "value", "y")


def pandera_available() -> bool:
    """True when the optional ``validate`` extra (pandera) is importable."""
    return importlib.util.find_spec("pandera") is not None


@dataclass
class ValidationReport:
    """Outcome of validating a raw consumption frame."""

    n_rows: int
    n_bad_dates: int = 0
    n_missing_dates: int = 0
    n_bad_values: int = 0
    n_missing_values: int = 0
    n_negative: int = 0
    n_duplicate_dates: int = 0
    schema_engine: str = "builtin"
    examples: dict[str, list[str]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not (
            self.n_rows < MIN_ROWS or self.n_bad_dates or self.n_missing_dates or self.n_bad_values
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_rows": self.n_rows,
            "n_bad_dates": self.n_bad_dates,
            "n_missing_dates": self.n_missing_dates,
            "n_bad_values": self.n_bad_values,
            "n_missing_values": self.n_missing_values,
            "n_negative": self.n_negative,
            "n_duplicate_dates": self.n_duplicate_dates,
            "schema_engine": self.schema_engine,
            "ok": self.ok,
            "warnings": list(self.warnings),
            "examples": {k: list(v) for k, v in self.examples.items()},
        }


def _examples(mask: pd.Series, series: pd.Series, k: int = DEFAULT_MAX_EXAMPLES) -> list[str]:
    idx = list(series.index[mask])[:k]
    return [f"row {i}: {series.loc[i]!r}" for i in idx]


def _combined(messages: list[str]) -> str:
    return "Invalid consumption data:\n  - " + "\n  - ".join(messages)


def validation_to_dataframe(report: ValidationReport) -> pd.DataFrame:
    """Render a report as a tidy ``check | status | count`` table.

    Hard failures are ``error`` when non-zero, soft findings ``warn``; a zero
    count is always ``ok``. Useful for the Excel/HTML reports and for tests.
    """
    hard = {
        "rows": report.n_rows if report.n_rows < MIN_ROWS else 0,
        "bad_dates": report.n_bad_dates,
        "bad_values": report.n_bad_values,
        "missing_dates": report.n_missing_dates,
    }
    soft = {
        "missing_values": report.n_missing_values,
        "negative_values": report.n_negative,
        "duplicate_dates": report.n_duplicate_dates,
    }
    rows = []
    for check, count in hard.items():
        rows.append({"check": check, "status": "error" if count else "ok", "count": int(count)})
    for check, count in soft.items():
        rows.append({"check": check, "status": "warn" if count else "ok", "count": int(count)})
    return pd.DataFrame(rows, columns=["check", "status", "count"])


def validate_consumption_frame(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    *,
    strict: bool = True,
    max_examples: int = DEFAULT_MAX_EXAMPLES,
) -> ValidationReport:
    """Validate a raw consumption frame.

    Returns a :class:`ValidationReport`. Raises ``ValueError`` with a friendly,
    row-annotated message on hard errors when ``strict`` (the default). Set
    ``strict=False`` to collect the same findings as warnings instead.
    """
    n = len(df)
    raw_date = df[date_col]
    raw_value = df[value_col]
    parsed_date = pd.to_datetime(raw_date, errors="coerce")
    parsed_value = pd.to_numeric(raw_value, errors="coerce")

    bad_date_mask = raw_date.notna() & parsed_date.isna()
    missing_date_mask = raw_date.isna()
    bad_value_mask = raw_value.notna() & parsed_value.isna()
    negative_mask = parsed_value < 0
    duplicate_mask = parsed_date.duplicated(keep=False)

    report = ValidationReport(
        n_rows=n,
        n_bad_dates=int(bad_date_mask.sum()),
        n_missing_dates=int(missing_date_mask.sum()),
        n_bad_values=int(bad_value_mask.sum()),
        n_missing_values=int(raw_value.isna().sum()),
        n_negative=int(negative_mask.sum()),
        n_duplicate_dates=int(duplicate_mask.sum()),
        schema_engine="pandera" if pandera_available() else "builtin",
        examples={
            "bad_dates": _examples(bad_date_mask, raw_date, max_examples),
            "bad_values": _examples(bad_value_mask, raw_value, max_examples),
            "negative": _examples(negative_mask, raw_value, max_examples),
        },
    )

    if report.n_negative:
        report.warnings.append(
            f"{report.n_negative} negative consumption value(s) will be treated as invalid."
        )
    if report.n_duplicate_dates:
        report.warnings.append(
            f"{report.n_duplicate_dates} duplicate timestamp(s) will be averaged."
        )
    if report.n_missing_values:
        report.warnings.append(
            f"{report.n_missing_values} missing consumption value(s) will be interpolated."
        )

    problems: list[str] = []
    if n < MIN_ROWS:
        problems.append(f"only {n} row(s); need at least {MIN_ROWS}.")
    if report.n_missing_dates:
        problems.append(
            f"{report.n_missing_dates} row(s) have an empty date in column {date_col!r}."
        )
    if report.n_bad_dates:
        detail = f" (e.g. {', '.join(report.examples['bad_dates'])})" if report.examples else ""
        problems.append(
            f"{report.n_bad_dates} row(s) have an unparseable date in column {date_col!r}{detail}."
        )
    if report.n_bad_values:
        detail = f" (e.g. {', '.join(report.examples['bad_values'])})" if report.examples else ""
        problems.append(
            f"{report.n_bad_values} row(s) have a non-numeric value in column {value_col!r}{detail}."
        )

    if problems and strict:
        raise ValueError(
            _combined(problems) + "\nFix the file, or pass validate=False to skip validation."
        )

    if strict and pandera_available():
        pandera_validate(df, date_col, value_col)
    return report


def pandera_validate(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    *,
    strict: bool = True,
) -> None:
    """Apply the declarative pandera schema to the two selected columns.

    Raises ``ValueError`` (wrapping pandera's ``SchemaError``/``SchemaErrors``)
    when the schema fails. Requires the optional ``validate`` extra.
    """
    pap = _pandera()
    frame = pd.DataFrame(
        {
            "date": df[date_col].astype(str),
            "consumption_kwh": df[value_col],
        }
    )
    schema = pap.DataFrameSchema(
        {
            "date": pap.Column(
                str,
                pap.Check(
                    lambda s: pd.to_datetime(s, errors="coerce").notna().all(),
                    error="date column contains unparseable values",
                ),
                nullable=False,
            ),
            "consumption_kwh": pap.Column(
                float,
                nullable=True,
                coerce=True,
            ),
        },
        strict=False,
        coerce=True,
    )
    try:
        schema.validate(frame, lazy=True)
    except (pap.errors.SchemaError, pap.errors.SchemaErrors) as exc:
        detail = getattr(exc, "failure_cases", None)
        count = "" if detail is None else f" ({len(detail)} check failure(s))"
        raise ValueError(
            f"Consumption data failed pandera schema validation{count}: {exc}"
        ) from exc


def _pandera():
    """Import the pandera pandas backend, tolerating both old and new layouts."""
    try:
        from pandera import pandas as pap  # pandera >= 0.20

        return pap
    except ImportError:  # pragma: no cover - very old pandera
        import pandera as pap

        return pap
