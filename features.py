"""BijliPredict AI - exogenous feature engineering.

Builds the per-day feature matrix the forecaster consumes alongside the
consumption history:

  * **calendar** - day-of-week and month sine/cosine plus an is_weekend flag.
  * **holidays** - an Indian public-holiday flag (``holidays`` package).
  * **temperature** - mean temperature, short lags, a rolling mean and
    cooling-degree-days (CDD), from the Open-Meteo series in `weather.py`.

A :class:`FeatureConfig` describes exactly which blocks are on and is persisted
in the training checkpoint, so `evaluate.py` and `predict.py` rebuild byte-for-
byte identical inputs.

Feature parity contract
-----------------------
Both the training and inference paths call :func:`build_feature_matrix` with the
same config, the same temperature series **plus enough leading buffer**
(:attr:`FeatureConfig.buffer_days`) for the lags/rolling windows, and then scale
with the same train-fitted scaler. `preprocessing.load_and_prepare` and the
predict-side rollout both honour that contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import holidays
import numpy as np
import pandas as pd

from weather import DEFAULT_CACHE_DIR as WEATHER_CACHE_DIR
from weather import TEMP_COLUMNS, get_temperature

TEMP_ROLL_DEFAULT = 7
CDD_BASE_DEFAULT = 24.0
TEMP_LAGS_DEFAULT = (1, 2, 3)

CALENDAR_COLUMNS = ["dow_sin", "dow_cos", "month_sin", "month_cos", "is_weekend"]
HOLIDAY_COLUMNS = ["is_holiday"]


@dataclass(frozen=True)
class FeatureConfig:
    """Which exogenous feature blocks to build (persisted in the checkpoint)."""

    calendar: bool = False
    holidays: bool = False
    temperature: bool = False
    city: str = "delhi"
    country: str = "IN"
    temp_lags: tuple[int, ...] = TEMP_LAGS_DEFAULT
    temp_roll: int = TEMP_ROLL_DEFAULT
    cdd_base: float = CDD_BASE_DEFAULT

    @property
    def enabled(self) -> bool:
        return self.calendar or self.holidays or self.temperature

    @property
    def buffer_days(self) -> int:
        """Leading days of temperature context needed for lags/rolling."""
        if not self.temperature:
            return 0
        return int(max((*self.temp_lags, self.temp_roll), default=0))

    def n_exog(self) -> int:
        """Number of exogenous columns :func:`build_feature_matrix` emits."""
        n = 0
        if self.calendar:
            n += len(CALENDAR_COLUMNS)
        if self.holidays:
            n += len(HOLIDAY_COLUMNS)
        if self.temperature:
            n += 3 + len(self.temp_lags)  # temp_mean + roll + cdd + lags
        return n

    def to_dict(self) -> dict:
        return {
            "calendar": self.calendar,
            "holidays": self.holidays,
            "temperature": self.temperature,
            "city": self.city,
            "country": self.country,
            "temp_lags": list(self.temp_lags),
            "temp_roll": self.temp_roll,
            "cdd_base": self.cdd_base,
        }

    @classmethod
    def from_dict(cls, data: dict | None) -> FeatureConfig:
        if not data:
            return cls()
        return cls(
            calendar=bool(data.get("calendar", False)),
            holidays=bool(data.get("holidays", False)),
            temperature=bool(data.get("temperature", False)),
            city=str(data.get("city", "delhi")),
            country=str(data.get("country", "IN")),
            temp_lags=tuple(int(v) for v in data.get("temp_lags", TEMP_LAGS_DEFAULT)),
            temp_roll=int(data.get("temp_roll", TEMP_ROLL_DEFAULT)),
            cdd_base=float(data.get("cdd_base", CDD_BASE_DEFAULT)),
        )


def calendar_features(dates) -> pd.DataFrame:
    """Day-of-week / month cyclical encodings plus an is_weekend flag."""
    idx = pd.DatetimeIndex(pd.to_datetime(list(dates)))
    idx.name = "date"
    dow = idx.dayofweek.to_numpy().astype(float)
    month = idx.month.to_numpy().astype(float)
    return pd.DataFrame(
        {
            "dow_sin": np.sin(2.0 * np.pi * dow / 7.0),
            "dow_cos": np.cos(2.0 * np.pi * dow / 7.0),
            "month_sin": np.sin(2.0 * np.pi * (month - 1.0) / 12.0),
            "month_cos": np.cos(2.0 * np.pi * (month - 1.0) / 12.0),
            "is_weekend": (dow >= 5.0).astype(float),
        },
        index=idx,
    )


def holiday_features(dates, country: str = "IN") -> pd.DataFrame:
    """Public-holiday flag (1.0 on a holiday) using the ``holidays`` package."""
    idx = pd.DatetimeIndex(pd.to_datetime(list(dates)))
    idx.name = "date"
    years = sorted({int(y) for y in idx.year})
    try:
        calendar = holidays.country_holidays(country, years=years)
    except Exception:  # unknown country code -> no holidays rather than a crash
        calendar = {}
    flagged = np.array([1.0 if d.date() in calendar else 0.0 for d in idx], dtype=float)
    return pd.DataFrame({"is_holiday": flagged}, index=idx)


def temperature_features(temperature: pd.DataFrame, config: FeatureConfig) -> pd.DataFrame:
    """Mean temp, lagged temps, rolling mean and cooling-degree-days.

    Computed over whatever temperature index is supplied (callers pass a range
    that starts ``config.buffer_days`` before the dates they actually need).
    """
    if "temp_mean" not in temperature.columns:
        raise ValueError("temperature frame must contain a 'temp_mean' column.")
    temp = temperature.sort_index()
    mean = temp["temp_mean"].astype(float)
    out = pd.DataFrame(index=temp.index)
    out["temp_mean"] = mean
    for lag in config.temp_lags:
        out[f"temp_lag{int(lag)}"] = mean.shift(int(lag))
    out[f"temp_roll{int(config.temp_roll)}"] = mean.rolling(
        int(config.temp_roll), min_periods=1
    ).mean()
    out["cdd"] = (mean - float(config.cdd_base)).clip(lower=0.0)
    # Leading rows lack full history; fill so downstream scaling never sees NaN.
    return out.ffill().bfill()


def build_feature_matrix(
    dates,
    *,
    temperature: pd.DataFrame | None = None,
    config: FeatureConfig | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Assemble the exogenous feature matrix for ``dates``.

    ``temperature`` (if the config needs it) must cover ``dates`` *plus*
    ``config.buffer_days`` of leading context, because lags/rolling are computed
    over its full index and then sliced. Returns ``(frame, column_names)``.
    """
    config = config or FeatureConfig()
    idx = pd.DatetimeIndex(pd.to_datetime(list(dates)))
    idx.name = "date"

    parts: list[pd.DataFrame] = []
    if config.calendar:
        parts.append(calendar_features(idx))
    if config.holidays:
        parts.append(holiday_features(idx, config.country))
    if config.temperature:
        if temperature is None:
            raise ValueError("temperature features requested but no temperature provided.")
        parts.append(temperature_features(temperature, config))

    if not parts:
        return pd.DataFrame(index=idx), []

    matrix = pd.concat([part.reindex(idx) for part in parts], axis=1)
    columns = list(matrix.columns)
    if matrix.loc[:, columns].isna().any().any():
        missing = matrix.index[matrix.loc[:, columns].isna().any(axis=1)]
        raise ValueError(
            f"Feature matrix has NaNs for {len(missing)} date(s) "
            f"(e.g. {missing[0].date()}); extend the temperature range by "
            f"{config.buffer_days} leading day(s)."
        )
    return matrix.astype(np.float32), columns


def temperature_for_dates(
    config: FeatureConfig,
    dates,
    *,
    weather_file: str | Path | None = None,
    cache_dir=WEATHER_CACHE_DIR,
    fetcher=None,
) -> pd.DataFrame | None:
    """Temperature series covering ``dates`` plus ``config.buffer_days`` of lead.

    Returns ``None`` when temperature features are disabled, so callers can pass
    the result straight to :func:`build_feature_matrix`. The extra leading days
    give the lag/rolling windows valid context; callers must therefore build the
    feature matrix over the **same** buffered range and then slice.
    """
    if not config.temperature:
        return None
    idx = pd.DatetimeIndex(pd.to_datetime(list(dates)))
    if len(idx) == 0:
        return pd.DataFrame(columns=TEMP_COLUMNS)
    start = idx.min() - pd.Timedelta(days=config.buffer_days)
    full = pd.date_range(start, idx.max(), freq="D")
    kwargs = {} if fetcher is None else {"fetcher": fetcher}
    return get_temperature(
        config.city, full, weather_file=weather_file, cache_dir=cache_dir, **kwargs
    )


def stack_window(consumption_window: np.ndarray, feature_window: np.ndarray | None) -> np.ndarray:
    """Stack a consumption window and its exogenous features into one tensor.

    ``consumption_window``: (input_len,) or (input_len, 1).
    ``feature_window``: (input_len, n_exog) or None.
    Returns (input_len, 1 + n_exog) float32 - the model's input layout.
    """
    cons = np.asarray(consumption_window, dtype=np.float32)
    if cons.ndim == 1:
        cons = cons.reshape(-1, 1)
    if feature_window is None:
        return cons.astype(np.float32)

    feats = np.asarray(feature_window, dtype=np.float32)
    if feats.ndim == 1:
        feats = feats.reshape(-1, 1)
    if feats.shape[0] != cons.shape[0]:
        raise ValueError(
            f"window length mismatch: consumption {cons.shape[0]} vs features {feats.shape[0]}"
        )
    return np.concatenate([cons, feats], axis=1).astype(np.float32)
