"""BijliPredict AI - data loading, cleaning, splitting and windowing.

Everything here is deliberately leakage-safe:

  * the chronological train/val/test split happens on the raw time series;
  * the scaler is fit on the TRAIN segment only;
  * sliding windows are built inside each segment, so no validation/test target
    is ever visible to training.

Public API:
    load_consumption(path)                 -> raw DataFrame (DatetimeIndex)
    clean_series(df)                       -> cleaned, gap-filled daily series
    chronological_split(df, ...)           -> {'train','val','test'} DataFrames
    make_windows(arr, input_len, horizon)  -> (X, y) numpy arrays
    load_and_prepare(path, ...)            -> prepared arrays + scaler + frames
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

DEFAULT_INPUT_LEN = 30
DEFAULT_HORIZON = 7
DEFAULT_SPLITS = (0.70, 0.15, 0.15)

_DATE_ALIASES = ("date", "datetime", "timestamp", "time", "ds")
_VALUE_ALIASES = ("consumption_kwh", "consumption", "kwh", "energy_kwh", "value", "y")


def _detect_column(df: pd.DataFrame, aliases: tuple[str, ...], kind: str) -> str:
    lower = {c.lower().strip(): c for c in df.columns}
    for a in aliases:
        if a in lower:
            return lower[a]
    # Loose contains match as a fallback.
    for a in aliases:
        for low, orig in lower.items():
            if a in low:
                return orig
    raise ValueError(
        f"Could not find a {kind} column. Looked for one of {aliases}; "
        f"file has columns {list(df.columns)}."
    )


def load_consumption(
    path: str | Path,
    date_col: str | None = None,
    value_col: str | None = None,
) -> pd.DataFrame:
    """Load a CSV into a DatetimeIndex frame with a single `consumption_kwh` column."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    raw = pd.read_csv(path)
    if raw.empty:
        raise ValueError(f"Data file is empty: {path}")

    dcol = date_col or _detect_column(raw, _DATE_ALIASES, "date")
    vcol = value_col or _detect_column(raw, _VALUE_ALIASES, "consumption")

    df = pd.DataFrame(
        {
            "date": pd.to_datetime(raw[dcol], errors="coerce"),
            "consumption_kwh": pd.to_numeric(raw[vcol], errors="coerce"),
        }
    )
    df = df.dropna(subset=["date"]).set_index("date").sort_index()
    return df


def _dedupe_series(df: pd.DataFrame) -> pd.Series:
    """Collapse duplicate timestamps and drop invalid / negative values.

    Performs no gap filling, so callers can choose when (and over which segment)
    interpolation happens.
    """
    if "consumption_kwh" not in df.columns:
        df = df.rename(columns={df.columns[0]: "consumption_kwh"})

    s = df["consumption_kwh"].astype(float)
    # Average duplicate timestamps, then drop non-finite / negative values.
    series = s.groupby(level=0).mean()
    series = series.replace([np.inf, -np.inf], np.nan).dropna()
    series = series[series >= 0.0].sort_index()
    if len(series) < 3:
        raise ValueError("Not enough valid rows after cleaning (need >= 3).")
    return series


def _fill_daily_gaps(series: pd.Series, fill_limit: int = 14) -> pd.DataFrame:
    """Reindex one segment to a continuous daily index and fill its gaps.

    Missing values are interpolated in time (bounded by `fill_limit`) then
    edge-filled, so a segment never borrows values from a neighbouring split.
    """
    full_index = pd.date_range(series.index.min(), series.index.max(), freq="D")
    series = series.reindex(full_index)
    n_missing = int(series.isna().sum())
    series = series.interpolate(method="time", limit=fill_limit, limit_direction="both")
    series = series.ffill().bfill()

    out = series.to_frame(name="consumption_kwh")
    out.index.name = "date"
    out.attrs["n_missing_filled"] = n_missing
    return out


def clean_series(df: pd.DataFrame, fill_limit: int = 14) -> pd.DataFrame:
    """Deduplicate, drop invalid rows, reindex to daily and fill gaps.

    Returns a frame with a continuous daily DatetimeIndex and non-negative
    `consumption_kwh`. Intended for standalone series (e.g. a prediction window);
    the training pipeline fills each split segment separately to avoid leakage.
    """
    return _fill_daily_gaps(_dedupe_series(df), fill_limit)


def chronological_split(
    df: pd.DataFrame,
    train_ratio: float = DEFAULT_SPLITS[0],
    val_ratio: float = DEFAULT_SPLITS[1],
    test_ratio: float = DEFAULT_SPLITS[2],
) -> dict[str, pd.DataFrame]:
    """Split a time-ordered frame into consecutive train/val/test segments."""
    total = train_ratio + val_ratio + test_ratio
    if not np.isclose(total, 1.0, atol=1e-6):
        raise ValueError(f"Split ratios must sum to 1.0, got {total}.")

    n = len(df)
    n_train = int(np.floor(n * train_ratio))
    n_val = int(np.floor(n * val_ratio))
    n_test = n - n_train - n_val
    if min(n_train, n_val, n_test) <= 0:
        raise ValueError(f"Split too small for {n} rows -> {n_train}/{n_val}/{n_test}.")

    return {
        "train": df.iloc[:n_train],
        "val": df.iloc[n_train : n_train + n_val],
        "test": df.iloc[n_train + n_val :],
    }


def make_windows(
    values: np.ndarray, input_len: int = DEFAULT_INPUT_LEN, horizon: int = DEFAULT_HORIZON
) -> tuple[np.ndarray, np.ndarray]:
    """Build sliding (input_len -> horizon) windows.

    X: (n_samples, input_len, 1)   y: (n_samples, horizon)
    """
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    n = len(values)
    n_windows = n - input_len - horizon + 1
    if n_windows <= 0:
        raise ValueError(
            f"Series too short ({n}) for window={input_len} + horizon={horizon}."
        )

    X = np.empty((n_windows, input_len, 1), dtype=np.float32)
    y = np.empty((n_windows, horizon), dtype=np.float32)
    for i in range(n_windows):
        X[i, :, 0] = values[i : i + input_len]
        y[i] = values[i + input_len : i + input_len + horizon]
    return X, y


def _scale(values: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    return scaler.transform(np.asarray(values, dtype=float).reshape(-1, 1)).reshape(-1)


def load_and_prepare(
    path: str | Path,
    input_len: int = DEFAULT_INPUT_LEN,
    horizon: int = DEFAULT_HORIZON,
    splits: tuple[float, float, float] = DEFAULT_SPLITS,
    scaler_path: str | Path | None = None,
    save_scaler: bool = True,
    scaler: StandardScaler | None = None,
    fill_limit: int = 14,
) -> dict:
    """Full preprocessing pipeline: load -> clean -> split -> scale -> window.

    The scaler is fit on TRAIN only, unless a pre-fitted `scaler` is supplied
    (used by evaluate/predict to reuse the exact training-time scaling). Gap
    filling is applied per split segment, so validation/test values can never be
    interpolated into the training segment. Returns a dict with per-split X/y
    arrays, the fitted scaler, and the split frames.
    """
    deduped = _dedupe_series(load_consumption(path))
    raw_frames = chronological_split(
        deduped.to_frame(name="consumption_kwh"), *splits
    )
    frames = {
        name: _fill_daily_gaps(frame["consumption_kwh"], fill_limit)
        for name, frame in raw_frames.items()
    }
    series = pd.concat(frames.values())

    if scaler is None:
        train_vals = frames["train"]["consumption_kwh"].to_numpy()
        scaler = StandardScaler().fit(train_vals.reshape(-1, 1))

    datasets: dict[str, dict[str, np.ndarray]] = {}
    for name, frame in frames.items():
        scaled = _scale(frame["consumption_kwh"].to_numpy(), scaler)
        try:
            X, y = make_windows(scaled, input_len, horizon)
            datasets[name] = {"X": X, "y": y}
        except ValueError:
            # A segment shorter than one window cannot produce samples.
            datasets[name] = {
                "X": np.empty((0, input_len, 1), dtype=np.float32),
                "y": np.empty((0, horizon), dtype=np.float32),
            }

    if save_scaler and scaler_path is not None:
        Path(scaler_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, scaler_path)

    return {
        "datasets": datasets,
        "scaler": scaler,
        "frames": frames,
        "series": series,
        "input_len": input_len,
        "horizon": horizon,
    }


def load_scaler(path: str | Path) -> StandardScaler:
    return joblib.load(path)


if __name__ == "__main__":  # tiny smoke test
    import sys

    src = sys.argv[1] if len(sys.argv) > 1 else "data/processed/daily_consumption.csv"
    prepared = load_and_prepare(src, scaler_path="artifacts/scaler.joblib")
    for name, d in prepared["datasets"].items():
        print(f"{name:>5}: X={d['X'].shape}  y={d['y'].shape}")
