"""BijliPredict AI - shared inference service used by the API and the UI.

Holds the model + scalers + feature metadata + conformal calibration in one
immutable bundle, and the pure helpers that turn a history of daily kWh into a
forecast, intervals and a bill. No web framework is imported here, so both
``api.py`` (FastAPI) and ``app.py`` (Streamlit) reuse it without pulling each
other's dependencies.

Errors raise plain Python exceptions (``ValueError`` / ``KeyError``); the API
layer translates those into HTTP status codes.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import calibration
from features import build_feature_matrix, temperature_for_dates
from inference import feature_info, load_model, resolve_device
from predict import INTERVAL_LEVELS, MONTH_DAYS, QUARTER_DAYS, rolling_forecast
from preprocessing import load_scaler
from tariffs import (
    DEFAULT_PRESETS_PATH,
    DEFAULT_RATE,
    Tariff,
    compute_bill,
    get_preset,
    parse_slabs,
)

DEFAULT_MODEL_PATH = "artifacts/best_model.pt"
DEFAULT_SCALER_PATH = "artifacts/scaler.joblib"
DEFAULT_CALIBRATION_PATH = "artifacts/calibration.json"

__all__ = [
    "DEFAULT_CALIBRATION_PATH",
    "DEFAULT_MODEL_PATH",
    "DEFAULT_PRESETS_PATH",
    "DEFAULT_SCALER_PATH",
    "MONTH_DAYS",
    "QUARTER_DAYS",
    "ServiceState",
    "compute_bill",
    "intervals",
    "load_service",
    "period_summary",
    "resolve_tariff",
    "run_forecast",
]


@dataclass
class ServiceState:
    model: object
    ckpt: dict
    input_len: int
    horizon: int
    scaler: object
    feature_config: object
    feature_columns: list[str]
    feature_scaler: object | None
    calib: dict | None
    device: object
    weather_file: str | None

    @property
    def model_name(self) -> str:
        return str(self.ckpt.get("model_type", "model"))


def load_service(
    *,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    scaler_path: str | Path | None = None,
    calibration_path: str | Path | None = DEFAULT_CALIBRATION_PATH,
    weather_file: str | None = None,
    device: str = "cpu",
) -> ServiceState:
    """Load every artifact a request needs into one immutable bundle."""
    dev = resolve_device(device)
    bundle = load_model(model_path, dev)
    model, ckpt = bundle["model"], bundle["ckpt"]

    resolved_scaler = scaler_path or ckpt.get("scaler_path") or DEFAULT_SCALER_PATH
    scaler = load_scaler(resolved_scaler)

    feature_config, feature_columns, feature_scaler_path = feature_info(ckpt)
    feature_scaler = None
    if feature_columns:
        if feature_scaler_path is None:
            raise ValueError("Checkpoint expects features but records no feature scaler path.")
        feature_scaler = load_scaler(feature_scaler_path)

    calib = None
    if calibration_path and Path(calibration_path).exists():
        calib = calibration.load(calibration_path)

    return ServiceState(
        model=model,
        ckpt=ckpt,
        input_len=int(ckpt["input_len"]),
        horizon=int(ckpt["horizon"]),
        scaler=scaler,
        feature_config=feature_config,
        feature_columns=list(feature_columns),
        feature_scaler=feature_scaler,
        calib=calib,
        device=dev,
        weather_file=weather_file,
    )


def _future_dates(last_date: _dt.date, periods: int) -> pd.DatetimeIndex:
    start = pd.Timestamp(last_date) + pd.Timedelta(days=1)
    return pd.date_range(start, periods=periods, freq="D")


def _build_exog(
    svc: ServiceState,
    history: np.ndarray,
    future_dates: pd.DatetimeIndex,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Build raw history/future feature blocks for the rolling forecast.

    Returns ``(None, None)`` for checkpoints without exogenous features. Mirrors
    the date/feature construction in ``predict.main`` so the service and CLI
    produce identical inputs.
    """
    if not svc.feature_columns:
        return None, None
    if svc.feature_scaler is None:
        raise ValueError("Checkpoint expects features but the feature scaler is missing.")

    last_history_date = pd.Timestamp(future_dates[0]) - pd.Timedelta(days=1)
    window_dates = pd.date_range(end=last_history_date, periods=len(history), freq="D")
    feat_start = window_dates[0] - pd.Timedelta(days=svc.feature_config.buffer_days)
    full_dates = pd.date_range(feat_start, future_dates[-1], freq="D")

    temperature = None
    if svc.feature_config.temperature:
        temperature = temperature_for_dates(
            svc.feature_config, full_dates, weather_file=svc.weather_file
        )
    matrix, columns = build_feature_matrix(
        full_dates, temperature=temperature, config=svc.feature_config
    )
    if columns != svc.feature_columns:
        raise ValueError(f"Feature mismatch: model expects {svc.feature_columns}, built {columns}.")
    return matrix.loc[window_dates].to_numpy(), matrix.loc[future_dates].to_numpy()


def run_forecast(
    svc: ServiceState,
    history: list[float],
    total_days: int,
    last_date: _dt.date,
) -> tuple[list[float], pd.DatetimeIndex]:
    """Rolling autoregressive forecast of `total_days` daily kWh values."""
    if len(history) < svc.input_len:
        raise ValueError(f"Need at least {svc.input_len} days of history, got {len(history)}.")
    if total_days < 1:
        raise ValueError("total_days must be >= 1.")

    window = np.asarray(history[-svc.input_len :], dtype=float)
    future_dates = _future_dates(last_date, total_days)
    history_features, future_features = _build_exog(svc, window, future_dates)

    preds = rolling_forecast(
        svc.model,
        svc.scaler,
        window,
        svc.input_len,
        svc.horizon,
        total_days,
        svc.device,
        feature_scaler=svc.feature_scaler,
        history_features=history_features,
        future_features=future_features,
    )
    return preds, future_dates


def intervals(pred: list[float], calib: dict | None) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Lower/upper kWh bounds per conformal level (empty without calibration)."""
    out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    if not calib:
        return out
    for level in INTERVAL_LEVELS:
        lower, upper = calibration.apply(pred, calib, level, steps=len(pred))
        out[int(round(level * 100))] = (np.asarray(lower), np.asarray(upper))
    return out


def resolve_tariff(
    *,
    rate: float | None = None,
    slab: str | None = None,
    preset: str | None = None,
    fixed: float | None = None,
    tax_pct: float | None = None,
    presets_file: str | Path = DEFAULT_PRESETS_PATH,
) -> Tariff:
    """Resolve a Tariff from preset / slab / flat rate, then apply overrides."""
    if preset:
        tariff = get_preset(preset, presets_file)
        if slab:
            tariff.slabs = parse_slabs(slab)
            tariff.rate = 0.0
            tariff.label = f"slab ({preset})"
    elif slab:
        tariff = Tariff(slabs=parse_slabs(slab), label="slab")
    else:
        tariff = Tariff(rate=rate if rate is not None else DEFAULT_RATE, label="flat")

    if fixed is not None:
        tariff.fixed = fixed
    if tax_pct is not None:
        tariff.tax_pct = tax_pct
    return tariff


def period_summary(bill: dict, days: int) -> dict:
    """Compact bill breakdown for a projection window."""
    return {
        "days": days,
        "total_kwh": round(bill["total_kwh"], 3),
        "energy_charge_inr": round(bill["energy_charge"], 2),
        "tax_inr": round(bill["tax"], 2),
        "fixed_charge_inr": round(bill["fixed"], 2),
        "estimated_bill_inr": round(bill["total"], 2),
    }
