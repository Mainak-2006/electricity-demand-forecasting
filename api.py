"""BijliPredict AI - FastAPI service exposing forecast, bill and projection.

The model, scalers, feature metadata and conformal calibration are loaded **once**
at startup (a FastAPI lifespan) and reused for every request. All forecasting
goes through ``service.py``, which the CLI and Streamlit app also use, so the
service can never drift from `bijli predict`.

Endpoints:
    GET  /health      service + model metadata
    POST /forecast    7-day (or custom) kWh forecast with conformal intervals
    POST /bill        the same forecast priced under a flat / slab tariff
    POST /projection  rolling long-horizon projection + monthly / 3-month cost

Run it:
    uvicorn api:app --host 0.0.0.0 --port 8000
    # or
    python api.py

Configuration via environment variables (all optional):
    BIJLI_MODEL_PATH        default artifacts/best_model.pt
    BIJLI_SCALER_PATH       default artifacts/scaler.joblib
    BIJLI_CALIBRATION_PATH  default artifacts/calibration.json (intervals)
    BIJLI_DEVICE            auto | cpu | cuda (default cpu)
    BIJLI_API_KEY           shared secret; when set, POST routes require a
                            matching `X-API-Key` header (constant-time compare).
                            Unset -> open service (local/dev default).
    BIJLI_RATE_LIMIT        max POST requests per minute per client IP
                            (default 60; 0 disables)
    BIJLI_MAX_BODY_BYTES    reject request bodies larger than this
                            (default 65536 = 64 KiB)

`/`, `/health` and the docs stay unauthenticated so container health checks and
service discovery keep working when `BIJLI_API_KEY` is set.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import math
import os
import secrets
import threading
import time
from collections import deque
from contextlib import asynccontextmanager

import numpy as np
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

import service
from service import (
    DEFAULT_CALIBRATION_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_PRESETS_PATH,
    ServiceState,
    compute_bill,
)

SERVICE_NAME = "bijli-predict-ai"
SERVICE_VERSION = "0.1.0"

# Hard input caps. The body limit is checked on Content-Length; the history cap
# is enforced by pydantic so a chunked body cannot smuggle a huge array in.
DEFAULT_MAX_HISTORY = 2000
DEFAULT_RATE_LIMIT = 60  # requests per minute per client IP
DEFAULT_MAX_BODY_BYTES = 64 * 1024
_RATE_LIMIT_WINDOW = 60.0
_UNPROTECTED_PATHS = frozenset({"/", "/health", "/docs", "/redoc", "/openapi.json"})


# --- request / response schemas ---------------------------------------------


class TariffSpec(BaseModel):
    rate: float | None = Field(None, description="flat tariff in INR/kWh")
    slab: str | None = Field(None, description="slab tariff 'LOW-HIGH:RATE,...'")
    preset: str | None = Field(None, description="named discom preset, e.g. 'bses'")
    fixed: float | None = Field(None, description="fixed charge for the period (INR)")
    tax_pct: float | None = Field(None, description="tax/surcharge %% on energy charge")
    presets_file: str = DEFAULT_PRESETS_PATH


class _HistoryRequest(BaseModel):
    history_kwh: list[float] = Field(
        ...,
        min_length=1,
        max_length=DEFAULT_MAX_HISTORY,
        description="recent daily kWh, oldest first",
    )
    last_date: _dt.date | None = Field(None, description="date of the last history value")

    @field_validator("history_kwh")
    @classmethod
    def _check_history(cls, v: list[float]) -> list[float]:
        if not v:
            raise ValueError("history_kwh must contain at least one value")
        arr = np.asarray(v, dtype=float)
        if not np.all(np.isfinite(arr)):
            raise ValueError("history_kwh must contain only finite numbers")
        return [float(x) for x in v]


class ForecastRequest(_HistoryRequest):
    horizon: int | None = Field(None, ge=1, le=365, description="days to forecast")


class BillRequest(ForecastRequest, TariffSpec):
    pass


class ProjectionRequest(_HistoryRequest, TariffSpec):
    project_days: int = Field(90, ge=1, le=730)


class Interval(BaseModel):
    level: float
    total_kwh_lower: float
    total_kwh_upper: float


class IntervalBand(BaseModel):
    lower_inr: float
    upper_inr: float


class ForecastDay(BaseModel):
    day: int
    date: _dt.date
    kwh: float
    lower_80: float | None = None
    upper_80: float | None = None
    lower_90: float | None = None
    upper_90: float | None = None


class ForecastResponse(BaseModel):
    model: str
    horizon: int
    unit: str = "kWh"
    total_kwh: float
    forecast: list[ForecastDay]
    intervals: dict[str, Interval] = Field(default_factory=dict)


class BillDay(BaseModel):
    day: int
    date: _dt.date
    kwh: float
    cost_inr: float


class BillResponse(BaseModel):
    model: str
    horizon: int
    tariff: str
    total_kwh: float
    energy_charge_inr: float
    tax_inr: float
    fixed_charge_inr: float
    estimated_bill_inr: float
    daily: list[BillDay]
    intervals: dict[str, IntervalBand] = Field(default_factory=dict)


class PeriodSummary(BaseModel):
    days: int
    total_kwh: float
    energy_charge_inr: float
    tax_inr: float
    fixed_charge_inr: float
    estimated_bill_inr: float


class ProjectionDay(BaseModel):
    day: int
    date: _dt.date
    kwh: float
    cost_inr: float
    cumulative_cost_inr: float


class ProjectionResponse(BaseModel):
    model: str
    project_days: int
    tariff: str
    total_kwh: float
    estimated_bill_inr: float
    monthly_30d: PeriodSummary
    three_month_90d: PeriodSummary
    daily: list[ProjectionDay]


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    model: str
    input_len: int
    horizon: int
    features: list[str]
    calibration: bool
    device: str


# --- helpers ----------------------------------------------------------------


def _json_safe(obj):
    """Replace non-finite floats with null so error bodies always serialize.

    A NaN/Infinity in the request would otherwise be echoed into the 422 body
    and crash the default handler's JSON serialization.
    """
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def get_service(request: Request) -> ServiceState:
    svc = getattr(request.app.state, "svc", None)
    if svc is None:  # pragma: no cover - only if lifespan did not run
        raise HTTPException(status_code=503, detail="Service not initialised.")
    return svc


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc


class RateLimiter:
    """Sliding-window limiter keyed by client IP.

    The POST routes are sync `def`, so Starlette runs them in a worker thread;
    the lock keeps the shared window state consistent.
    """

    def __init__(self, limit: int, window: float = _RATE_LIMIT_WINDOW) -> None:
        self.limit = limit
        self.window = window
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def check(self, key: str) -> tuple[bool, float]:
        """Return ``(allowed, retry_after_seconds)`` and record the hit."""
        if self.limit <= 0:
            return True, 0.0
        now = time.monotonic()
        with self._lock:
            hits = self._hits.setdefault(key, deque())
            while hits and now - hits[0] > self.window:
                hits.popleft()
            if len(hits) >= self.limit:
                return False, self.window - (now - hits[0])
            hits.append(now)
        return True, 0.0


def require_api_key(request: Request) -> None:
    """Enforce ``X-API-Key`` when the app was configured with a shared secret."""
    expected = getattr(request.app.state, "api_key", None)
    if not expected:
        return
    provided = request.headers.get("X-API-Key", "")
    if not secrets.compare_digest(provided.encode("utf-8"), expected.encode("utf-8")):
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key.",
            headers={"WWW-Authenticate": "X-API-Key"},
        )


def _resolve_tariff(spec: TariffSpec):
    try:
        return service.resolve_tariff(
            rate=spec.rate,
            slab=spec.slab,
            preset=spec.preset,
            fixed=spec.fixed,
            tax_pct=spec.tax_pct,
            presets_file=spec.presets_file,
        )
    except (KeyError, FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def _forecast(svc: ServiceState, history: list[float], days: int, last_date: _dt.date):
    try:
        return service.run_forecast(svc, history, days, last_date)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


# --- routes -----------------------------------------------------------------


def build_router(app: FastAPI) -> None:
    @app.get("/", include_in_schema=False)
    def index() -> dict:
        return {
            "service": SERVICE_NAME,
            "version": SERVICE_VERSION,
            "endpoints": ["/health", "/forecast", "/bill", "/projection", "/docs"],
        }

    @app.get("/health", response_model=HealthResponse)
    def health(svc: ServiceState = Depends(get_service)) -> HealthResponse:
        return HealthResponse(
            status="ok",
            service=SERVICE_NAME,
            version=SERVICE_VERSION,
            model=svc.model_name,
            input_len=svc.input_len,
            horizon=svc.horizon,
            features=list(svc.feature_columns),
            calibration=svc.calib is not None,
            device=str(svc.device),
        )

    @app.post("/forecast", response_model=ForecastResponse)
    def forecast(
        req: ForecastRequest,
        svc: ServiceState = Depends(get_service),
        _auth: None = Depends(require_api_key),
    ) -> ForecastResponse:
        horizon = req.horizon or svc.horizon
        last_date = req.last_date or _dt.date.today()
        preds, dates = _forecast(svc, req.history_kwh, horizon, last_date)
        forecast_kwh = preds[:horizon]
        iv = service.intervals(forecast_kwh, svc.calib)

        days = []
        for i, (d, kwh) in enumerate(zip(dates, forecast_kwh, strict=True), start=1):
            day = ForecastDay(day=i, date=d.date(), kwh=round(float(kwh), 3))
            if 80 in iv:
                day.lower_80 = round(float(iv[80][0][i - 1]), 3)
                day.upper_80 = round(float(iv[80][1][i - 1]), 3)
            if 90 in iv:
                day.lower_90 = round(float(iv[90][0][i - 1]), 3)
                day.upper_90 = round(float(iv[90][1][i - 1]), 3)
            days.append(day)

        bands = {
            str(pct): Interval(
                level=pct / 100,
                total_kwh_lower=round(float(np.sum(iv[pct][0])), 3),
                total_kwh_upper=round(float(np.sum(iv[pct][1])), 3),
            )
            for pct in sorted(iv)
        }
        return ForecastResponse(
            model=svc.model_name,
            horizon=horizon,
            total_kwh=round(float(sum(forecast_kwh)), 3),
            forecast=days,
            intervals=bands,
        )

    @app.post("/bill", response_model=BillResponse)
    def bill(
        req: BillRequest,
        svc: ServiceState = Depends(get_service),
        _auth: None = Depends(require_api_key),
    ) -> BillResponse:
        horizon = req.horizon or svc.horizon
        last_date = req.last_date or _dt.date.today()
        tariff = _resolve_tariff(req)
        preds, dates = _forecast(svc, req.history_kwh, horizon, last_date)
        forecast_kwh = preds[:horizon]
        result = compute_bill(forecast_kwh, tariff)
        iv = service.intervals(forecast_kwh, svc.calib)

        daily = [
            BillDay(day=i, date=d.date(), kwh=round(float(k), 3), cost_inr=round(float(c), 2))
            for i, (d, k, c) in enumerate(
                zip(dates, forecast_kwh, result["daily_costs"], strict=True), start=1
            )
        ]
        bands = {
            str(pct): IntervalBand(
                lower_inr=round(compute_bill(iv[pct][0], tariff)["total"], 2),
                upper_inr=round(compute_bill(iv[pct][1], tariff)["total"], 2),
            )
            for pct in sorted(iv)
        }
        return BillResponse(
            model=svc.model_name,
            horizon=horizon,
            tariff=result["tariff"],
            total_kwh=round(result["total_kwh"], 3),
            energy_charge_inr=round(result["energy_charge"], 2),
            tax_inr=round(result["tax"], 2),
            fixed_charge_inr=round(result["fixed"], 2),
            estimated_bill_inr=round(result["total"], 2),
            daily=daily,
            intervals=bands,
        )

    @app.post("/projection", response_model=ProjectionResponse)
    def projection(
        req: ProjectionRequest,
        svc: ServiceState = Depends(get_service),
        _auth: None = Depends(require_api_key),
    ) -> ProjectionResponse:
        last_date = req.last_date or _dt.date.today()
        tariff = _resolve_tariff(req)
        preds, dates = _forecast(svc, req.history_kwh, req.project_days, last_date)
        result = compute_bill(preds, tariff)
        monthly = compute_bill(preds[: service.MONTH_DAYS], tariff)
        quarterly = compute_bill(preds[: service.QUARTER_DAYS], tariff)

        cumulative = np.cumsum(result["daily_costs"])
        daily = [
            ProjectionDay(
                day=i,
                date=d.date(),
                kwh=round(float(k), 3),
                cost_inr=round(float(c), 2),
                cumulative_cost_inr=round(float(cum), 2),
            )
            for i, (d, k, c, cum) in enumerate(
                zip(dates, preds, result["daily_costs"], cumulative, strict=True), start=1
            )
        ]
        return ProjectionResponse(
            model=svc.model_name,
            project_days=req.project_days,
            tariff=result["tariff"],
            total_kwh=round(result["total_kwh"], 3),
            estimated_bill_inr=round(result["total"], 2),
            monthly_30d=service.period_summary(monthly, min(service.MONTH_DAYS, len(preds))),
            three_month_90d=service.period_summary(
                quarterly, min(service.QUARTER_DAYS, len(preds))
            ),
            daily=daily,
        )


def create_app(
    *,
    model_path: str | os.PathLike | None = None,
    scaler_path: str | os.PathLike | None = None,
    calibration_path: str | os.PathLike | None = None,
    weather_file: str | None = None,
    device: str | None = None,
    api_key: str | None = None,
    rate_limit: int | None = None,
    max_body_bytes: int | None = None,
) -> FastAPI:
    """Build the FastAPI app; artifacts resolve from args, then env, then defaults."""
    model_path = model_path or os.environ.get("BIJLI_MODEL_PATH", DEFAULT_MODEL_PATH)
    scaler_path = scaler_path or os.environ.get("BIJLI_SCALER_PATH")
    if calibration_path is None:
        calibration_path = os.environ.get("BIJLI_CALIBRATION_PATH", DEFAULT_CALIBRATION_PATH)
    device = device or os.environ.get("BIJLI_DEVICE", "cpu")
    if api_key is None:
        api_key = os.environ.get("BIJLI_API_KEY") or None
    if rate_limit is None:
        rate_limit = _env_int("BIJLI_RATE_LIMIT", DEFAULT_RATE_LIMIT)
    if max_body_bytes is None:
        max_body_bytes = _env_int("BIJLI_MAX_BODY_BYTES", DEFAULT_MAX_BODY_BYTES)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.svc = service.load_service(
            model_path=model_path,
            scaler_path=scaler_path,
            calibration_path=calibration_path,
            weather_file=weather_file,
            device=device,
        )
        yield

    app = FastAPI(
        title="BijliPredict AI",
        version=SERVICE_VERSION,
        summary="Household electricity forecasting, billing and cost projection.",
        lifespan=lifespan,
    )
    app.state.api_key = api_key
    app.state.rate_limiter = RateLimiter(rate_limit)
    app.state.max_body_bytes = int(max_body_bytes)

    @app.middleware("http")
    async def _guard_requests(request: Request, call_next):
        if request.method in {"POST", "PUT", "PATCH"}:
            limit = request.app.state.max_body_bytes
            declared = request.headers.get("content-length")
            if declared is not None and declared.isdigit() and int(declared) > limit:
                return JSONResponse(
                    status_code=413,
                    content={"detail": f"Request body exceeds the {limit}-byte limit."},
                )
        limiter = request.app.state.rate_limiter
        if limiter is not None and request.url.path not in _UNPROTECTED_PATHS:
            client = request.client.host if request.client else "unknown"
            allowed, retry_after = limiter.check(client)
            if not allowed:
                return JSONResponse(
                    status_code=429,
                    content={"detail": "Rate limit exceeded. Slow down."},
                    headers={"Retry-After": str(max(1, math.ceil(retry_after)))},
                )
        return await call_next(request)

    @app.exception_handler(RequestValidationError)
    async def _validation_handler(_request: Request, exc: RequestValidationError) -> JSONResponse:
        detail = _json_safe(jsonable_encoder(exc.errors()))
        return JSONResponse(status_code=422, content={"detail": detail})

    build_router(app)
    return app


app = create_app()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the BijliPredict AI API server.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    return parser


def main(args: argparse.Namespace | None = None) -> None:
    import uvicorn

    if args is None:
        args = build_parser().parse_args()
    uvicorn.run("api:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
