"""BijliPredict AI - Open-Meteo weather client with an on-disk cache.

Fetches daily temperature (mean / max / min) for Indian cities from the
Open-Meteo **archive** (historical reanalysis) and **forecast** APIs, caches the
result at ``data/raw/weather/{city}.csv`` and exposes a single helper that fills
any date it cannot fetch from the archive/forecast with **day-of-year
climatology**.

Why two APIs? The archive lags the present by ~5 days and the forecast reaches
only ~16 days. Anything further out (e.g. the 90-day cost projection) therefore
falls back to climatological normals.

Public API:
    resolve_city(city)                       -> (name, lat, lon)
    load_weather(path)                       -> DataFrame(date, temp_mean/max/min)
    fetch(city, start, end, ...)             -> available temps (+ cache update)
    get_climatology(city, ...)               -> doy-indexed normals
    climatology_for_dates(clim, dates)       -> normals aligned to dates
    get_temperature(city, dates, ...)        -> temps for `dates`, gaps filled

Nothing here is required to run offline: pass ``weather_file=`` to read a local
CSV, or inject a ``fetcher`` in tests instead of hitting the network.
"""

from __future__ import annotations

import argparse
import csv
import json
import urllib.parse
import urllib.request
from datetime import date as _date
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_CACHE_DIR = "data/raw/weather"
DEFAULT_TIMEZONE = "Asia/Kolkata"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
DAILY_VARS = "temperature_2m_mean,temperature_2m_max,temperature_2m_min"
ARCHIVE_LAG_DAYS = 6  # archive typically trails the present by ~5 days
FORECAST_PAST_DAYS = 5
FORECAST_DAYS = 16

TEMP_COLUMNS = ["temp_mean", "temp_max", "temp_min"]

# Canonical city -> (latitude, longitude). Indian metros the project targets.
CITY_COORDS: dict[str, tuple[float, float]] = {
    "delhi": (28.6139, 77.2090),
    "mumbai": (19.0760, 72.8777),
    "bengaluru": (12.9716, 77.5946),
    "chennai": (13.0827, 80.2707),
    "kolkata": (22.5726, 88.3639),
    "hyderabad": (17.3850, 78.4867),
    "pune": (18.5204, 73.8567),
    "ahmedabad": (23.0225, 72.5714),
    "jaipur": (26.9124, 75.7873),
    "lucknow": (26.8467, 80.9462),
}

# Common alternative spellings -> canonical key.
CITY_ALIASES = {
    "new delhi": "delhi",
    "ncr": "delhi",
    "bangalore": "bengaluru",
    "bombay": "mumbai",
    "madras": "chennai",
    "calcutta": "kolkata",
}


def available_cities() -> list[str]:
    return sorted(CITY_COORDS)


def resolve_city(city: str) -> tuple[str, float, float]:
    """Map a city name/alias to ``(canonical_name, latitude, longitude)``."""
    key = (city or "").strip().lower()
    key = CITY_ALIASES.get(key, key)
    if key not in CITY_COORDS:
        raise ValueError(f"Unknown city '{city}'. Choose from: {', '.join(available_cities())}.")
    lat, lon = CITY_COORDS[key]
    return key, lat, lon


# --- HTTP -----------------------------------------------------------------


def _http_get_json(url: str, params: dict, timeout: float = 30.0) -> dict:
    """GET ``url?params`` and parse the JSON body.

    Prefers httpx when installed (the ``weather`` extra); falls back to the
    stdlib so the module keeps working with zero extra dependencies.
    """
    headers = {"User-Agent": "bijli-predict-ai"}
    try:
        import httpx
    except ImportError:
        query = urllib.parse.urlencode(params)
        req = urllib.request.Request(f"{url}?{query}", headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    resp = httpx.get(url, params=params, headers=headers, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def _parse_daily(payload: dict) -> pd.DataFrame:
    """Turn an Open-Meteo `daily` payload into a date-indexed temperature frame."""
    daily = payload.get("daily") or {}
    times = daily.get("time")
    if not times:
        raise ValueError("Open-Meteo response has no 'daily.time' field.")
    data: dict[str, object] = {"date": pd.to_datetime(times)}
    for name, key in (
        ("temp_mean", "temperature_2m_mean"),
        ("temp_max", "temperature_2m_max"),
        ("temp_min", "temperature_2m_min"),
    ):
        data[name] = daily.get(key, np.full(len(times), np.nan))
    df = pd.DataFrame(data).set_index("date").sort_index()
    for col in TEMP_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.index.name = "date"
    return df


def _client_params(city: str) -> tuple[str, dict]:
    name, lat, lon = resolve_city(city)
    return name, {
        "latitude": lat,
        "longitude": lon,
        "daily": DAILY_VARS,
        "timezone": DEFAULT_TIMEZONE,
    }


def fetch_archive(
    city: str,
    start,
    end,
    *,
    fetcher=_http_get_json,
) -> pd.DataFrame:
    """Fetch historical daily temperatures for ``[start, end]`` from the archive."""
    _, base = _client_params(city)
    params = {
        **base,
        "start_date": str(pd.Timestamp(start).date()),
        "end_date": str(pd.Timestamp(end).date()),
    }
    return _parse_daily(fetcher(ARCHIVE_URL, params))


def fetch_forecast(
    city: str,
    *,
    fetcher=_http_get_json,
    past_days: int = FORECAST_PAST_DAYS,
    forecast_days: int = FORECAST_DAYS,
) -> pd.DataFrame:
    """Fetch recent + upcoming daily temperatures from the forecast API."""
    _, base = _client_params(city)
    params = {**base, "past_days": past_days, "forecast_days": forecast_days}
    return _parse_daily(fetcher(FORECAST_URL, params))


# --- cache ----------------------------------------------------------------


def _cache_path(city: str, cache_dir: str | Path) -> Path:
    name, _, _ = resolve_city(city)
    return Path(cache_dir) / f"{name}.csv"


def load_weather(path: str | Path) -> pd.DataFrame:
    """Read a weather CSV (columns: date + temperature columns)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Weather file not found: {path}")
    raw = pd.read_csv(path)
    lower = {c.lower().strip(): c for c in raw.columns}
    date_col = next((lower[k] for k in ("date", "datetime", "time", "ds") if k in lower), None)
    if date_col is None:
        raise ValueError(f"Weather file {path} has no date column: {list(raw.columns)}")
    df = pd.DataFrame({"date": pd.to_datetime(raw[date_col], errors="coerce")})
    for col in TEMP_COLUMNS:
        if col in lower:
            df[col] = pd.to_numeric(raw[lower[col]], errors="coerce")
        elif col.replace("_", "-") in lower:
            df[col] = pd.to_numeric(raw[lower[col.replace("_", "-")]], errors="coerce")
    if not any(c in df.columns for c in TEMP_COLUMNS):
        raise ValueError(f"Weather file {path} has no temperature columns.")
    if "temp_mean" not in df.columns:
        present = [c for c in TEMP_COLUMNS if c in df.columns]
        df["temp_mean"] = df[present].mean(axis=1)
    df = df.dropna(subset=["date"]).set_index("date").sort_index()
    df.index.name = "date"
    for col in TEMP_COLUMNS:
        if col not in df.columns:
            df[col] = df["temp_mean"]
    return df[TEMP_COLUMNS]


def _read_cache(city: str, cache_dir: str | Path) -> pd.DataFrame:
    path = _cache_path(city, cache_dir)
    if not path.exists():
        return pd.DataFrame(
            index=pd.DatetimeIndex([], name="date"), columns=TEMP_COLUMNS, dtype=float
        )
    return load_weather(path)


def _write_cache(city: str, df: pd.DataFrame, cache_dir: str | Path) -> None:
    path = _cache_path(city, cache_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    out.index = out.index.strftime("%Y-%m-%d")
    out.to_csv(path, quoting=csv.QUOTE_MINIMAL)


def fetch(
    city: str,
    start,
    end,
    *,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    fetcher=_http_get_json,
    refresh: bool = False,
    today=None,
) -> pd.DataFrame:
    """Return cached temps for ``[start, end]`` (fetching whatever is missing).

    Missing dates whose timestamp is at or before ``today - ARCHIVE_LAG_DAYS``
    come from the archive API; later ones come from the forecast API. Dates the
    APIs cannot serve are simply absent from the result (``get_temperature``
    fills them from climatology).
    """
    today_ts = pd.Timestamp(today or _date.today()).normalize()
    needed = pd.date_range(start, end, freq="D")
    cache = pd.DataFrame() if refresh else _read_cache(city, cache_dir)
    missing = needed.difference(cache.index)

    if len(missing):
        frames = []
        archive_end = today_ts - pd.Timedelta(days=ARCHIVE_LAG_DAYS)
        arch_dates = missing[missing <= archive_end]
        if len(arch_dates):
            frames.append(fetch_archive(city, arch_dates.min(), arch_dates.max(), fetcher=fetcher))
        if len(missing[missing > archive_end]):
            frames.append(fetch_forecast(city, fetcher=fetcher))
        if frames:
            cache = pd.concat([cache, *frames]).sort_index()
            cache = cache[~cache.index.duplicated(keep="last")]
            _write_cache(city, cache, cache_dir)

    return cache.reindex(needed)


# --- climatology ----------------------------------------------------------


def climatology(archive: pd.DataFrame) -> pd.DataFrame:
    """Average temperature by day-of-year (1..366), from a multi-year archive."""
    if archive.empty:
        raise ValueError("Cannot compute climatology from an empty archive.")
    df = archive.copy()
    df["_doy"] = df.index.dayofyear
    clim = df.groupby("_doy")[TEMP_COLUMNS].mean()
    clim.index.name = "doy"
    return clim


def climatology_for_dates(clim: pd.DataFrame, dates) -> pd.DataFrame:
    """Look up day-of-year normals for arbitrary ``dates``."""
    idx = pd.DatetimeIndex(pd.to_datetime(list(dates)))
    vals = clim.reindex(idx.dayofyear)
    vals.index = idx
    vals.index.name = "date"
    return vals


def get_climatology(
    city: str,
    *,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    fetcher=_http_get_json,
    years: int = 5,
    today=None,
    refresh: bool = False,
) -> pd.DataFrame:
    """Day-of-year temperature normals from ``years`` of archived history."""
    today_ts = pd.Timestamp(today or _date.today()).normalize()
    end = today_ts - pd.Timedelta(days=ARCHIVE_LAG_DAYS)
    start = end - pd.Timedelta(days=int(365.25 * years))
    archive = fetch(
        city, start, end, cache_dir=cache_dir, fetcher=fetcher, refresh=refresh, today=today_ts
    )
    return climatology(archive.dropna(how="all"))


def get_temperature(
    city: str,
    dates,
    *,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    fetcher=_http_get_json,
    weather_file: str | Path | None = None,
    today=None,
) -> pd.DataFrame:
    """Temperature for every requested date, gaps filled from climatology.

    Precedence per date: ``weather_file`` (if given) > archive/forecast cache >
    day-of-year climatology.
    """
    idx = pd.DatetimeIndex(pd.to_datetime(list(dates)))
    idx.name = "date"

    if weather_file is not None:
        base = load_weather(weather_file)
    else:
        base = fetch(city, idx.min(), idx.max(), cache_dir=cache_dir, fetcher=fetcher, today=today)

    result = base.reindex(idx)
    missing = result["temp_mean"].isna()
    if missing.any():
        clim = get_climatology(city, cache_dir=cache_dir, fetcher=fetcher, today=today)
        filled = climatology_for_dates(clim, idx[missing])
        result.loc[missing, TEMP_COLUMNS] = filled[TEMP_COLUMNS].to_numpy()
    # Last resort for dates the classifier could not serve: mean over all temps.
    if result["temp_mean"].isna().any() and not base.empty:
        result = result.fillna(base[TEMP_COLUMNS].mean(numeric_only=True))
    return result


# --- CLI ------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Fetch and cache Open-Meteo temperatures.")
    p.add_argument("--city", default="delhi", help=f"one of: {', '.join(available_cities())}")
    p.add_argument("--start", default=None, help="start date (YYYY-MM-DD)")
    p.add_argument("--end", default=None, help="end date (YYYY-MM-DD)")
    p.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    p.add_argument("--refresh", action="store_true", help="ignore the cache and refetch")
    return p


def main(args: argparse.Namespace | None = None) -> None:
    if args is None:
        args = build_parser().parse_args()
    name, lat, lon = resolve_city(args.city)
    end = args.end or str(_date.today())
    start = args.start or str(pd.Timestamp(end) - pd.Timedelta(days=30))
    df = fetch(args.city, start, end, cache_dir=args.cache_dir, refresh=args.refresh)
    print(f"{name} ({lat:.3f}, {lon:.3f})  {start} -> {end}: {len(df.dropna())} days")
    if not df.empty:
        print(df.dropna().tail(7).to_string())
    print(f"cache: {_cache_path(args.city, args.cache_dir)}")


if __name__ == "__main__":
    main()
