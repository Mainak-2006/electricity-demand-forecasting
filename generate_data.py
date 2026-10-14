"""BijliPredict AI - synthetic Indian household electricity consumption generator.

Produces a realistic daily `date,consumption_kwh` series so the full pipeline is
reproducible and runnable offline. The signal is a sum of interpretable
components that a sequence model can actually learn:

    * base load          (fridge, standby, lighting)
    * weekly seasonality (weekends higher - people are home)
    * temperature-driven cooling/heating load (CDD/HDD, see below)
    * slow year-over-year growth
    * Gaussian noise
    * occasional power-outage dips (common in the iAWE deployment)

**Weather coupling.** By default the load is driven by a synthetic temperature
series (a Delhi-like seasonal cycle) written to
``data/raw/weather/<city>.csv`` in the format :mod:`weather` reads. Cooling
degree days (T above ``cooling_base``) drive the summer AC peak and heating
degree days (T below ``heating_base``) drive the winter geyser bump, so the
`--features weather` build of the pipeline has a real signal to learn. Pass
``--no-couple-weather`` to fall back to the older fixed day-of-year seasonality.

**Households.** ``--households N`` generates N heterogeneous households in the
long format ``date,household,consumption_kwh`` (shared city weather, per-home
base load / amplitudes / outage rates). ``N=1`` (the default) keeps the original
wide format.

This is NOT a substitute for real meter data; it exists so the experiment can be
run end-to-end. Drop a real CSV into data/raw/ and point preprocessing at it, or
use iawe_ingest.py for the real iAWE dataset.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from logging_utils import add_log_level_argument, get_logger, setup_logging
from manifest import RunManifest, add_manifest_arguments
from weather import DEFAULT_CACHE_DIR, resolve_city

DEFAULT_START = "2023-01-01"
DEFAULT_DAYS = 3 * 365
DEFAULT_CITY = "delhi"

# Temperature coupling defaults (Delhi-like household behaviour).
DEFAULT_COOLING_BASE = 24.0  # AC starts above this daily mean temperature
DEFAULT_HEATING_BASE = 18.0  # geysers / room heaters below this
DEFAULT_COOLING_FACTOR = 0.35  # kWh per cooling degree day
DEFAULT_HEATING_FACTOR = 0.28  # kWh per heating degree day

# Synthetic temperature seasonality: daily-mean ~24.5C, ~11C amplitude.
DEFAULT_TEMP_MEAN = 24.5
DEFAULT_TEMP_AMPLITUDE = 11.0
DEFAULT_TEMP_PEAK_DOY = 155  # early June
DEFAULT_TEMP_NOISE = 1.2


def generate_temperature(
    start: str = DEFAULT_START,
    days: int = DEFAULT_DAYS,
    seed: int = 42,
    mean: float = DEFAULT_TEMP_MEAN,
    amplitude: float = DEFAULT_TEMP_AMPLITUDE,
    peak_doy: int = DEFAULT_TEMP_PEAK_DOY,
    noise_std: float = DEFAULT_TEMP_NOISE,
) -> pd.DataFrame:
    """Generate a Delhi-like daily temperature series.

    Returns a frame indexed by date with ``temp_mean`` / ``temp_max`` /
    ``temp_min`` columns - exactly the shape :func:`weather.load_weather` reads.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start=start, periods=days, freq="D")
    doy = dates.dayofyear.to_numpy()

    seasonal = mean + amplitude * np.cos(2.0 * np.pi * (doy - peak_doy) / 365.25)
    temp_mean = seasonal + rng.normal(0.0, noise_std, size=days)
    # Diurnal spread: daytime peak above, night-time minimum below the mean.
    temp_max = temp_mean + 5.5 + rng.normal(0.0, 0.6, size=days)
    temp_min = temp_mean - 5.0 + rng.normal(0.0, 0.6, size=days)

    frame = pd.DataFrame(
        {
            "temp_mean": temp_mean,
            "temp_max": temp_max,
            "temp_min": temp_min,
        },
        index=pd.DatetimeIndex(dates, name="date"),
    )
    return frame.round(2)


def generate_series(
    start: str = DEFAULT_START,
    days: int = DEFAULT_DAYS,
    seed: int = 42,
    base_load: float = 3.5,
    weekly_amplitude: float = 1.4,
    annual_amplitude: float = 3.2,
    noise_std: float = 0.55,
    outage_prob: float = 0.03,
    outage_factor: float = 0.35,
    temperature: pd.DataFrame | None = None,
    couple_weather: bool = True,
    cooling_base: float = DEFAULT_COOLING_BASE,
    heating_base: float = DEFAULT_HEATING_BASE,
    cooling_factor: float = DEFAULT_COOLING_FACTOR,
    heating_factor: float = DEFAULT_HEATING_FACTOR,
) -> pd.DataFrame:
    """Generate a synthetic daily household consumption series in kWh.

    With ``couple_weather`` the seasonal component is driven by ``temperature``
    (generated internally when not supplied); otherwise the original fixed
    day-of-year ``annual_amplitude`` seasonality is used.
    """
    rng = np.random.default_rng(seed)

    dates = pd.date_range(start=start, periods=days, freq="D")
    t = np.arange(days, dtype=float)

    base = np.full(days, base_load)

    # Slow growth: ~+0.15 kWh/year, applied smoothly.
    growth = 0.15 * (t / 365.0)

    # Weekly seasonality: weekends (Sat=5, Sun=6) higher than weekdays.
    dow = dates.dayofweek.to_numpy()
    weekend = np.where(dow >= 5, 1.0, 0.0)
    weekly = weekly_amplitude * (weekend - 0.28)

    if couple_weather:
        if temperature is None:
            temperature = generate_temperature(start=start, days=days, seed=seed + 9999)
        temp_mean = temperature["temp_mean"].reindex(dates).to_numpy(dtype=float)
        if np.isnan(temp_mean).any():
            raise ValueError("temperature does not cover every generated date")
        cooling = cooling_factor * np.clip(temp_mean - cooling_base, 0.0, None)
        heating = heating_factor * np.clip(heating_base - temp_mean, 0.0, None)
        seasonal = cooling + heating
    else:
        doy = dates.dayofyear.to_numpy()
        summer = annual_amplitude * 0.5 * (1.0 + np.cos(2.0 * np.pi * (doy - 165) / 365.25))
        winter = 0.9 * np.exp(-0.5 * ((doy - 10) / 25.0) ** 2)
        seasonal = summer + winter

    signal = base + growth + weekly + seasonal

    # Gaussian noise (multiplicative-ish -> proportional to load).
    signal = signal + rng.normal(0.0, noise_std, size=days)
    signal = signal * (1.0 + rng.normal(0.0, 0.04, size=days))

    # Occasional power outages / trips: sudden low-consumption days.
    outage_mask = rng.random(days) < outage_prob
    signal = np.where(outage_mask, signal * outage_factor, signal)

    # Household consumption cannot be negative.
    consumption = np.clip(signal, 0.2, None)

    df = pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "consumption_kwh": np.round(consumption, 3),
        }
    )
    return df


def _household_params(rng: np.random.Generator) -> dict:
    """Draw heterogeneous per-household generator parameters."""
    return {
        "base_load": float(rng.uniform(2.2, 5.5)),
        "weekly_amplitude": float(rng.uniform(0.8, 2.0)),
        "annual_amplitude": float(rng.uniform(2.0, 4.5)),
        "noise_std": float(rng.uniform(0.35, 0.8)),
        "outage_prob": float(rng.uniform(0.01, 0.06)),
        "cooling_factor": float(rng.uniform(0.25, 0.45)),
        "heating_factor": float(rng.uniform(0.18, 0.38)),
    }


def generate_households(
    start: str = DEFAULT_START,
    days: int = DEFAULT_DAYS,
    n_households: int = 1,
    seed: int = 42,
    temperature: pd.DataFrame | None = None,
    couple_weather: bool = True,
) -> pd.DataFrame:
    """Generate one or more households.

    ``n_households == 1`` returns the wide ``date,consumption_kwh`` frame;
    more returns the long ``date,household,consumption_kwh`` frame.
    """
    if n_households < 1:
        raise ValueError("n_households must be >= 1")
    if temperature is None:
        temperature = generate_temperature(start=start, days=days, seed=seed + 9999)

    if n_households == 1:
        return generate_series(
            start=start,
            days=days,
            seed=seed,
            temperature=temperature,
            couple_weather=couple_weather,
        )

    rng = np.random.default_rng(seed + 12345)
    frames = []
    for i in range(n_households):
        params = _household_params(rng)
        series = generate_series(
            start=start,
            days=days,
            seed=seed + i,
            temperature=temperature,
            couple_weather=couple_weather,
            **params,
        )
        series.insert(1, "household", f"h{i + 1:02d}")
        frames.append(series)
    return pd.concat(frames, ignore_index=True)


def _write_weather(temperature: pd.DataFrame, path: Path) -> None:
    out = temperature.copy()
    out.index = out.index.strftime("%Y-%m-%d")
    out.index.name = "date"
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate synthetic household data.")
    p.add_argument("--start", default=DEFAULT_START, help="start date (YYYY-MM-DD)")
    p.add_argument("--days", type=int, default=DEFAULT_DAYS, help="number of days")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--households", type=int, default=1, help="number of households (1 = wide format)"
    )
    p.add_argument("--city", default=DEFAULT_CITY, help="city for the synthetic temperature")
    p.add_argument(
        "--no-couple-weather",
        action="store_true",
        help="use fixed day-of-year seasonality instead of temperature-driven load",
    )
    p.add_argument(
        "--weather-out",
        default=None,
        help=f"synthetic temperature CSV (default: {DEFAULT_CACHE_DIR}/<city>.csv)",
    )
    p.add_argument("--raw-out", default="data/raw/synthetic_household.csv", help="raw CSV path")
    p.add_argument(
        "--processed-out",
        default="data/processed/daily_consumption.csv",
        help="processed CSV path (ready for training)",
    )
    p.add_argument(
        "--sample-input",
        default="data/sample_input.csv",
        help="last-30-day history used by predict.py",
    )
    add_log_level_argument(p)
    add_manifest_arguments(p)
    return p


def main(args: argparse.Namespace | None = None) -> None:
    if args is None:
        args = build_parser().parse_args()
    setup_logging(args.log_level)
    log = get_logger("data")

    couple = not args.no_couple_weather
    city, _, _ = resolve_city(args.city)
    log.info(
        "event=generate_data days=%d seed=%d households=%d couple_weather=%s",
        args.days,
        args.seed,
        args.households,
        couple,
    )

    manifest = RunManifest(
        "data",
        args=vars(args),
        manifest_dir=args.manifest_dir,
        enabled=not args.no_manifest,
    )

    temperature = generate_temperature(start=args.start, days=args.days, seed=args.seed + 9999)
    weather_path = Path(args.weather_out or f"{DEFAULT_CACHE_DIR}/{city}.csv")
    _write_weather(temperature, weather_path)
    manifest.add_output("weather", weather_path)

    df = generate_households(
        start=args.start,
        days=args.days,
        n_households=args.households,
        seed=args.seed,
        temperature=temperature,
        couple_weather=couple,
    )

    for path in (args.raw_out, args.processed_out):
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        manifest.add_output(Path(path).stem, out)

    sample = Path(args.sample_input)
    sample.parent.mkdir(parents=True, exist_ok=True)
    # Last 30 days of one household, matching the model's input window.
    wide = df[df["household"] == "h01"] if "household" in df.columns else df
    wide.tail(30)[["date", "consumption_kwh"]].to_csv(sample, index=False)
    manifest.add_output("sample_input", sample)

    n_households = int(df["household"].nunique()) if "household" in df.columns else 1
    print(f"Generated {len(df)} rows over {args.days} days ({n_households} household(s))")
    print(f"  raw        : {args.raw_out}")
    print(f"  processed  : {args.processed_out}")
    print(f"  weather    : {weather_path}")
    print(f"  sample     : {args.sample_input} (last 30 days)")
    print(
        f"  consumption: mean={df['consumption_kwh'].mean():.2f} kWh  "
        f"min={df['consumption_kwh'].min():.2f}  max={df['consumption_kwh'].max():.2f}"
    )

    manifest.set("n_days", args.days)
    manifest.set("n_households", n_households)
    manifest.write()
    log.info("event=run_end command=data status=ok")


if __name__ == "__main__":
    main()
