"""BijliPredict AI - synthetic Indian household electricity consumption generator.

Produces a realistic daily `date,consumption_kwh` series so the full pipeline is
reproducible and runnable offline. The signal is a sum of interpretable
components that a sequence model can actually learn:

    * base load          (fridge, standby, lighting)
    * weekly seasonality (weekends higher - people are home)
    * annual seasonality (summer AC peak around May-June; mild winter bump)
    * slow year-over-year growth
    * Gaussian noise
    * occasional power-outage dips (common in the iAWE deployment)

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

DEFAULT_START = "2023-01-01"
DEFAULT_DAYS = 3 * 365


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
) -> pd.DataFrame:
    """Generate a synthetic daily household consumption series in kWh."""
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

    # Annual seasonality: AC-driven summer peak near day-of-year ~165 (mid-June),
    # plus a small winter bump (geysers) near the start of the year.
    doy = dates.dayofyear.to_numpy()
    summer = annual_amplitude * 0.5 * (1.0 + np.cos(2.0 * np.pi * (doy - 165) / 365.25))
    winter = 0.9 * np.exp(-0.5 * ((doy - 10) / 25.0) ** 2)

    signal = base + growth + weekly + summer + winter

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


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate synthetic household data.")
    p.add_argument("--start", default=DEFAULT_START, help="start date (YYYY-MM-DD)")
    p.add_argument("--days", type=int, default=DEFAULT_DAYS, help="number of days")
    p.add_argument("--seed", type=int, default=42)
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

    df = generate_series(start=args.start, days=args.days, seed=args.seed)
    log.info("event=generate_data days=%d seed=%d", len(df), args.seed)

    manifest = RunManifest(
        "data",
        args=vars(args),
        manifest_dir=args.manifest_dir,
        enabled=not args.no_manifest,
    )

    for path in (args.raw_out, args.processed_out):
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        manifest.add_output(Path(path).stem, out)

    sample = Path(args.sample_input)
    sample.parent.mkdir(parents=True, exist_ok=True)
    # Last 30 days, matching the model's input window.
    df.tail(30).to_csv(sample, index=False)
    manifest.add_output("sample_input", sample)

    print(f"Generated {len(df)} days: {df['date'].iloc[0]} -> {df['date'].iloc[-1]}")
    print(f"  raw        : {args.raw_out}")
    print(f"  processed  : {args.processed_out}")
    print(f"  sample     : {args.sample_input} (last 30 days)")
    print(
        f"  consumption: mean={df['consumption_kwh'].mean():.2f} kWh  "
        f"min={df['consumption_kwh'].min():.2f}  max={df['consumption_kwh'].max():.2f}"
    )

    manifest.set("n_days", len(df))
    manifest.write()
    log.info("event=run_end command=data status=ok")


if __name__ == "__main__":
    main()
