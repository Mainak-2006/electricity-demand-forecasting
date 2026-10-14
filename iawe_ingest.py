"""BijliPredict AI - iAWE ingestion helper.

Converts the real iAWE (Indian Dataset for Ambient Water and Energy) electricity
files into the `date,consumption_kwh` daily format used by the pipeline.

How to get the data (done once, outside the repo):

  1. Visit https://iawe.github.io/ and download `electricity.tar.gz`.
  2. Extract it into data/raw/iawe/  so you end up with e.g.
         data/raw/iawe/electricity/*.csv

Each CSV has columns: timestamp, W, VAR, VA, f, V, PF, A where W is instantaneous
*active power* in watts. Samples are irregular (no rows while idle), so we
integrate power over the elapsed time of each row to get energy:

        energy_kWh = sum(W_i * dt_hours_i) / 1000

and then aggregate to a daily total. The aggregate "main meter" file(s) are
auto-detected by filename; override with --file.

NOTE: iAWE covers only ~73 days for a single New Delhi home. That is far too
short for a reliable nationwide model - treat results as a smoke test, not a
production forecast. This is why generate_data.py exists as the default source.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

# Multi-word fragments that identify the aggregate/main meter.
MAIN_METER_PHRASES = ("main meter", "main_meter", "mainmeter")
# Single-word tokens that identify the aggregate/main meter.
MAIN_METER_TOKENS = frozenset({"meter", "aggregate", "total", "main"})
# Multi-word fragments that identify an individual appliance.
APPLIANCE_PHRASES = ("air condition", "air-condition")
# Single-word tokens that must never be treated as the whole-home meter.
# NOTE: matched as whole tokens, so "ac" no longer matches "trace"/"carbon".
APPLIANCE_TOKENS = frozenset(
    {
        "fridge",
        "refrigerator",
        "ac",
        "washing",
        "washer",
        "laptop",
        "iron",
        "kitchen",
        "tv",
        "television",
        "water",
        "geyser",
    }
)


def _read_power_csv(path: Path) -> pd.DataFrame:
    """Read one iAWE CSV and return (datetime, W) with numeric columns."""
    df = pd.read_csv(path)
    cols = {c.lower().strip(): c for c in df.columns}

    ts_col = next((cols[k] for k in ("timestamp", "time", "datetime") if k in cols), None)
    w_col = next((cols[k] for k in ("w", "power", "active_power") if k in cols), None)
    if ts_col is None or w_col is None:
        raise ValueError(f"{path.name}: need 'timestamp' and 'W' columns, got {list(df.columns)}")

    ts = df[ts_col]
    # iAWE timestamps are unix epoch seconds; fall back to generic parsing.
    ts_num = pd.to_numeric(ts, errors="coerce")
    if ts_num.notna().mean() > 0.9:
        dt = pd.to_datetime(ts_num, unit="s", utc=True).dt.tz_convert("Asia/Kolkata")
    else:
        dt = pd.to_datetime(ts, errors="coerce", utc=True).dt.tz_convert("Asia/Kolkata")

    out = pd.DataFrame(
        {"datetime": dt.dt.tz_localize(None), "W": pd.to_numeric(df[w_col], errors="coerce")}
    )
    return out.dropna(subset=["datetime", "W"]).sort_values("datetime")


def _tokenize(name: str) -> set[str]:
    """Split a filename into lowercase alphanumeric tokens."""
    return {t for t in re.split(r"[^a-z0-9]+", name.lower()) if t}


def _is_main_meter(name: str) -> bool:
    low = name.lower()
    tokens = _tokenize(name)
    if any(p in low for p in APPLIANCE_PHRASES) or (tokens & APPLIANCE_TOKENS):
        return False
    if any(p in low for p in MAIN_METER_PHRASES) or (tokens & MAIN_METER_TOKENS):
        return True
    # Bare numeric filenames (e.g. "1.csv") are often the aggregate meters.
    return Path(name).stem.isdigit()


def _integrate_to_daily(df: pd.DataFrame) -> pd.Series:
    """Integrate instantaneous W over elapsed time -> daily kWh."""
    df = df.sort_values("datetime")
    dt_hours = df["datetime"].diff().dt.total_seconds().fillna(0.0) / 3600.0
    # Cap absurd gaps (power outages / file gaps) so a single gap does not
    # add a huge phantom block; iAWE has outages up to ~9h/day.
    dt_hours = dt_hours.clip(lower=0.0, upper=0.1)
    energy_kwh = (df["W"] * dt_hours) / 1000.0
    daily = energy_kwh.groupby(df["datetime"].dt.date).sum()
    idx = pd.to_datetime(pd.Series(daily.index))
    return pd.Series(daily.to_numpy(), index=idx, name="consumption_kwh")


def ingest(directory: Path, files: list[Path] | None = None) -> pd.DataFrame:
    if files:
        candidates = files
    else:
        csvs = sorted(directory.glob("**/*.csv"))
        if not csvs:
            raise FileNotFoundError(
                f"No CSV files found under {directory}. Extract electricity.tar.gz "
                "into data/raw/iawe/ first."
            )
        candidates = [p for p in csvs if _is_main_meter(p.name)]
        if not candidates:
            # Fall back to every file if detection is inconclusive.
            candidates = csvs

    daily_frames = [_integrate_to_daily(_read_power_csv(p)) for p in candidates]
    combined = pd.concat(daily_frames, axis=1).sum(axis=1, skipna=True)
    combined = combined.sort_index()

    out = pd.DataFrame(
        {
            "date": combined.index.strftime("%Y-%m-%d"),
            "consumption_kwh": combined.round(3).to_numpy(),
        }
    )
    return out.reset_index(drop=True)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Convert iAWE raw CSVs to daily kWh.")
    p.add_argument("--dir", default="data/raw/iawe", help="directory with iAWE CSVs")
    p.add_argument("--file", nargs="*", default=None, help="explicit main-meter CSV path(s)")
    p.add_argument("--out", default="data/processed/daily_consumption.csv")
    return p


def main() -> None:
    args = build_parser().parse_args()
    directory = Path(args.dir)
    files = [Path(f) for f in args.file] if args.file else None

    try:
        df = ingest(directory, files)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        print("Hint: run generate_data.py for a synthetic dataset instead.", file=sys.stderr)
        raise SystemExit(1) from exc

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"iAWE ingested: {len(df)} days -> {args.out}")
    print(
        f"  consumption: mean={df['consumption_kwh'].mean():.2f} kWh  "
        f"min={df['consumption_kwh'].min():.2f}  max={df['consumption_kwh'].max():.2f}"
    )


if __name__ == "__main__":
    main()
