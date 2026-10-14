"""BijliPredict AI - electricity tariff / bill estimation.

Two modes are supported:

  * flat  : a single INR/kWh rate applied to every unit (the MVP default).
  * slab  : telescopic (progressive) slabs, as used by most Indian discoms.
            The first N units are charged at the first rate, the next at the
            second rate, and so on.

Optional additions on top of the energy charge:
  * fixed      : a one-time fixed charge for the billing period (INR).
  * tax_pct    : a percentage surcharge applied to the energy charge.

Bill = energy_charge + tax + fixed
"""

from __future__ import annotations

import dataclasses
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_RATE = 7.0
DEFAULT_PRESETS_PATH = "tariff_presets.toml"


@dataclass
class Tariff:
    rate: float = DEFAULT_RATE
    slabs: list[tuple[float, float]] | None = None  # [(upper_limit_kwh, rate), ...]
    fixed: float = 0.0
    tax_pct: float = 0.0
    label: str = field(default="flat")

    def energy_cost(self, kwh: float) -> float:
        """Cost of a single day's consumption under this tariff (energy only)."""
        kwh = max(0.0, float(kwh))
        if self.slabs:
            cost = 0.0
            lower = 0.0
            for upper, rate in self.slabs:
                if kwh <= lower:
                    break
                units = min(kwh, upper) - lower
                if units > 0:
                    cost += units * rate
                lower = upper
            return cost
        return kwh * self.rate

    def describe(self) -> str:
        if self.slabs:
            parts = []
            lower = 0.0
            for upper, rate in self.slabs:
                parts.append(f"{lower:g}-{upper:g}:₹{rate:g}")
                lower = upper
            return "slab [" + ", ".join(parts) + "]"
        return f"₹{self.rate:g}/kWh (flat)"


def parse_slabs(spec: str) -> list[tuple[float, float]]:
    """Parse '0-100:3.5,100-200:4.5,200-100000:7' into [(upper, rate), ...].

    Slabs must start at 0 and be contiguous (each lower bound equals the previous
    upper bound), so the telescopic charge is unambiguous. The returned list only
    carries the upper bound; the lower bound is implied by the previous entry.
    """
    parsed: list[tuple[float, float, float]] = []  # (lower, upper, rate)
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            rng, rate_str = token.split(":")
            low_str, high_str = rng.split("-")
            lower, upper, rate = float(low_str), float(high_str), float(rate_str)
        except ValueError as exc:
            raise ValueError(
                f"Bad slab '{token}'. Expected format 'LOW-HIGH:RATE', e.g. '0-100:3.5'."
            ) from exc
        if not lower < upper:
            raise ValueError(f"Bad slab '{token}': lower bound must be < upper bound.")
        if rate < 0:
            raise ValueError(f"Bad slab '{token}': rate must be non-negative.")
        parsed.append((lower, upper, rate))

    if not parsed:
        raise ValueError("No valid slabs parsed from --slab.")

    parsed.sort(key=lambda s: s[0])
    if abs(parsed[0][0]) > 1e-9:
        raise ValueError(
            f"Slab tariff must start at 0, but the first slab starts at {parsed[0][0]:g}."
        )
    for (_, prev_upper, _), (lower, _, _) in zip(parsed, parsed[1:], strict=False):
        if abs(lower - prev_upper) > 1e-6:
            raise ValueError(
                f"Slabs must be contiguous: gap/overlap between {prev_upper:g} and {lower:g}."
            )
    return [(upper, rate) for _, upper, rate in parsed]


def _tariff_from_dict(name: str, raw: dict) -> Tariff:
    """Build a Tariff from a preset table (see tariff_presets.toml)."""
    slabs = None
    if "slabs" in raw:
        spec = raw["slabs"]
        slabs = parse_slabs(spec if isinstance(spec, str) else ",".join(spec))
    return Tariff(
        rate=float(raw.get("rate", DEFAULT_RATE)),
        slabs=slabs,
        fixed=float(raw.get("fixed", 0.0)),
        tax_pct=float(raw.get("tax_pct", 0.0)),
        label=str(raw.get("label", name)),
    )


def load_presets(path: str | Path = DEFAULT_PRESETS_PATH) -> dict[str, Tariff]:
    """Load every named tariff preset from a TOML file (keyed by name)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Tariff presets file not found: {path}. Expected names like 'bses'."
        )
    data = tomllib.loads(path.read_text())
    return {name: _tariff_from_dict(name, raw) for name, raw in data.items()}


def get_preset(name: str, path: str | Path = DEFAULT_PRESETS_PATH) -> Tariff:
    """Return an independent copy of the named preset."""
    presets = load_presets(path)
    if name not in presets:
        available = ", ".join(sorted(presets))
        raise KeyError(f"Unknown tariff preset '{name}'. Available: {available}.")
    return dataclasses.replace(presets[name])


def compute_bill(forecast_kwh: list[float], tariff: Tariff) -> dict:
    """Return per-day costs and the bill breakdown for a period.

    Slabs are telescopic over the *period total*, so each day is priced at the
    margin: daily_cost[i] = cost(cumulative_through_i) - cost(through_{i-1}).
    The per-day costs therefore sum exactly to the period energy charge. For a
    flat rate this reduces to kwh * rate per day.
    """
    values = [max(0.0, float(v)) for v in forecast_kwh]
    daily: list[float] = []
    cumulative = 0.0
    prev_cumulative = 0.0
    for v in values:
        cumulative += v
        daily.append(tariff.energy_cost(cumulative) - tariff.energy_cost(prev_cumulative))
        prev_cumulative = cumulative

    energy = tariff.energy_cost(cumulative)
    tax = energy * tariff.tax_pct / 100.0
    total = energy + tax + tariff.fixed
    return {
        "daily_costs": daily,
        "energy_charge": energy,
        "tax": tax,
        "fixed": tariff.fixed,
        "total": total,
        "total_kwh": cumulative,
        "tariff": tariff.describe(),
    }
