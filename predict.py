"""BijliPredict AI - forecast demand and estimate the electricity bill.

Produces a 7-day forecast plus a longer **cost projection** (monthly and
3-month) by rolling the model forward autoregressively.

Examples:
    python predict.py --input data/sample_input.csv --rate 7
    python predict.py --slab "0-100:3.5,100-200:4.5,200-400:6.5,400-100000:7" \
        --fixed 50 --tax-pct 5
    python predict.py --project-days 90

Reads a CSV with at least `window` days of `date,consumption_kwh`, runs the
saved model, forecasts consumption and cost per day, prints the bill, and
writes:
    outputs/predictions.csv       (date, predicted_kwh, estimated_cost_inr)
    outputs/cost_forecast.png     (7-day daily + cumulative cost chart)
    outputs/cost_projection.png   (90-day cumulative cost projection)
    outputs/projection_90d.csv    (per-day projected kWh and cost)
    outputs/cost_summary.json     (bill breakdown + monthly/3-month projection)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import calibration
from features import build_feature_matrix, temperature_for_dates
from inference import feature_info, load_model, make_model_input, resolve_device
from logging_utils import add_log_level_argument, get_logger, setup_logging
from manifest import RunManifest, add_manifest_arguments
from preprocessing import clean_series, is_long_frame, load_consumption, load_scaler
from report import build_excel, build_html
from tariffs import (
    DEFAULT_PRESETS_PATH,
    DEFAULT_RATE,
    Tariff,
    compute_bill,
    get_preset,
    parse_slabs,
)

DEFAULT_INPUT = "data/sample_input.csv"
DEFAULT_ARTIFACTS = "artifacts"
DEFAULT_OUTPUT = "outputs/predictions.csv"
DEFAULT_PLOT = "outputs/cost_forecast.png"
DEFAULT_SUMMARY = "outputs/cost_summary.json"
DEFAULT_PROJECTION_DAYS = 90
DEFAULT_PROJECTION_PLOT = "outputs/cost_projection.png"
DEFAULT_PROJECTION_CSV = "outputs/projection_90d.csv"
DEFAULT_EXCEL = "outputs/bijli_report.xlsx"
DEFAULT_HTML = "outputs/bijli_report.html"
DEFAULT_CALIBRATION = "artifacts/calibration.json"
MONTH_DAYS = 30
QUARTER_DAYS = 90
INTERVAL_LEVELS = (0.80, 0.90)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Forecast demand and estimate the bill.")
    p.add_argument("--input", default=DEFAULT_INPUT, help="recent history CSV")
    p.add_argument("--model-path", default=f"{DEFAULT_ARTIFACTS}/best_model.pt")
    p.add_argument("--scaler-path", default=f"{DEFAULT_ARTIFACTS}/scaler.joblib")
    p.add_argument("--rate", type=float, default=DEFAULT_RATE, help="flat tariff in INR/kWh")
    p.add_argument(
        "--slab",
        default=None,
        help="slab tariff 'LOW-HIGH:RATE,...' (overrides --rate), e.g. "
        "'0-100:3.5,100-200:4.5,200-100000:7'",
    )
    p.add_argument(
        "--preset",
        default=None,
        help="named discom tariff from tariff_presets.toml (e.g. bses, tata_power)",
    )
    p.add_argument(
        "--presets-file",
        default=DEFAULT_PRESETS_PATH,
        help="path to the tariff presets TOML file",
    )
    p.add_argument(
        "--fixed",
        type=float,
        default=None,
        help="fixed charge for the period (INR); overrides a preset's value",
    )
    p.add_argument(
        "--tax-pct",
        type=float,
        default=None,
        help="tax/surcharge %% on energy charge; overrides a preset's value",
    )
    p.add_argument(
        "--project-days",
        type=int,
        default=DEFAULT_PROJECTION_DAYS,
        help="days to project for monthly/3-month cost (rolling forecast)",
    )
    p.add_argument("--out", default=DEFAULT_OUTPUT, help="predictions CSV path")
    p.add_argument("--plot-out", default=DEFAULT_PLOT, help="cost chart PNG path")
    p.add_argument("--summary-out", default=DEFAULT_SUMMARY, help="bill summary JSON path")
    p.add_argument(
        "--projection-plot-out", default=DEFAULT_PROJECTION_PLOT, help="projection PNG path"
    )
    p.add_argument(
        "--projection-csv-out", default=DEFAULT_PROJECTION_CSV, help="projection CSV path"
    )
    p.add_argument(
        "--excel-out",
        default=DEFAULT_EXCEL,
        help="multi-sheet Excel report path (empty to skip)",
    )
    p.add_argument(
        "--html-out",
        default=DEFAULT_HTML,
        help="standalone HTML report path (empty to skip)",
    )
    p.add_argument("--device", default="cpu", choices=["auto", "cpu", "cuda"])
    p.add_argument(
        "--calibration-path",
        default=DEFAULT_CALIBRATION,
        help="conformal calibration JSON written by evaluate.py (empty to skip intervals)",
    )
    p.add_argument(
        "--weather-file",
        default=None,
        help="pre-fetched weather CSV for temperature features (checkpoint-dependent)",
    )
    p.add_argument(
        "--no-validate",
        action="store_true",
        help="skip input-data validation (see validation.py)",
    )
    add_log_level_argument(p)
    add_manifest_arguments(p)
    return p


def build_tariff(args: argparse.Namespace) -> Tariff:
    """Resolve a Tariff from --preset / --slab / --rate, then apply overrides.

    Precedence: an explicit --slab beats a preset's slabs; --fixed / --tax-pct
    override whatever the preset or flat default supplied when given.
    """
    if getattr(args, "preset", None):
        tariff = get_preset(args.preset, getattr(args, "presets_file", DEFAULT_PRESETS_PATH))
        if args.slab:
            tariff.slabs = parse_slabs(args.slab)
            tariff.rate = 0.0
            tariff.label = f"slab ({args.preset})"
    elif args.slab:
        tariff = Tariff(slabs=parse_slabs(args.slab), label="slab")
    else:
        tariff = Tariff(rate=args.rate, label="flat")

    if getattr(args, "fixed", None) is not None:
        tariff.fixed = args.fixed
    if getattr(args, "tax_pct", None) is not None:
        tariff.tax_pct = args.tax_pct
    return tariff


def save_cost_plot(
    dates, forecast, daily_costs, total_cost, path: Path, band: tuple | None = None
) -> None:
    fig, ax1 = plt.subplots(figsize=(11, 5))
    days = [f"Day {i}\n{d.date()}" for i, d in enumerate(dates, start=1)]

    bars = ax1.bar(days, daily_costs, color="tab:orange", alpha=0.85, label="Daily cost")
    ax1.set_ylabel("Cost (₹)", color="tab:orange")
    ax1.tick_params(axis="y", labelcolor="tab:orange")
    ax1.set_title("Forecast electricity cost")
    for b, c in zip(bars, daily_costs, strict=True):
        ax1.text(
            b.get_x() + b.get_width() / 2,
            b.get_height(),
            f"₹{c:.1f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax2 = ax1.twinx()
    cumulative = pd.Series(daily_costs).cumsum().tolist()
    if band is not None:
        cum_lower, cum_upper = band
        ax2.fill_between(
            range(len(dates)), cum_lower, cum_upper, color="tab:blue", alpha=0.15, label="80% band"
        )
    ax2.plot(days, cumulative, color="tab:blue", marker="o", label="Cumulative cost")
    ax2.set_ylabel("Cumulative cost (₹)", color="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:blue")

    ax1.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def rolling_forecast(
    model,
    scaler,
    history_kwh: np.ndarray,
    input_len: int,
    horizon: int,
    total_days: int,
    device: torch.device,
    *,
    feature_scaler=None,
    history_features: np.ndarray | None = None,
    future_features: np.ndarray | None = None,
) -> list[float]:
    """Autoregressive multi-step forecast.

    Repeatedly runs the fixed-horizon model and feeds predictions back as input
    until `total_days` daily values are produced. The first `horizon` values are
    identical to a single direct forecast from the same window.

    When the checkpoint was trained on exogenous features, pass the **raw**
    `history_features` (``input_len`` rows, aligned to the input window) and
    `future_features` (``total_days`` rows for the days being forecast); they are
    scaled with `feature_scaler` and advanced in lockstep as the window slides.
    """
    values = [float(v) for v in history_kwh]
    preds: list[float] = []

    feat_all = None
    if feature_scaler is not None and history_features is not None and future_features is not None:
        feat_all = np.vstack(
            [
                np.asarray(history_features, dtype=float),
                np.asarray(future_features, dtype=float),
            ]
        )

    p = 0  # number of future daily values appended so far
    while len(preds) < total_days:
        window = np.asarray(values[-input_len:], dtype=float)
        fwin = feat_all[p : p + input_len] if feat_all is not None else None
        x = make_model_input(scaler, feature_scaler, window, fwin)
        with torch.no_grad():
            p_scaled = model(torch.from_numpy(x).to(device)).cpu().numpy()
        step = scaler.inverse_transform(p_scaled.reshape(-1, 1)).reshape(-1)
        step = [max(0.0, float(v)) for v in step]
        preds.extend(step)
        values.extend(step)
        p += len(step)
    return preds[:total_days]


def save_projection_plot(
    dates, daily_costs, path: Path, marks=(MONTH_DAYS, QUARTER_DAYS), band: tuple | None = None
) -> None:
    cumulative = np.cumsum(daily_costs)
    fig, ax1 = plt.subplots(figsize=(12, 5))
    x = np.arange(len(daily_costs))
    ax1.bar(x, daily_costs, color="tab:orange", alpha=0.6, label="Daily cost")
    ax1.set_xlabel("Days ahead")
    ax1.set_ylabel("Daily cost (₹)", color="tab:orange")
    ax1.tick_params(axis="y", labelcolor="tab:orange")
    ax1.set_title("Cost projection (rolling model forecast)")

    ax2 = ax1.twinx()
    if band is not None:
        cum_lower, cum_upper = band
        ax2.fill_between(x, cum_lower, cum_upper, color="tab:blue", alpha=0.15, label="80% band")
    ax2.plot(x, cumulative, color="tab:blue", linewidth=1.8, label="Cumulative cost")
    ax2.set_ylabel("Cumulative cost (₹)", color="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:blue")

    for m in marks:
        if 0 < m <= len(daily_costs):
            ax2.axvline(m - 1, color="grey", linestyle="--", alpha=0.7)
            ax2.annotate(
                f"{m}d: ₹{cumulative[m - 1]:.0f}",
                xy=(m - 1, cumulative[m - 1]),
                xytext=(m - 1, cumulative[m - 1]),
                fontsize=9,
                color="black",
                ha="center",
                va="bottom",
            )

    ax1.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _period_summary(bill: dict) -> dict:
    return {
        "total_kwh": round(bill["total_kwh"], 3),
        "energy_charge_inr": round(bill["energy_charge"], 2),
        "tax_inr": round(bill["tax"], 2),
        "fixed_charge_inr": round(bill["fixed"], 2),
        "estimated_bill_inr": round(bill["total"], 2),
    }


def _cumulative_bill(kwh, tariff) -> np.ndarray:
    """Cumulative bill (INR) after each day, applying the tariff to the prefix."""
    kwh = np.asarray(kwh, dtype=float)
    return np.array([compute_bill(kwh[: i + 1], tariff)["total"] for i in range(len(kwh))])


def _intervals(pred, calib, steps) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Lower/upper kWh bounds per level for a 1-D prediction sequence."""
    out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for level in INTERVAL_LEVELS:
        lower, upper = calibration.apply(pred, calib, level, steps=steps)
        out[int(round(level * 100))] = (np.asarray(lower), np.asarray(upper))
    return out


def main(args: argparse.Namespace | None = None) -> None:
    if args is None:
        args = build_parser().parse_args()
    setup_logging(args.log_level)
    log = get_logger("predict")
    manifest = RunManifest(
        "predict",
        args=vars(args),
        manifest_dir=args.manifest_dir,
        enabled=not args.no_manifest,
    )
    manifest.add_input("input", args.input)
    manifest.add_input("model", args.model_path)
    manifest.add_input("scaler", args.scaler_path)
    try:
        _run(args, log, manifest)
    except Exception as exc:
        manifest.set("error", f"{type(exc).__name__}: {exc}")
        manifest.write(status="error", error=f"{type(exc).__name__}: {exc}")
        raise
    path = manifest.write()
    log.info("event=run_end command=predict status=ok manifest=%s", path)


def _run(args: argparse.Namespace, log, manifest: RunManifest) -> None:
    start = time.perf_counter()
    device = resolve_device(args.device)
    tariff = build_tariff(args)
    log.info(
        "event=run_start command=predict device=%s tariff=%s project_days=%d",
        device,
        tariff.describe(),
        args.project_days,
    )

    bundle = load_model(args.model_path, device)
    model, ckpt = bundle["model"], bundle["ckpt"]
    input_len, horizon = ckpt["input_len"], ckpt["horizon"]
    scaler = load_scaler(args.scaler_path)

    raw_history = load_consumption(args.input, validate=not args.no_validate)
    if is_long_frame(raw_history):
        raise SystemExit(
            f"{args.input} is multi-household long format; predict expects a single "
            "household (date,consumption_kwh). Extract one household's history first."
        )
    history = clean_series(raw_history)
    if len(history) < input_len:
        raise SystemExit(
            f"Need at least {input_len} days of history, got {len(history)} from {args.input}."
        )
    window = history.tail(input_len)
    last_date = window.index[-1]
    project_days = max(args.project_days, horizon)

    # --- Exogenous features (only if the checkpoint was trained with them) ---
    feature_config, feature_columns, feature_scaler_path = feature_info(ckpt)
    feature_scaler = None
    history_features = future_features = None
    if feature_columns:
        if feature_scaler_path is None:
            raise SystemExit("Checkpoint expects features but records no feature scaler path.")
        feature_scaler = load_scaler(feature_scaler_path)
        window_dates = window.index
        future_dates = pd.date_range(
            last_date + pd.Timedelta(days=1), periods=project_days, freq="D"
        )
        feat_start = window_dates[0] - pd.Timedelta(days=feature_config.buffer_days)
        full_dates = pd.date_range(feat_start, future_dates[-1], freq="D")
        temperature = None
        if feature_config.temperature:
            temperature = temperature_for_dates(
                feature_config, full_dates, weather_file=args.weather_file
            )
        feature_matrix, columns = build_feature_matrix(
            full_dates, temperature=temperature, config=feature_config
        )
        if columns != feature_columns:
            raise SystemExit(
                f"Feature mismatch: model expects {feature_columns} but built {columns}."
            )
        history_features = feature_matrix.loc[window_dates].to_numpy()
        future_features = feature_matrix.loc[future_dates].to_numpy()

    # One rolling forecast covers both the 7-day output and the projection;
    # its first `horizon` values are the direct 7-day forecast.
    projection = rolling_forecast(
        model,
        scaler,
        window["consumption_kwh"].to_numpy(),
        input_len,
        horizon,
        project_days,
        device,
        feature_scaler=feature_scaler,
        history_features=history_features,
        future_features=future_features,
    )
    forecast = projection[:horizon]
    dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon, freq="D")
    proj_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=project_days, freq="D")

    bill = compute_bill(forecast, tariff)
    daily_costs = bill["daily_costs"]

    proj_bill = compute_bill(projection, tariff)
    monthly = compute_bill(projection[:MONTH_DAYS], tariff)
    quarterly = compute_bill(projection[:QUARTER_DAYS], tariff)

    # --- Conformal intervals (optional; requires artifacts/calibration.json) ---
    calib = None
    if args.calibration_path and Path(args.calibration_path).exists():
        calib = calibration.load(args.calibration_path)
    forecast_iv = _intervals(forecast, calib, horizon) if calib else {}
    projection_iv = _intervals(projection, calib, project_days) if calib else {}

    # --- Report ---
    print()
    print("BIJLIPREDICT AI")
    print("-----------------------------")
    print()
    print(f"Forecast Period: {horizon} Days")
    print(f"Tariff: {tariff.describe()}")
    print()
    for i, (d, kwh, cost) in enumerate(zip(dates, forecast, daily_costs, strict=True), start=1):
        print(f"Day {i}: {kwh:.1f} kWh   ₹{cost:6.2f}   ({d.date()})")
    print()
    print("-----------------------------")
    print(f"Total Forecast: {bill['total_kwh']:.1f} kWh")
    print()
    print(f"Energy Charge      : ₹{bill['energy_charge']:.2f}")
    if tariff.tax_pct:
        print(f"Tax ({tariff.tax_pct:g}%)       : ₹{bill['tax']:.2f}")
    if tariff.fixed:
        print(f"Fixed Charge       : ₹{bill['fixed']:.2f}")
    print(f"Estimated Bill     : ₹{bill['total']:.2f}")
    if forecast_iv:
        for pct in sorted(forecast_iv):
            lo_bill = compute_bill(forecast_iv[pct][0], tariff)["total"]
            hi_bill = compute_bill(forecast_iv[pct][1], tariff)["total"]
            print(f"{pct}% bill interval  : ₹{lo_bill:.2f} – ₹{hi_bill:.2f}")
    print()

    print("Long-term cost projection (rolling forecast)")
    print(
        f"  Monthly   ({min(MONTH_DAYS, len(projection)):>2} days): "
        f"{monthly['total_kwh']:8.1f} kWh   Estimated Bill: ₹{monthly['total']:.2f}"
    )
    print(
        f"  3-Month   ({min(QUARTER_DAYS, len(projection)):>2} days): "
        f"{quarterly['total_kwh']:8.1f} kWh   Estimated Bill: ₹{quarterly['total']:.2f}"
    )
    print("  NOTE: long-horizon rollout accumulates error; treat as an estimate.")
    print()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    forecast_cols: dict = {
        "date": dates.strftime("%Y-%m-%d"),
        "predicted_kwh": [round(v, 3) for v in forecast],
    }
    for pct in sorted(forecast_iv):
        forecast_cols[f"lower_{pct}"] = [round(v, 3) for v in forecast_iv[pct][0]]
        forecast_cols[f"upper_{pct}"] = [round(v, 3) for v in forecast_iv[pct][1]]
    forecast_cols["estimated_cost_inr"] = [round(c, 2) for c in daily_costs]
    pd.DataFrame(forecast_cols).to_csv(out, index=False)

    forecast_band = None
    if 80 in forecast_iv:
        forecast_band = (
            _cumulative_bill(forecast_iv[80][0], tariff),
            _cumulative_bill(forecast_iv[80][1], tariff),
        )
    save_cost_plot(
        dates, forecast, daily_costs, bill["total"], Path(args.plot_out), band=forecast_band
    )

    proj_out = Path(args.projection_csv_out)
    proj_out.parent.mkdir(parents=True, exist_ok=True)
    cumulative = np.cumsum(proj_bill["daily_costs"])
    proj_cols: dict = {
        "date": proj_dates.strftime("%Y-%m-%d"),
        "projected_kwh": [round(v, 3) for v in projection],
    }
    for pct in sorted(projection_iv):
        proj_cols[f"lower_{pct}"] = [round(v, 3) for v in projection_iv[pct][0]]
        proj_cols[f"upper_{pct}"] = [round(v, 3) for v in projection_iv[pct][1]]
    proj_cols["daily_cost_inr"] = [round(c, 2) for c in proj_bill["daily_costs"]]
    proj_cols["cumulative_cost_inr"] = [round(c, 2) for c in cumulative]
    pd.DataFrame(proj_cols).to_csv(proj_out, index=False)

    projection_band = None
    if 80 in projection_iv:
        projection_band = (
            _cumulative_bill(projection_iv[80][0], tariff),
            _cumulative_bill(projection_iv[80][1], tariff),
        )
    save_projection_plot(
        proj_dates, proj_bill["daily_costs"], Path(args.projection_plot_out), band=projection_band
    )

    summary = {
        "forecast_days": horizon,
        "tariff": bill["tariff"],
        "total_kwh": round(bill["total_kwh"], 3),
        "energy_charge_inr": round(bill["energy_charge"], 2),
        "tax_inr": round(bill["tax"], 2),
        "fixed_charge_inr": round(bill["fixed"], 2),
        "estimated_bill_inr": round(bill["total"], 2),
        "daily": [
            {
                "date": d.strftime("%Y-%m-%d"),
                "kwh": round(k, 3),
                "cost_inr": round(c, 2),
            }
            for d, k, c in zip(dates, forecast, daily_costs, strict=True)
        ],
        "projection": {
            "project_days": project_days,
            "monthly_30d": _period_summary(monthly),
            "three_month_90d": _period_summary(quarterly),
        },
        "interval": {
            "method": calib.get("method") if calib else None,
            "levels": {
                str(pct): {
                    "level": pct / 100,
                    "total_kwh_lower": round(float(np.sum(forecast_iv[pct][0])), 3),
                    "total_kwh_upper": round(float(np.sum(forecast_iv[pct][1])), 3),
                    "estimated_bill_lower_inr": round(
                        compute_bill(forecast_iv[pct][0], tariff)["total"], 2
                    ),
                    "estimated_bill_upper_inr": round(
                        compute_bill(forecast_iv[pct][1], tariff)["total"], 2
                    ),
                }
                for pct in sorted(forecast_iv)
            },
        },
    }
    Path(args.summary_out).write_text(json.dumps(summary, indent=2))

    print("Prediction saved:")
    print(str(out))
    print("Cost chart saved:")
    print(str(args.plot_out))
    print("Projection chart saved:")
    print(str(args.projection_plot_out))
    print("Projection CSV saved:")
    print(str(proj_out))
    print("Bill summary saved:")
    print(str(args.summary_out))

    if args.excel_out:
        excel_path = build_excel(out.parent, args.excel_out)
        print("Excel report saved:")
        print(str(excel_path))

    if args.html_out:
        html_path = build_html(out.parent, args.html_out)
        print("HTML report saved:")
        print(str(html_path))

    manifest.add_output("predictions", out)
    manifest.add_output("cost_plot", args.plot_out)
    manifest.add_output("projection_plot", args.projection_plot_out)
    manifest.add_output("projection_csv", proj_out)
    manifest.add_output("summary", args.summary_out)
    if args.excel_out:
        manifest.add_output("excel_report", args.excel_out)
    if args.html_out:
        manifest.add_output("html_report", args.html_out)
    manifest.set("device", str(device))
    manifest.set("tariff", bill["tariff"])
    manifest.set("total_kwh", round(bill["total_kwh"], 3))
    manifest.set("estimated_bill_inr", round(bill["total"], 2))
    manifest.record_duration("predict", time.perf_counter() - start)
    log.info(
        "event=predict_complete total_kwh=%.3f estimated_bill_inr=%.2f",
        bill["total_kwh"],
        bill["total"],
    )


if __name__ == "__main__":
    main()
