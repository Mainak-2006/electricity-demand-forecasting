"""BijliPredict AI - evaluate the trained model on the held-out test period.

Reports MAE, RMSE and WAPE (plus MAPE) and compares against two naive
baselines:
    * seasonal-naive (period 7): repeat the last observed week
    * persistence: repeat the last observed day

With the optional `stats` extra installed, SARIMAX and ETS block forecasts are
reported alongside. Passing ``--cv rolling`` additionally evaluates the frozen
model over expanding-window rolling-origin folds.

Outputs:
    outputs/evaluation.json      (single-split metrics + coverage)
    outputs/evaluation_cv.json   (per-fold + mean/std metrics, with --cv rolling)
    outputs/forecast.png
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import baselines
import calibration
import cv
from features import temperature_for_dates
from inference import feature_info, load_model, predict, resolve_device
from logging_utils import add_log_level_argument, get_logger, setup_logging
from manifest import RunManifest, add_manifest_arguments
from preprocessing import load_and_prepare, load_consumption, load_scaler, prepare_full_series

DEFAULT_DATA = "data/processed/daily_consumption.csv"
DEFAULT_ARTIFACTS = "artifacts"
DEFAULT_OUTPUTS = "outputs"
DEFAULT_CALIBRATION = f"{DEFAULT_ARTIFACTS}/calibration.json"

EPS = 1e-8


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    err = y_true - y_pred
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    denom = float(np.sum(np.abs(y_true)))
    wape = float(np.sum(np.abs(err)) / (denom + EPS) * 100.0)
    mape = float(np.mean(np.abs(err) / np.maximum(np.abs(y_true), EPS)) * 100.0)
    return {"mae": mae, "rmse": rmse, "wape": wape, "mape": mape}


def window_metrics(model, X, y, scaler, device) -> tuple[np.ndarray, np.ndarray, dict]:
    """Predict a set of scaled windows and score them in kWh.

    Returns ``(y_true, y_pred, metrics)``; predictions are clipped at zero
    because household consumption cannot be negative.
    """
    pred_scaled = predict(model, X, device)
    y_true = scaler.inverse_transform(y.reshape(-1, 1)).reshape(y.shape)
    y_pred = np.clip(
        scaler.inverse_transform(pred_scaled.reshape(-1, 1)).reshape(pred_scaled.shape), 0.0, None
    )
    return y_true, y_pred, compute_metrics(y_true, y_pred)


def stats_baselines(
    train_values, test_values, y_true, input_len: int, horizon: int
) -> dict[str, dict[str, float]]:
    """SARIMAX / ETS rolling forecasts scored on the model's test windows.

    Both baselines forecast ``horizon`` days from each test window's origin, so
    their metrics are directly comparable to the model's. Returns ``{}`` when
    the optional `stats` extra is unavailable; individual entries carry an
    ``"error"`` key if a model fails to fit.
    """
    if not baselines.statsmodels_available():
        return {}
    out: dict[str, dict[str, float]] = {}
    for name in ("sarimax", "ets"):
        try:
            pred = baselines.rolling_block(
                np.asarray(train_values, dtype=float).reshape(-1),
                np.asarray(test_values, dtype=float).reshape(-1),
                input_len,
                horizon,
                name,
            )
        except Exception as exc:  # pragma: no cover - depends on optional dep
            out[name] = {"error": str(exc)}
            continue
        out[name] = compute_metrics(y_true, pred)
    return out


def rolling_cv(
    args,
    model,
    scaler,
    device,
    input_len: int,
    horizon: int,
    feature_config,
    feature_scaler,
    prepared: dict,
) -> dict:
    """Evaluate the frozen model over rolling-origin folds.

    Folds are anchored at the train/val boundary so every fold is fully
    out-of-sample; the loaded checkpoint is reused (no per-fold retraining).
    Windows whose whole target falls inside a fold's test range are scored.
    """
    full = prepare_full_series(
        args.data,
        input_len=input_len,
        horizon=horizon,
        scaler=scaler,
        feature_config=feature_config,
        feature_scaler=feature_scaler,
        weather_file=args.weather_file,
        validate=not getattr(args, "no_validate", False),
    )
    n = len(full["series"])
    train_end = len(prepared["frames"]["train"]) + len(prepared["frames"]["val"])
    train_end = min(max(train_end, 1), n - 1)

    folds = cv.rolling_origin_splits(
        n,
        n_folds=args.cv_folds,
        test_size=args.cv_test_size,
        min_train_size=train_end,
    )
    X_all, y_all = full["X"], full["y"]
    rows: list[dict] = []
    for fold in folds:
        lo = max(0, fold.test_start - input_len)
        hi = min(len(X_all), fold.test_end - input_len - horizon + 1)
        if hi <= lo:
            continue
        _, _, metrics = window_metrics(model, X_all[lo:hi], y_all[lo:hi], scaler, device)
        rows.append(
            {
                "fold": fold.index,
                "train_end": fold.train_end,
                "test_start": fold.test_start,
                "test_end": fold.test_end,
                "test_windows": int(hi - lo),
                "metrics": metrics,
            }
        )
    if not rows:
        raise SystemExit("Rolling CV produced no evaluable folds - dataset too short.")

    keys = ("mae", "rmse", "wape", "mape")
    return {
        "method": "rolling-origin",
        "n_folds": len(rows),
        "min_train_size": train_end,
        "series_length": n,
        "wape_mean": float(np.mean([r["metrics"]["wape"] for r in rows])),
        "wape_std": float(np.std([r["metrics"]["wape"] for r in rows])),
        "metric_means": {k: float(np.mean([r["metrics"][k] for r in rows])) for k in keys},
        "metric_stds": {k: float(np.std([r["metrics"][k] for r in rows])) for k in keys},
        "folds": rows,
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate the trained forecaster.")
    p.add_argument("--data", default=DEFAULT_DATA)
    p.add_argument("--model-path", default=f"{DEFAULT_ARTIFACTS}/best_model.pt")
    p.add_argument("--scaler-path", default=f"{DEFAULT_ARTIFACTS}/scaler.joblib")
    p.add_argument("--out-dir", default=DEFAULT_OUTPUTS)
    p.add_argument("--device", default="cpu", choices=["auto", "cpu", "cuda"])
    p.add_argument("--splits", type=float, nargs=3, default=(0.70, 0.15, 0.15))
    p.add_argument(
        "--calibration-out",
        default=DEFAULT_CALIBRATION,
        help="where to write the fitted conformal calibration JSON",
    )
    p.add_argument(
        "--calibration-scale",
        default=calibration.DEFAULT_SCALE,
        choices=["absolute", "power"],
        help="how interval width scales with the forecast level (see calibration.py)",
    )
    p.add_argument(
        "--calibration-gamma",
        type=float,
        default=calibration.DEFAULT_GAMMA,
        help="exponent for --calibration-scale power (sigma = yhat ** gamma)",
    )
    p.add_argument(
        "--weather-file",
        default=None,
        help="pre-fetched weather CSV to use instead of the Open-Meteo cache",
    )
    p.add_argument(
        "--cv",
        default="none",
        choices=["none", "rolling"],
        help="cross-validation scheme (rolling = expanding-window origin CV)",
    )
    p.add_argument("--cv-folds", type=int, default=5, help="number of rolling-origin CV folds")
    p.add_argument(
        "--cv-test-size",
        type=int,
        default=None,
        help="test-window length (days) per CV fold (default: even split)",
    )
    p.add_argument(
        "--cv-out",
        default=None,
        help="where to write evaluation_cv.json (default: <out-dir>/evaluation_cv.json)",
    )
    p.add_argument(
        "--no-stats-baselines",
        action="store_true",
        help="skip the SARIMAX/ETS baselines (also skipped if the stats extra is missing)",
    )
    p.add_argument(
        "--no-validate",
        action="store_true",
        help="skip input-data validation (see validation.py)",
    )
    add_log_level_argument(p)
    add_manifest_arguments(p)
    return p


def main(args: argparse.Namespace | None = None) -> None:
    if args is None:
        args = build_parser().parse_args()
    setup_logging(args.log_level)
    log = get_logger("evaluate")
    manifest = RunManifest(
        "evaluate",
        args=vars(args),
        manifest_dir=args.manifest_dir,
        enabled=not args.no_manifest,
    )
    manifest.add_input("data", args.data)
    manifest.add_input("model", args.model_path)
    manifest.add_input("scaler", args.scaler_path)
    try:
        _run(args, log, manifest)
    except Exception as exc:
        manifest.set("error", f"{type(exc).__name__}: {exc}")
        manifest.write(status="error", error=f"{type(exc).__name__}: {exc}")
        raise
    path = manifest.write()
    log.info("event=run_end command=evaluate status=ok manifest=%s", path)


def _run(args: argparse.Namespace, log, manifest: RunManifest) -> None:
    start = time.perf_counter()
    device = resolve_device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("event=run_start command=evaluate device=%s cv=%s", device, args.cv)

    bundle = load_model(args.model_path, device)
    model, ckpt = bundle["model"], bundle["ckpt"]
    input_len, horizon = ckpt["input_len"], ckpt["horizon"]
    scaler = load_scaler(args.scaler_path)

    feature_config, feature_columns, feature_scaler_path = feature_info(ckpt)
    feature_scaler = None
    temperature = None
    if feature_columns:
        if feature_scaler_path is None:
            raise SystemExit("Checkpoint expects features but records no feature scaler path.")
        feature_scaler = load_scaler(feature_scaler_path)
        if feature_config.temperature:
            raw = load_consumption(args.data, validate=not args.no_validate)
            temperature = temperature_for_dates(
                feature_config, raw.index, weather_file=args.weather_file
            )

    prepared = load_and_prepare(
        args.data,
        input_len=input_len,
        horizon=horizon,
        splits=tuple(args.splits),
        save_scaler=False,
        scaler=scaler,
        feature_config=feature_config,
        temperature=temperature,
        feature_scaler=feature_scaler,
        validate=not args.no_validate,
    )
    X_test = prepared["datasets"]["test"]["X"]
    y_test = prepared["datasets"]["test"]["y"]
    if len(X_test) == 0:
        raise SystemExit("No test windows available - dataset too short for the split.")
    multi = bool(prepared.get("multi_household"))
    if multi and args.cv == "rolling":
        raise SystemExit(
            "Rolling-origin CV is defined on a single time series; evaluate "
            "long-format (multi-household) data without --cv, per household."
        )

    # --- Predictions (scaled -> kWh) ---
    pred_scaled = predict(model, X_test, device)

    def inv(a):
        return scaler.inverse_transform(a.reshape(-1, 1)).reshape(a.shape)

    y_true = inv(y_test)
    y_pred = np.clip(inv(pred_scaled), 0.0, None)  # consumption is non-negative

    # --- Conformal intervals: per-horizon widths calibrated on the val fold ---
    X_val = prepared["datasets"]["val"]["X"]
    y_val = prepared["datasets"]["val"]["y"]
    calib = None
    coverage = None
    if len(X_val) > 0:
        y_val_true = inv(y_val)
        y_val_pred = np.clip(inv(predict(model, X_val, device)), 0.0, None)
        calib = calibration.fit(
            y_val_true,
            y_val_pred,
            levels=calibration.DEFAULT_LEVELS,
            scale=args.calibration_scale,
            gamma=args.calibration_gamma,
        )
        calibration.save(calib, args.calibration_out)
        coverage = calibration.coverage_report(calib, y_true, y_pred)

    # --- Baselines (from the raw consumption channel of the input window) ---
    X_raw = inv(X_test[:, :, 0])
    seasonal_naive = X_raw[:, -horizon:]  # repeat last week
    persistence = np.repeat(X_raw[:, -1:], horizon, axis=1)  # repeat last day

    model_metrics = compute_metrics(y_true, y_pred)
    baseline_metrics = {
        "seasonal_naive_p7": compute_metrics(y_true, seasonal_naive),
        "persistence": compute_metrics(y_true, persistence),
    }
    per_horizon_mae = [float(np.mean(np.abs(y_true[:, k] - y_pred[:, k]))) for k in range(horizon)]

    # --- Statistical baselines (optional; rolling SARIMAX / ETS forecasts) ---
    # Single-series only: pooled windows across households have no one timeline
    # to run a rolling statistical forecast over.
    stats = {}
    if not args.no_stats_baselines and not multi:
        train_values = np.concatenate(
            [
                prepared["frames"]["train"]["consumption_kwh"].to_numpy(),
                prepared["frames"]["val"]["consumption_kwh"].to_numpy(),
            ]
        )
        stats = stats_baselines(
            train_values,
            prepared["frames"]["test"]["consumption_kwh"].to_numpy(),
            y_true,
            input_len,
            horizon,
        )

    # --- Reconstruct a per-date series by averaging overlapping forecasts ---
    plot_path = out_dir / "forecast.png"
    if multi:
        # Pooled windows span several homes, so there is no single date axis to
        # overlay; report per-horizon skill across the whole pool instead.
        naive_horizon_mae = [
            float(np.mean(np.abs(y_true[:, k] - seasonal_naive[:, k]))) for k in range(horizon)
        ]
        fig, ax = plt.subplots(figsize=(10, 4.5))
        xs = np.arange(1, horizon + 1)
        width = 0.38
        ax.bar(xs - width / 2, per_horizon_mae, width, label=ckpt["model_type"].upper(), color="tab:red")
        ax.bar(xs + width / 2, naive_horizon_mae, width, label="Seasonal-naive", color="tab:blue")
        ax.set_title(f"Pooled multi-household test MAE by horizon (1..{horizon} days)")
        ax.set_xlabel("Horizon (days ahead)")
        ax.set_ylabel("MAE (kWh)")
        ax.set_xticks(xs)
        ax.legend()
        ax.grid(alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(plot_path, dpi=120)
        plt.close(fig)
    else:
        dates = prepared["frames"]["test"].index
        pred_sum: dict = defaultdict(float)
        pred_cnt: dict = defaultdict(int)
        actual_by_date: dict = {}
        for i in range(len(X_test)):
            for k in range(horizon):
                idx = i + input_len + k
                if idx >= len(dates):
                    continue
                d = dates[idx]
                pred_sum[d] += float(y_pred[i, k])
                pred_cnt[d] += 1
                actual_by_date[d] = float(y_true[i, k])
        plot_dates = sorted(actual_by_date)
        plot_actual = np.array([actual_by_date[d] for d in plot_dates])
        plot_pred = np.array([pred_sum[d] / pred_cnt[d] for d in plot_dates])

        # --- Plot ---
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        axes[0].plot(plot_dates, plot_actual, label="Actual", color="black", linewidth=1.2)
        axes[0].plot(
            plot_dates, plot_pred, label="Predicted (avg)", color="tab:red", linewidth=1.2
        )
        axes[0].set_title("Test period - actual vs predicted daily consumption")
        axes[0].set_ylabel("kWh / day")
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        last = -1
        d0 = input_len
        if len(X_test) > 0:
            td = [dates[i + d0] for i in range(horizon) if i + d0 < len(dates)]
            axes[1].plot(td, y_true[last, : len(td)], "o-", label="Actual", color="black")
            axes[1].plot(
                td,
                y_pred[last, : len(td)],
                "s--",
                label=f"{ckpt['model_type'].upper()} forecast",
                color="tab:red",
            )
            axes[1].plot(
                td,
                seasonal_naive[last, : len(td)],
                "^:",
                label="Seasonal-naive",
                color="tab:blue",
            )
            axes[1].set_title(f"Example {horizon}-day forecast (last test window)")
            axes[1].set_ylabel("kWh / day")
            axes[1].legend()
            axes[1].grid(alpha=0.3)

        fig.tight_layout()
        fig.savefig(plot_path, dpi=120)
        plt.close(fig)

    result = {
        "model": ckpt["model_type"],
        "data_path": str(args.data),
        "test_windows": int(len(X_test)),
        "horizon": horizon,
        "multi_household": multi,
        "model_metrics": model_metrics,
        "baseline_metrics": baseline_metrics,
        "improvement_vs_seasonal_naive_pct": {
            "mae": 100.0
            * (1 - model_metrics["mae"] / (baseline_metrics["seasonal_naive_p7"]["mae"] + EPS)),
            "rmse": 100.0
            * (1 - model_metrics["rmse"] / (baseline_metrics["seasonal_naive_p7"]["rmse"] + EPS)),
        },
        "per_horizon_mae": per_horizon_mae,
    }
    if stats:
        result["stats_baselines"] = stats
    if coverage is not None:
        result["calibration"] = {
            "method": calib["method"],
            "scale": calib["scale"],
            "gamma": calib["gamma"],
            "n_val": calib["n_cal"],
            "path": str(args.calibration_out),
            "coverage": coverage,
        }
    json_path = out_dir / "evaluation.json"
    json_path.write_text(json.dumps(result, indent=2))

    # --- Rolling-origin cross-validation (optional) ---
    cv_result = None
    if args.cv == "rolling":
        cv_result = rolling_cv(
            args,
            model,
            scaler,
            device,
            input_len,
            horizon,
            feature_config,
            feature_scaler,
            prepared,
        )
        cv_path = Path(args.cv_out) if args.cv_out else out_dir / "evaluation_cv.json"
        cv_path.write_text(json.dumps(cv_result, indent=2))

    # --- Report ---
    print(f"Model: {ckpt['model_type'].upper()}   test windows: {len(X_test)}   horizon: {horizon}")
    print(f"{'metric':<10}{'model':>10}{'seasonal7':>12}{'persist':>12}")
    for m in ("mae", "rmse", "wape", "mape"):
        print(
            f"{m.upper():<10}{model_metrics[m]:>10.3f}"
            f"{baseline_metrics['seasonal_naive_p7'][m]:>12.3f}"
            f"{baseline_metrics['persistence'][m]:>12.3f}"
        )
    print(
        f"\nPer-horizon MAE (day 1..{horizon}): " + ", ".join(f"{v:.3f}" for v in per_horizon_mae)
    )
    if coverage is not None:
        print(
            f"\nConformal intervals ({calib['scale']}"
            + (f", gamma={calib['gamma']:g}" if calib["scale"] == "power" else "")
            + f"; {calib['n_cal']} val windows):"
        )
        print(f"{'nominal':>10}{'empirical':>12}{'mean width':>13}")
        for c in coverage.values():
            print(
                f"{c['target'] * 100:>9.0f}%{c['empirical'] * 100:>11.1f}%{c['mean_width']:>12.3f}"
            )
        print(f"Saved: {args.calibration_out}")
    if stats:
        scored = {k: v for k, v in stats.items() if "error" not in v}
        if scored:
            print("\nStatistical baselines (rolling 7-day forecasts):")
            header = f"{'metric':<10}" + "".join(f"{name:>12}" for name in scored)
            print(header)
            for m in ("mae", "rmse", "wape", "mape"):
                print(f"{m.upper():<10}" + "".join(f"{v[m]:>12.3f}" for v in scored.values()))
        for name, v in stats.items():
            if "error" in v:
                print(f"  ({name}: unavailable - {v['error']})")
    if cv_result is not None:
        print(
            f"\nRolling-origin CV ({cv_result['n_folds']} folds, "
            f"{cv_result['min_train_size']}d min train):"
        )
        print(f"{'fold':>5}{'windows':>9}{'MAE':>9}{'RMSE':>9}{'WAPE':>9}")
        for r in cv_result["folds"]:
            m = r["metrics"]
            print(
                f"{r['fold']:>5}{r['test_windows']:>9}"
                f"{m['mae']:>9.3f}{m['rmse']:>9.3f}{m['wape']:>9.2f}"
            )
        print(f"WAPE across folds: {cv_result['wape_mean']:.2f}% +/- {cv_result['wape_std']:.2f}%")
        cv_out = Path(args.cv_out) if args.cv_out else out_dir / "evaluation_cv.json"
        print(f"Saved: {cv_out}")
    print(f"\nSaved: {json_path}")
    print(f"Saved: {plot_path}")

    manifest.add_output("evaluation", json_path)
    manifest.add_output("forecast_plot", plot_path)
    if coverage is not None:
        manifest.add_output("calibration", args.calibration_out)
    if cv_result is not None:
        manifest.add_output("evaluation_cv", cv_path)
    manifest.set("device", str(device))
    manifest.set("test_windows", int(len(X_test)))
    manifest.set("model_metrics", model_metrics)
    manifest.record_duration("evaluate", time.perf_counter() - start)
    log.info(
        "event=evaluate_complete wape=%.3f mae=%.3f test_windows=%d",
        model_metrics["wape"],
        model_metrics["mae"],
        len(X_test),
    )


if __name__ == "__main__":
    main()
