"""BijliPredict AI - evaluate the trained model on the held-out test period.

Reports MAE, RMSE and WAPE (plus MAPE) and compares against two naive
baselines:
    * seasonal-naive (period 7): repeat the last observed week
    * persistence: repeat the last observed day

Outputs:
    outputs/evaluation.json
    outputs/forecast.png
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from inference import load_model, predict, resolve_device
from preprocessing import load_and_prepare, load_scaler

DEFAULT_DATA = "data/processed/daily_consumption.csv"
DEFAULT_ARTIFACTS = "artifacts"
DEFAULT_OUTPUTS = "outputs"

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


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate the trained forecaster.")
    p.add_argument("--data", default=DEFAULT_DATA)
    p.add_argument("--model-path", default=f"{DEFAULT_ARTIFACTS}/best_model.pt")
    p.add_argument("--scaler-path", default=f"{DEFAULT_ARTIFACTS}/scaler.joblib")
    p.add_argument("--out-dir", default=DEFAULT_OUTPUTS)
    p.add_argument("--device", default="cpu", choices=["auto", "cpu", "cuda"])
    p.add_argument("--splits", type=float, nargs=3, default=(0.70, 0.15, 0.15))
    return p


def main() -> None:
    args = build_parser().parse_args()
    device = resolve_device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bundle = load_model(args.model_path, device)
    model, ckpt = bundle["model"], bundle["ckpt"]
    input_len, horizon = ckpt["input_len"], ckpt["horizon"]
    scaler = load_scaler(args.scaler_path)

    prepared = load_and_prepare(
        args.data,
        input_len=input_len,
        horizon=horizon,
        splits=tuple(args.splits),
        save_scaler=False,
        scaler=scaler,
    )
    X_test = prepared["datasets"]["test"]["X"]
    y_test = prepared["datasets"]["test"]["y"]
    if len(X_test) == 0:
        raise SystemExit("No test windows available - dataset too short for the split.")

    # --- Predictions (scaled -> kWh) ---
    pred_scaled = predict(model, X_test, device)

    def inv(a):
        return scaler.inverse_transform(a.reshape(-1, 1)).reshape(a.shape)

    y_true = inv(y_test)
    y_pred = np.clip(inv(pred_scaled), 0.0, None)  # consumption is non-negative

    # --- Baselines (from the raw input window) ---
    X_raw = inv(X_test)
    seasonal_naive = X_raw[:, -horizon:]  # repeat last week
    persistence = np.repeat(X_raw[:, -1:], horizon, axis=1)  # repeat last day

    model_metrics = compute_metrics(y_true, y_pred)
    baseline_metrics = {
        "seasonal_naive_p7": compute_metrics(y_true, seasonal_naive),
        "persistence": compute_metrics(y_true, persistence),
    }
    per_horizon_mae = [float(np.mean(np.abs(y_true[:, k] - y_pred[:, k]))) for k in range(horizon)]

    # --- Reconstruct a per-date series by averaging overlapping forecasts ---
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
    axes[0].plot(plot_dates, plot_pred, label="Predicted (avg)", color="tab:red", linewidth=1.2)
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
            td, seasonal_naive[last, : len(td)], "^:", label="Seasonal-naive", color="tab:blue"
        )
        axes[1].set_title(f"Example {horizon}-day forecast (last test window)")
        axes[1].set_ylabel("kWh / day")
        axes[1].legend()
        axes[1].grid(alpha=0.3)

    fig.tight_layout()
    plot_path = out_dir / "forecast.png"
    fig.savefig(plot_path, dpi=120)
    plt.close(fig)

    result = {
        "model": ckpt["model_type"],
        "data_path": str(args.data),
        "test_windows": int(len(X_test)),
        "horizon": horizon,
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
    json_path = out_dir / "evaluation.json"
    json_path.write_text(json.dumps(result, indent=2))

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
    print(f"\nSaved: {json_path}")
    print(f"Saved: {plot_path}")


if __name__ == "__main__":
    main()
