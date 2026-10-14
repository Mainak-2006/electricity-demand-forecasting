"""BijliPredict AI - ablation study over model architecture and feature blocks.

Trains one model per (model x feature-set) combination on the *same* split and
scores each on the held-out test windows, so the marginal contribution of every
design choice is directly comparable. A seasonal-naive baseline (repeat the last
observed week) is scored the same way as a reference floor.

Outputs:
    outputs/ablation/ablation.json   (full per-variant metrics + ranking)
    outputs/ablation/ablation.csv    (flat table, sorted by test WAPE)
    outputs/ablation/ablation.png    (test WAPE per variant)

Example:
    python ablation.py --models gru,lstm --features none,calendar --epochs 30
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import evaluate
from features import FeatureConfig, temperature_for_dates
from inference import resolve_device
from logging_utils import add_log_level_argument, get_logger, setup_logging
from manifest import RunManifest, add_manifest_arguments
from preprocessing import load_and_prepare, load_consumption
from train import enable_determinism, feature_config_for, fit_model, set_seed

DEFAULT_DATA = "data/processed/daily_consumption.csv"
DEFAULT_OUTPUTS = "outputs/ablation"
FEATURE_SETS = ("none", "calendar", "weather")


def count_params(model) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def _comma_list(value: str | list[str]) -> list[str]:
    """Parse ``a,b`` (or an already-parsed list from config.toml) into a list."""
    if isinstance(value, list):
        return value
    return [item.strip() for item in value.replace(" ", ",").split(",") if item.strip()]


def prepare_for(args, feature_config: FeatureConfig) -> tuple[dict, object | None]:
    """Load and window the dataset for one feature configuration."""
    temperature = None
    if feature_config.temperature:
        raw = load_consumption(args.data, validate=not args.no_validate)
        temperature = temperature_for_dates(
            feature_config, raw.index, weather_file=args.weather_file
        )
    prepared = load_and_prepare(
        args.data,
        input_len=args.window,
        horizon=args.horizon,
        splits=tuple(args.splits),
        save_scaler=False,
        feature_config=feature_config,
        temperature=temperature,
        validate=not args.no_validate,
    )
    return prepared, temperature


def run_variant(args, model_type: str, feature_name: str, device, log) -> dict:
    """Train and score one (model, feature-set) variant. Returns its metrics row."""
    feature_config = feature_config_for(feature_name, city=args.city, holidays=not args.no_holidays)
    variant = f"{model_type}+{feature_name}"
    log.info("event=ablation_variant variant=%s", variant)

    prepared, _ = prepare_for(args, feature_config)
    fit = fit_model(
        prepared,
        model_type=model_type,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        seed=args.seed,
        deterministic=args.deterministic,
        device=device,
        verbose=args.verbose,
        log=log,
    )
    model = fit["model"]
    scaler = prepared["scaler"]
    horizon = prepared["horizon"]

    X_val, y_val = prepared["datasets"]["val"]["X"], prepared["datasets"]["val"]["y"]
    val_metrics = None
    if len(X_val) > 0:
        _, _, val_metrics = evaluate.window_metrics(model, X_val, y_val, scaler, device)

    X_test, y_test = prepared["datasets"]["test"]["X"], prepared["datasets"]["test"]["y"]
    if len(X_test) == 0:
        raise SystemExit(f"Variant {variant}: no test windows available.")
    y_true, y_pred, test_metrics = evaluate.window_metrics(model, X_test, y_test, scaler, device)

    # Seasonal-naive floor scored exactly like the model (repeat the last week).
    x_raw = scaler.inverse_transform(X_test[:, :, 0].reshape(-1, 1)).reshape(X_test[:, :, 0].shape)
    seasonal_naive = np.clip(x_raw[:, -horizon:], 0.0, None)
    baseline = evaluate.compute_metrics(y_true, seasonal_naive)

    row = {
        "variant": variant,
        "model": model_type,
        "features": feature_name,
        "n_features": len(prepared["feature_columns"]),
        "params": count_params(model),
        "epochs_trained": fit["epochs_trained"],
        "best_val_loss": fit["best_val_loss"],
        "val_mae": None if val_metrics is None else val_metrics["mae"],
        "test_mae": test_metrics["mae"],
        "test_rmse": test_metrics["rmse"],
        "test_wape": test_metrics["wape"],
        "test_mape": test_metrics["mape"],
        "seasonal_naive_wape": baseline["wape"],
        "wape_improvement_pct": 100.0
        * (1.0 - test_metrics["wape"] / (baseline["wape"] + evaluate.EPS)),
    }
    if args.verbose:
        print(
            f"  {variant:<16} test WAPE={row['test_wape']:.2f}%  "
            f"MAE={row['test_mae']:.3f}  (naive {row['seasonal_naive_wape']:.2f}%)"
        )
    return row


def _write_plot(rows: list[dict], path: Path) -> None:
    ordered = sorted(rows, key=lambda r: r["test_wape"])
    names = [r["variant"] for r in ordered]
    values = [r["test_wape"] for r in ordered]
    fig, ax = plt.subplots(figsize=(9, max(3.0, 0.5 * len(names) + 1.5)))
    y = np.arange(len(names))
    ax.barh(y, values, color="tab:red")
    ax.set_yticks(y, names)
    ax.invert_yaxis()
    ax.set_xlabel("Test WAPE (%) - lower is better")
    ax.set_title("Ablation: model x feature-set (held-out test windows)")
    for yi, v in zip(y, values, strict=True):
        ax.text(v, yi, f" {v:.2f}%", va="center", fontsize=8)
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run the model/feature ablation study.")
    p.add_argument("--data", default=DEFAULT_DATA)
    p.add_argument(
        "--models",
        type=_comma_list,
        default=["gru", "lstm"],
        help="comma-separated architectures to compare (gru,lstm)",
    )
    p.add_argument(
        "--features",
        type=_comma_list,
        default=["none", "calendar"],
        help="comma-separated feature sets (none,calendar,weather)",
    )
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--window", type=int, default=30)
    p.add_argument("--horizon", type=int, default=7)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--splits", type=float, nargs=3, default=(0.70, 0.15, 0.15))
    p.add_argument("--city", default="delhi", help="city for the weather feature set")
    p.add_argument("--weather-file", default=None, help="pre-fetched weather CSV")
    p.add_argument("--no-holidays", action="store_true", help="drop the holiday flag")
    p.add_argument("--out-dir", default=DEFAULT_OUTPUTS)
    p.add_argument("--quiet", dest="verbose", action="store_false", help="less output")
    p.add_argument(
        "--no-validate", action="store_true", help="skip input-data validation"
    )
    add_log_level_argument(p)
    add_manifest_arguments(p)
    return p


def main(args: argparse.Namespace | None = None) -> None:
    if args is None:
        args = build_parser().parse_args()
    setup_logging(args.log_level)
    log = get_logger("ablation")
    manifest = RunManifest(
        "ablation",
        args=vars(args),
        manifest_dir=args.manifest_dir,
        enabled=not args.no_manifest,
    )
    manifest.add_input("data", args.data)
    try:
        _run(args, log, manifest)
    except Exception as exc:
        manifest.set("error", f"{type(exc).__name__}: {exc}")
        manifest.write(status="error", error=f"{type(exc).__name__}: {exc}")
        raise
    path = manifest.write()
    log.info("event=run_end command=ablation status=ok manifest=%s", path)


def _run(args: argparse.Namespace, log, manifest: RunManifest) -> None:
    start = time.perf_counter()
    if args.deterministic:
        enable_determinism(args.seed)
    else:
        set_seed(args.seed)
    device = resolve_device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    invalid = [m for m in args.models if m not in ("gru", "lstm")] + [
        f for f in args.features if f not in FEATURE_SETS
    ]
    if invalid:
        raise SystemExit(
            f"ablation: unknown model/feature {invalid}; "
            f"models are gru,lstm and features are {', '.join(FEATURE_SETS)}"
        )
    log.info(
        "event=run_start command=ablation models=%s features=%s device=%s",
        args.models,
        args.features,
        device,
    )

    rows: list[dict] = []
    total = len(args.models) * len(args.features)
    print(f"Ablation: {total} variant(s) on {args.data}")
    for model_type in args.models:
        for feature_name in args.features:
            rows.append(run_variant(args, model_type, feature_name, device, log))

    ranking = sorted(rows, key=lambda r: r["test_wape"])
    result = {
        "data_path": str(args.data),
        "window": args.window,
        "horizon": args.horizon,
        "splits": list(args.splits),
        "seed": args.seed,
        "n_variants": len(rows),
        "variants": rows,
        "best_variant": ranking[0]["variant"],
        "ranking": [r["variant"] for r in ranking],
    }

    json_path = out_dir / "ablation.json"
    json_path.write_text(json.dumps(result, indent=2))

    csv_path = out_dir / "ablation.csv"
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(ranking)

    plot_path = out_dir / "ablation.png"
    _write_plot(rows, plot_path)

    # --- Report ---
    print(f"\n{'variant':<18}{'params':>8}{'test MAE':>10}{'test WAPE':>11}{'vs naive':>10}")
    for r in ranking:
        print(
            f"{r['variant']:<18}{r['params']:>8}{r['test_mae']:>10.3f}"
            f"{r['test_wape']:>11.2f}{r['wape_improvement_pct']:>9.1f}%"
        )
    print(f"\nBest variant: {result['best_variant']}")
    print(f"Saved: {json_path}")
    print(f"Saved: {csv_path}")
    print(f"Saved: {plot_path}")

    manifest.add_output("ablation", json_path)
    manifest.add_output("ablation_csv", csv_path)
    manifest.add_output("ablation_plot", plot_path)
    manifest.set("device", str(device))
    manifest.set("best_variant", result["best_variant"])
    manifest.set("ranking", result["ranking"])
    manifest.record_duration("ablation", time.perf_counter() - start)
    log.info(
        "event=ablation_complete variants=%d best=%s",
        len(rows),
        result["best_variant"],
    )


if __name__ == "__main__":
    main()
