"""BijliPredict AI - random hyperparameter search for the forecaster.

Samples architectures and optimiser settings from a configurable space, trains
each on the same split and keeps the one with the lowest validation loss. The
winning configuration is reported, and optionally retrained and saved as a
standard checkpoint so ``evaluate``/``predict`` can consume it directly.

Outputs:
    outputs/tuning/tuning.json     (every trial + the best config)
    outputs/tuning/tuning.csv      (flat trial table, sorted by val loss)
    <--save-artifacts DIR>/        (optional: best_model.pt + scalers)

Example:
    python tune.py --trials 16 --epochs 30 --save-artifacts artifacts/tuned
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from features import temperature_for_dates
from inference import resolve_device
from logging_utils import add_log_level_argument, get_logger, setup_logging
from manifest import RunManifest, add_manifest_arguments
from preprocessing import load_and_prepare, load_consumption
from train import (
    enable_determinism,
    feature_config_for,
    fit_model,
    save_checkpoint,
    set_seed,
)

DEFAULT_DATA = "data/processed/daily_consumption.csv"
DEFAULT_OUTPUTS = "outputs/tuning"

BATCH_CHOICES = (16, 32, 64)
HIDDEN1_CHOICES = (32, 64, 96)
HIDDEN2_CHOICES = (16, 32, 48)
DROPOUT_CHOICES = (0.0, 0.1, 0.2, 0.3)
DENSE_CHOICES = (16, 32, 64)
WINDOW_CHOICES = (14, 21, 30)


def sample_config(rng: np.random.Generator, args) -> dict:
    """Draw one hyperparameter configuration from the search space."""
    cfg = {
        "model": args.model,
        "lr": float(10 ** rng.uniform(np.log10(args.lr_min), np.log10(args.lr_max))),
        "batch_size": int(rng.choice(BATCH_CHOICES)),
        "hidden1": int(rng.choice(HIDDEN1_CHOICES)),
        "hidden2": int(rng.choice(HIDDEN2_CHOICES)),
        "dropout": float(rng.choice(DROPOUT_CHOICES)),
        "dense": int(rng.choice(DENSE_CHOICES)),
        "window": int(rng.choice(WINDOW_CHOICES)) if args.search_window else int(args.window),
    }
    return cfg


def prepare_once(args, feature_config, window: int, cache: dict) -> dict:
    """Load and window the dataset for ``window`` (memoised across trials)."""
    if window not in cache:
        temperature = None
        if feature_config.temperature:
            raw = load_consumption(args.data, validate=not args.no_validate)
            temperature = temperature_for_dates(
                feature_config, raw.index, weather_file=args.weather_file
            )
        cache[window] = load_and_prepare(
            args.data,
            input_len=window,
            horizon=args.horizon,
            splits=tuple(args.splits),
            save_scaler=False,
            feature_config=feature_config,
            temperature=temperature,
            validate=not args.no_validate,
        )
    return cache[window]


def _save_best(args, cfg: dict, feature_config, device, out_dir: Path, log) -> Path:
    """Retrain the best config and persist a checkpoint plus its scalers."""
    artifacts = Path(args.save_artifacts)
    artifacts.mkdir(parents=True, exist_ok=True)
    scaler_path = artifacts / "scaler.joblib"
    feature_scaler_path = artifacts / "feature_scaler.joblib"

    temperature = None
    if feature_config.temperature:
        raw = load_consumption(args.data, validate=not args.no_validate)
        temperature = temperature_for_dates(
            feature_config, raw.index, weather_file=args.weather_file
        )
    prepared = load_and_prepare(
        args.data,
        input_len=cfg["window"],
        horizon=args.horizon,
        splits=tuple(args.splits),
        scaler_path=scaler_path,
        save_scaler=True,
        feature_config=feature_config,
        temperature=temperature,
        feature_scaler_path=feature_scaler_path,
        validate=not args.no_validate,
    )
    fit = fit_model(
        prepared,
        model_type=cfg["model"],
        epochs=args.final_epochs or args.epochs,
        batch_size=cfg["batch_size"],
        lr=cfg["lr"],
        patience=args.patience,
        seed=args.seed,
        deterministic=args.deterministic,
        device=device,
        hidden1=cfg["hidden1"],
        hidden2=cfg["hidden2"],
        dropout=cfg["dropout"],
        dense=cfg["dense"],
        verbose=args.verbose,
        log=log,
    )
    model_path = artifacts / "best_model.pt"
    save_checkpoint(
        model_path,
        model=fit["model"],
        model_type=cfg["model"],
        model_config=fit["model_config"],
        input_len=cfg["window"],
        horizon=args.horizon,
        scaler_path=scaler_path,
        feature_scaler_path=feature_scaler_path,
        feature_columns=prepared["feature_columns"],
        feature_config=feature_config,
        history=fit["history"],
        best_val_loss=fit["best_val_loss"],
        data_path=str(args.data),
        seed=args.seed,
        deterministic=args.deterministic,
    )
    log.info("event=tune_saved model=%s", model_path)
    return model_path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Random hyperparameter search for the forecaster.")
    p.add_argument("--data", default=DEFAULT_DATA)
    p.add_argument("--model", choices=["gru", "lstm"], default="gru")
    p.add_argument("--trials", type=int, default=12, help="number of configurations to sample")
    p.add_argument("--epochs", type=int, default=40, help="epochs per trial")
    p.add_argument("--final-epochs", type=int, default=None, help="epochs for the final retrain")
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--lr-min", type=float, default=1e-4)
    p.add_argument("--lr-max", type=float, default=1e-2)
    p.add_argument("--window", type=int, default=30)
    p.add_argument(
        "--search-window",
        action="store_true",
        help="also search input length over {14, 21, 30}",
    )
    p.add_argument("--horizon", type=int, default=7)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--splits", type=float, nargs=3, default=(0.70, 0.15, 0.15))
    p.add_argument(
        "--features",
        choices=["none", "calendar", "weather"],
        default="calendar",
        help="feature set held fixed across trials",
    )
    p.add_argument("--city", default="delhi")
    p.add_argument("--weather-file", default=None)
    p.add_argument("--no-holidays", action="store_true")
    p.add_argument("--out-dir", default=DEFAULT_OUTPUTS)
    p.add_argument(
        "--save-artifacts",
        default=None,
        help="directory to retrain + save the best checkpoint into",
    )
    p.add_argument("--quiet", dest="verbose", action="store_false")
    p.add_argument("--no-validate", action="store_true")
    add_log_level_argument(p)
    add_manifest_arguments(p)
    return p


def main(args: argparse.Namespace | None = None) -> None:
    if args is None:
        args = build_parser().parse_args()
    setup_logging(args.log_level)
    log = get_logger("tune")
    manifest = RunManifest(
        "tune",
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
    log.info("event=run_end command=tune status=ok manifest=%s", path)


def _run(args: argparse.Namespace, log, manifest: RunManifest) -> None:
    start = time.perf_counter()
    if args.deterministic:
        enable_determinism(args.seed)
    else:
        set_seed(args.seed)
    device = resolve_device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    feature_config = feature_config_for(
        args.features, city=args.city, holidays=not args.no_holidays
    )
    log.info(
        "event=run_start command=tune trials=%d model=%s features=%s device=%s",
        args.trials,
        args.model,
        args.features,
        device,
    )

    rng = np.random.default_rng(args.seed)
    cache: dict[int, dict] = {}
    trials: list[dict] = []
    best = None
    print(f"Tuning: {args.trials} trial(s), model={args.model}, features={args.features}")

    for i in range(1, args.trials + 1):
        cfg = sample_config(rng, args)
        prepared = prepare_once(args, feature_config, cfg["window"], cache)
        fit = fit_model(
            prepared,
            model_type=cfg["model"],
            epochs=args.epochs,
            batch_size=cfg["batch_size"],
            lr=cfg["lr"],
            patience=args.patience,
            seed=args.seed,
            deterministic=args.deterministic,
            device=device,
            hidden1=cfg["hidden1"],
            hidden2=cfg["hidden2"],
            dropout=cfg["dropout"],
            dense=cfg["dense"],
            verbose=False,
            log=log,
        )
        row = {
            "trial": i,
            **cfg,
            "n_features": len(prepared["feature_columns"]),
            "best_val_loss": fit["best_val_loss"],
            "epochs_trained": fit["epochs_trained"],
        }
        trials.append(row)
        marker = ""
        if best is None or row["best_val_loss"] < best["best_val_loss"]:
            best = row
            marker = " *"
        print(
            f"  trial {i:>3}/{args.trials}  val={row['best_val_loss']:.5f}  "
            f"lr={cfg['lr']:.2e} bs={cfg['batch_size']} h1={cfg['hidden1']} "
            f"h2={cfg['hidden2']} do={cfg['dropout']} dense={cfg['dense']} "
            f"win={cfg['window']}{marker}"
        )

    ranking = sorted(trials, key=lambda r: r["best_val_loss"])
    best_config = {
        k: best[k]
        for k in (
            "model",
            "lr",
            "batch_size",
            "hidden1",
            "hidden2",
            "dropout",
            "dense",
            "window",
        )
    }
    result = {
        "data_path": str(args.data),
        "horizon": args.horizon,
        "splits": list(args.splits),
        "seed": args.seed,
        "n_trials": len(trials),
        "trials": trials,
        "best_trial": best["trial"],
        "best_val_loss": best["best_val_loss"],
        "best_config": best_config,
        "ranking": [r["trial"] for r in ranking],
    }

    json_path = out_dir / "tuning.json"
    csv_path = out_dir / "tuning.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(trials[0].keys()))
        writer.writeheader()
        writer.writerows(ranking)

    outputs = [json_path, csv_path]
    if args.save_artifacts:
        model_path = _save_best(args, best_config, feature_config, device, out_dir, log)
        result["saved_model"] = str(model_path)
        outputs.append(model_path)

    json_path.write_text(json.dumps(result, indent=2))

    # --- Report ---
    print(f"\nBest trial #{best['trial']}  val loss={best['best_val_loss']:.5f}")
    print(
        "  lr={lr:.2e}  batch_size={batch_size}  hidden1={hidden1}  hidden2={hidden2}  "
        "dropout={dropout}  dense={dense}  window={window}".format(**best_config)
    )
    for path in outputs:
        print(f"Saved: {path}")

    manifest.add_output("tuning", json_path)
    manifest.add_output("tuning_csv", csv_path)
    if args.save_artifacts:
        manifest.add_output("model", result["saved_model"])
    manifest.set("device", str(device))
    manifest.set("n_trials", len(trials))
    manifest.set("best_config", best_config)
    manifest.set("best_val_loss", best["best_val_loss"])
    manifest.record_duration("tune", time.perf_counter() - start)
    log.info(
        "event=tune_complete trials=%d best_val_loss=%.5f",
        len(trials),
        best["best_val_loss"],
    )


if __name__ == "__main__":
    main()
