"""BijliPredict AI - train a GRU (or LSTM) 7-day electricity forecaster.

Example:
    python train.py --model gru --epochs 100
"""

from __future__ import annotations

import argparse
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from features import FeatureConfig, temperature_for_dates
from inference import resolve_device
from logging_utils import add_log_level_argument, get_logger, setup_logging
from manifest import RunManifest, add_manifest_arguments
from models import get_model
from preprocessing import load_and_prepare, load_consumption

DEFAULT_DATA = "data/processed/daily_consumption.csv"
DEFAULT_ARTIFACTS = "artifacts"


def feature_config_for(name: str, *, city: str = "delhi", holidays: bool = True) -> FeatureConfig:
    """Map a feature-set name (none | calendar | weather) onto a FeatureConfig.

    Shared by training, the ablation study and the tuner so all three agree on
    what a feature set means. ``weather`` additionally adds temperature features.
    """
    if name == "none":
        return FeatureConfig()
    return FeatureConfig(calendar=True, holidays=holidays, temperature=name == "weather", city=city)


def resolve_feature_config(args: argparse.Namespace) -> FeatureConfig:
    """Map the --features switch onto a FeatureConfig.

    ``calendar`` turns on calendar+holiday blocks; ``weather`` additionally adds
    temperature. ``--no-holidays`` drops the holiday flag from either.
    """
    return feature_config_for(args.features, city=args.city, holidays=not args.no_holidays)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def enable_determinism(seed: int) -> None:
    """Seed every RNG and force deterministic kernels for reproducible runs."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    set_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_loader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    generator: torch.Generator | None = None,
) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, drop_last=False, generator=generator
    )


def fit_model(
    prepared: dict,
    *,
    model_type: str,
    epochs: int,
    batch_size: int,
    lr: float,
    patience: int = 15,
    seed: int = 42,
    deterministic: bool = False,
    device: str | torch.device = "cpu",
    hidden1: int = 64,
    hidden2: int = 32,
    dropout: float = 0.2,
    dense: int = 32,
    verbose: bool = True,
    log=None,
) -> dict:
    """Train one model on a prepared dataset and return it with its history.

    Shared by ``train.py``, ``ablation.py`` and ``tune.py`` so every experiment
    runs the identical loop, seeding, early stopping and best-checkpoint
    selection. ``prepared`` is a :func:`preprocessing.load_and_prepare` result.
    """
    data = prepared["datasets"]
    horizon = prepared["horizon"]
    X_train, y_train = data["train"]["X"], data["train"]["y"]
    X_val, y_val = data["val"]["X"], data["val"]["y"]
    if len(X_train) == 0:
        raise SystemExit("No training windows produced - dataset too short.")
    if len(X_val) == 0:
        if log is not None:
            log.warning("event=empty_validation action=disable_early_stopping")
        X_val, y_val = X_train, y_train

    train_generator = None
    if deterministic:
        train_generator = torch.Generator()
        train_generator.manual_seed(seed)

    train_loader = make_loader(X_train, y_train, batch_size, shuffle=True, generator=train_generator)
    val_loader = make_loader(X_val, y_val, batch_size, shuffle=False)

    model_config = {
        "input_size": 1 + len(prepared["feature_columns"]),
        "hidden1": hidden1,
        "hidden2": hidden2,
        "dropout": dropout,
        "dense": dense,
        "output_size": horizon,
    }
    model = get_model(model_type, **model_config).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val = float("inf")
    best_state = None
    epochs_no_improve = 0
    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            running += loss.item() * len(xb)
        train_loss = running / len(train_loader.dataset)

        model.eval()
        val_running = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                val_running += criterion(model(xb), yb).item() * len(xb)
        val_loss = val_running / len(val_loader.dataset)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        improved = val_loss < best_val - 1e-6
        if improved:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if verbose and (epoch == 1 or epoch % 10 == 0 or improved):
            flag = " *" if improved else ""
            print(
                f"epoch {epoch:3d}/{epochs}  train={train_loss:.5f}  val={val_loss:.5f}{flag}"
            )

        if patience > 0 and epochs_no_improve >= patience:
            if verbose:
                print(f"Early stopping at epoch {epoch} (no improvement for {patience}).")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "model": model,
        "model_config": model_config,
        "history": history,
        "best_val_loss": best_val,
        "epochs_trained": len(history["train_loss"]),
    }


def save_checkpoint(
    path: str | Path,
    *,
    model,
    model_type: str,
    model_config: dict,
    input_len: int,
    horizon: int,
    scaler_path: str | Path,
    feature_scaler_path: str | Path | None,
    feature_columns: list[str],
    feature_config: FeatureConfig,
    history: dict,
    best_val_loss: float,
    data_path: str,
    seed: int,
    deterministic: bool,
) -> Path:
    """Persist a training checkpoint in the format inference/evaluate expect."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "model_type": model_type,
        "model_config": model_config,
        "state_dict": model.state_dict(),
        "input_len": input_len,
        "horizon": horizon,
        "scaler_path": str(scaler_path),
        "history": history,
        "epochs_trained": len(history["train_loss"]),
        "best_val_loss": best_val_loss,
        "data_path": str(data_path),
        "seed": seed,
        "deterministic": deterministic,
        "feature_config": feature_config.to_dict(),
        "feature_columns": feature_columns,
        "feature_scaler_path": str(feature_scaler_path) if feature_columns else None,
        "city": feature_config.city,
    }
    torch.save(ckpt, path)
    return path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train the BijliPredict forecaster.")
    p.add_argument("--data", default=DEFAULT_DATA, help="processed consumption CSV")
    p.add_argument("--model", choices=["gru", "lstm"], default="gru")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--window", type=int, default=30, help="input length (days)")
    p.add_argument("--horizon", type=int, default=7, help="forecast length (days)")
    p.add_argument("--patience", type=int, default=15, help="early-stopping patience")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--deterministic",
        action="store_true",
        help="force deterministic kernels and seeded shuffling for reproducible runs",
    )
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--artifacts-dir", default=DEFAULT_ARTIFACTS)
    p.add_argument(
        "--features",
        choices=["none", "calendar", "weather"],
        default="calendar",
        help="exogenous features: 'calendar' adds calendar+holiday, 'weather' also temperature",
    )
    p.add_argument("--city", default="delhi", help="city for temperature features")
    p.add_argument(
        "--no-holidays",
        action="store_true",
        help="drop the holiday flag from the calendar feature set",
    )
    p.add_argument(
        "--weather-file",
        default=None,
        help="pre-fetched weather CSV to use instead of the Open-Meteo cache",
    )
    p.add_argument(
        "--splits",
        type=float,
        nargs=3,
        default=(0.70, 0.15, 0.15),
        metavar=("TRAIN", "VAL", "TEST"),
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
    log = get_logger("train")
    manifest = RunManifest(
        "train",
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
    log.info("event=run_end command=train status=ok manifest=%s", path)


def _run(args: argparse.Namespace, log, manifest: RunManifest) -> None:
    start = time.perf_counter()
    if args.deterministic:
        enable_determinism(args.seed)
    else:
        set_seed(args.seed)
    device = resolve_device(args.device)
    log.info(
        "event=run_start command=train model=%s device=%s features=%s deterministic=%s",
        args.model,
        device,
        args.features,
        args.deterministic,
    )

    artifacts = Path(args.artifacts_dir)
    artifacts.mkdir(parents=True, exist_ok=True)
    scaler_path = artifacts / "scaler.joblib"
    feature_scaler_path = artifacts / "feature_scaler.joblib"

    feature_config = resolve_feature_config(args)
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
        scaler_path=scaler_path,
        save_scaler=True,
        feature_config=feature_config,
        temperature=temperature,
        feature_scaler_path=feature_scaler_path,
        validate=not args.no_validate,
    )
    data = prepared["datasets"]
    feature_columns = prepared["feature_columns"]
    X_train = data["train"]["X"]
    X_val = data["val"]["X"]

    print(f"Device : {device}")
    print(f"Model  : {args.model.upper()}  (window={args.window}, horizon={args.horizon})")
    if feature_columns:
        print(f"Features ({len(feature_columns)}): {', '.join(feature_columns)}")
    else:
        print("Features: none")
    print(f"Train  : {X_train.shape}   Val: {X_val.shape}")
    if len(X_train) == 0:
        raise SystemExit("No training windows produced - dataset too short.")

    fit = fit_model(
        prepared,
        model_type=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        seed=args.seed,
        deterministic=args.deterministic,
        device=device,
        verbose=True,
        log=log,
    )
    model = fit["model"]
    model_config = fit["model_config"]
    history = fit["history"]
    best_val = fit["best_val_loss"]

    model_path = artifacts / "best_model.pt"
    save_checkpoint(
        model_path,
        model=model,
        model_type=args.model,
        model_config=model_config,
        input_len=args.window,
        horizon=args.horizon,
        scaler_path=scaler_path,
        feature_scaler_path=feature_scaler_path,
        feature_columns=feature_columns,
        feature_config=feature_config,
        history=history,
        best_val_loss=best_val,
        data_path=str(args.data),
        seed=args.seed,
        deterministic=args.deterministic,
    )

    print(f"\nBest val loss : {best_val:.5f}")
    print(f"Model saved   : {model_path}")
    print(f"Scaler saved  : {scaler_path}")
    if feature_columns:
        print(f"Feat scaler   : {feature_scaler_path}")

    manifest.add_output("model", model_path)
    manifest.add_output("scaler", scaler_path)
    if feature_columns:
        manifest.add_output("feature_scaler", feature_scaler_path)
    manifest.set("device", str(device))
    manifest.set("feature_columns", feature_columns)
    manifest.set("train_windows", int(len(X_train)))
    manifest.set("val_windows", int(len(X_val)))
    manifest.set("epochs_trained", len(history["train_loss"]))
    manifest.set("best_val_loss", best_val)
    manifest.record_duration("train", time.perf_counter() - start)
    log.info(
        "event=train_complete epochs=%d best_val_loss=%.5f hours=%.4f",
        len(history["train_loss"]),
        best_val,
        (time.perf_counter() - start) / 3600.0,
    )


if __name__ == "__main__":
    main()
