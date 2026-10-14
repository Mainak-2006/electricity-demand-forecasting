"""Tests for the hyperparameter tuner (`tune.py`)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

import tune
from inference import load_model


def _dataset(tmp_path, n=300, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    vals = 3.0 + 0.5 * np.sin(np.arange(n) / 7.0) + rng.normal(0, 0.3, n)
    path = tmp_path / "data.csv"
    pd.DataFrame(
        {"date": dates.strftime("%Y-%m-%d"), "consumption_kwh": np.round(vals, 3)}
    ).to_csv(path, index=False)
    return path


def _args(data, out, extra=()):
    parser = tune.build_parser()
    return parser.parse_args(
        ["--data", str(data), "--out-dir", str(out), "--no-manifest", "--quiet", *extra]
    )


def test_sample_config_respects_space():
    args = tune.build_parser().parse_args([])
    rng = np.random.default_rng(0)
    for _ in range(25):
        cfg = tune.sample_config(rng, args)
        assert args.lr_min <= cfg["lr"] <= args.lr_max
        assert cfg["batch_size"] in tune.BATCH_CHOICES
        assert cfg["hidden1"] in tune.HIDDEN1_CHOICES
        assert cfg["hidden2"] in tune.HIDDEN2_CHOICES
        assert cfg["dropout"] in tune.DROPOUT_CHOICES
        assert cfg["dense"] in tune.DENSE_CHOICES
        assert cfg["window"] == args.window


def test_sample_config_can_search_window():
    args = tune.build_parser().parse_args(["--search-window"])
    rng = np.random.default_rng(1)
    windows = {tune.sample_config(rng, args)["window"] for _ in range(40)}
    assert windows <= set(tune.WINDOW_CHOICES)


def test_tune_writes_outputs(tmp_path):
    data = _dataset(tmp_path)
    out = tmp_path / "tune"
    args = _args(
        data, out, ["--trials", "2", "--epochs", "1", "--window", "14", "--horizon", "3"]
    )
    tune.main(args)

    result = json.loads((out / "tuning.json").read_text())
    assert result["n_trials"] == 2
    assert set(result["best_config"]) == {
        "model",
        "lr",
        "batch_size",
        "hidden1",
        "hidden2",
        "dropout",
        "dense",
        "window",
    }
    losses = [r["best_val_loss"] for r in result["trials"]]
    assert result["best_val_loss"] == min(losses)
    assert (out / "tuning.csv").exists()


def test_tune_saves_loadable_checkpoint(tmp_path):
    data = _dataset(tmp_path)
    out = tmp_path / "tune"
    artifacts = tmp_path / "artifacts"
    args = _args(
        data,
        out,
        [
            "--trials",
            "1",
            "--epochs",
            "1",
            "--window",
            "14",
            "--horizon",
            "3",
            "--save-artifacts",
            str(artifacts),
        ],
    )
    tune.main(args)

    assert (artifacts / "best_model.pt").exists()
    assert (artifacts / "scaler.joblib").exists()
    bundle = load_model(artifacts / "best_model.pt", "cpu")
    assert bundle["ckpt"]["horizon"] == 3
    assert bundle["ckpt"]["input_len"] == 14
    result = json.loads((out / "tuning.json").read_text())
    assert result["saved_model"] == str(artifacts / "best_model.pt")
