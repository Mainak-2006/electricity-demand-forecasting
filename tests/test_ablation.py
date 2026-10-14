"""Tests for the ablation study (`ablation.py`)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import ablation


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
    parser = ablation.build_parser()
    return parser.parse_args(
        [
            "--data",
            str(data),
            "--out-dir",
            str(out),
            "--no-manifest",
            "--quiet",
            *extra,
        ]
    )


def test_comma_list_parsing():
    assert ablation._comma_list("gru,lstm") == ["gru", "lstm"]
    assert ablation._comma_list("none, calendar") == ["none", "calendar"]
    assert ablation._comma_list(["gru"]) == ["gru"]


def test_count_params_positive():
    from models import get_model

    model = get_model("gru", input_size=3, output_size=7)
    assert ablation.count_params(model) > 0


def test_ablation_writes_outputs(tmp_path):
    data = _dataset(tmp_path)
    out = tmp_path / "abl"
    args = _args(
        data,
        out,
        ["--models", "gru", "--features", "none,calendar", "--epochs", "1",
         "--window", "14", "--horizon", "3"],
    )
    ablation.main(args)

    result = json.loads((out / "ablation.json").read_text())
    assert result["n_variants"] == 2
    assert {r["variant"] for r in result["variants"]} == {"gru+none", "gru+calendar"}
    assert all(r["test_wape"] > 0 for r in result["variants"])
    assert (out / "ablation.csv").exists()
    assert (out / "ablation.png").exists()

    by_variant = {r["variant"]: r["test_wape"] for r in result["variants"]}
    ordered = [by_variant[v] for v in result["ranking"]]
    assert ordered == sorted(ordered)
    assert result["best_variant"] == result["ranking"][0]


def test_ablation_rejects_unknown_variant(tmp_path):
    data = _dataset(tmp_path, n=200)
    args = _args(data, tmp_path / "abl", ["--models", "transformer", "--epochs", "1"])
    with pytest.raises(SystemExit):
        ablation.main(args)
