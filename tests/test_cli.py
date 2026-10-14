"""Tests for the unified CLI (`cli.py`) and tariff presets."""

from __future__ import annotations

import argparse
import importlib

import pytest

import cli
from predict import build_tariff
from tariffs import DEFAULT_PRESETS_PATH, get_preset, load_presets, parse_slabs

# --- config loading -------------------------------------------------------


def test_load_config_reads_sections(tmp_path):
    cfg = tmp_path / "config.toml"
    cfg.write_text('[train]\nepochs = 7\nmodel = "lstm"\n')
    assert cli.load_config(cfg) == {"train": {"epochs": 7, "model": "lstm"}}


def test_load_config_missing_returns_empty(tmp_path):
    assert cli.load_config(tmp_path / "nope.toml") == {}


def test_build_parser_applies_config_defaults():
    parser = cli.build_parser(
        "train",
        importlib.import_module("train"),
        {"train": {"epochs": 7, "model": "lstm", "unknown_key": 1}},
    )
    args = parser.parse_args([])
    assert args.epochs == 7
    assert args.model == "lstm"
    assert not hasattr(args, "unknown_key")  # unknown config keys are ignored


# --- dispatch -------------------------------------------------------------


def test_cli_dispatches_to_module_main(monkeypatch, tmp_path):
    cfg = tmp_path / "config.toml"
    cfg.write_text('[train]\nepochs = 5\nmodel = "lstm"\n')
    captured = {}
    monkeypatch.setattr("train.main", lambda args: captured.setdefault("args", args))

    cli.main(["--config", str(cfg), "train"])

    assert captured["args"].epochs == 5
    assert captured["args"].model == "lstm"
    assert captured["args"].config == str(cfg)


def test_cli_flag_overrides_config(monkeypatch, tmp_path):
    cfg = tmp_path / "config.toml"
    cfg.write_text("[train]\nepochs = 5\n")
    captured = {}
    monkeypatch.setattr("train.main", lambda args: captured.setdefault("args", args))

    cli.main(["--config", str(cfg), "train", "--epochs", "9"])

    assert captured["args"].epochs == 9


def test_cli_config_flag_can_follow_command(monkeypatch, tmp_path):
    cfg = tmp_path / "config.toml"
    cfg.write_text("[train]\nepochs = 3\n")
    captured = {}
    monkeypatch.setattr("train.main", lambda args: captured.setdefault("args", args))

    cli.main(["train", "--config", str(cfg)])

    assert captured["args"].epochs == 3


@pytest.mark.parametrize("command", ["data", "evaluate", "predict", "report"])
def test_cli_dispatches_every_command(monkeypatch, command):
    module_name, _ = cli.COMMANDS[command]
    called = {}
    monkeypatch.setattr(f"{module_name}.main", lambda args: called.setdefault("args", args))
    cli.main([command])
    assert "args" in called


def test_cli_unknown_command_exits():
    with pytest.raises(SystemExit):
        cli.main(["frobnicate"])


def test_cli_no_command_prints_help(capsys):
    cli.main([])
    out = capsys.readouterr().out
    assert "usage: bijli" in out
    assert "predict" in out


# --- tariff presets -------------------------------------------------------


def test_presets_file_loads():
    presets = load_presets(DEFAULT_PRESETS_PATH)
    assert "bses" in presets
    assert presets["bses"].slabs is not None


def test_preset_matches_slab_grammar():
    tariff = get_preset("bses", DEFAULT_PRESETS_PATH)
    assert tariff.slabs == parse_slabs("0-200:3.0,200-400:4.5,400-800:6.5,800-1000000:8.0")


def test_get_preset_returns_independent_copy():
    first = get_preset("bses", DEFAULT_PRESETS_PATH)
    first.fixed = 999.0
    assert get_preset("bses", DEFAULT_PRESETS_PATH).fixed != 999.0


def test_get_preset_unknown_raises():
    with pytest.raises(KeyError):
        get_preset("does-not-exist", DEFAULT_PRESETS_PATH)


def _tariff_args(**overrides) -> argparse.Namespace:
    base = {
        "preset": None,
        "presets_file": DEFAULT_PRESETS_PATH,
        "slab": None,
        "rate": 7.0,
        "fixed": None,
        "tax_pct": None,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def test_build_tariff_flat_default():
    tariff = build_tariff(_tariff_args())
    assert tariff.slabs is None
    assert tariff.rate == 7.0


def test_build_tariff_from_preset():
    tariff = build_tariff(_tariff_args(preset="bses"))
    assert tariff.slabs is not None
    assert tariff.tax_pct == 5.0


def test_build_tariff_slab_overrides_preset():
    tariff = build_tariff(_tariff_args(preset="bses", slab="0-100:1.0,100-100000:2.0"))
    assert tariff.slabs == [(100.0, 1.0), (100000.0, 2.0)]


def test_build_tariff_explicit_overrides_preset_extras():
    tariff = build_tariff(_tariff_args(preset="bses", tax_pct=0.0, fixed=25.0))
    assert tariff.tax_pct == 0.0
    assert tariff.fixed == 25.0
