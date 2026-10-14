"""Tests for run manifests (manifest.py)."""

from __future__ import annotations

import json

import pytest

from manifest import (
    MANIFEST_VERSION,
    RunManifest,
    add_manifest_arguments,
    file_info,
    file_sha256,
    to_jsonable,
)


def test_file_sha256_and_info(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("date,consumption_kwh\n2023-01-01,3.0\n")
    digest = file_sha256(path)
    assert isinstance(digest, str) and len(digest) == 64
    info = file_info(path)
    assert info["exists"] is True
    assert info["sha256"] == digest
    assert info["bytes"] > 0


def test_file_info_missing_file(tmp_path):
    info = file_info(tmp_path / "nope.csv")
    assert info == {"path": str(tmp_path / "nope.csv"), "exists": False}
    assert file_sha256(tmp_path / "nope.csv") is None


def test_to_jsonable_nested_types(tmp_path):
    payload = to_jsonable({"path": tmp_path, "items": (1, 2), "flag": True, "none": None})
    assert payload["path"] == str(tmp_path)
    assert payload["items"] == [1, 2]
    assert payload["flag"] is True
    assert payload["none"] is None


def test_manifest_write_and_reload(tmp_path):
    data = tmp_path / "data.csv"
    data.write_text("a\n1\n")
    mf = RunManifest("train", args={"epochs": 3}, manifest_dir=tmp_path / "m")
    mf.add_input("data", data)
    mf.add_output("model", tmp_path / "model.pt")
    mf.set("device", "cpu")
    mf.record_duration("train", 1.234)
    path = mf.write()
    assert path is not None and path.exists()

    record = json.loads(path.read_text())
    assert record["manifest_version"] == MANIFEST_VERSION
    assert record["command"] == "train"
    assert record["status"] == "ok"
    assert record["args"]["epochs"] == 3
    assert record["inputs"]["data"]["exists"] is True
    assert record["outputs"]["model"].endswith("model.pt")
    assert record["durations"]["train"] == 1.234
    assert record["extra"]["device"] == "cpu"


def test_manifest_disabled_writes_nothing(tmp_path):
    mf = RunManifest("predict", manifest_dir=tmp_path / "m", enabled=False)
    assert mf.write() is None
    assert not (tmp_path / "m").exists()


def test_manifest_context_manager_records_error(tmp_path):
    with pytest.raises(ValueError):
        with RunManifest("evaluate", manifest_dir=tmp_path) as mf:
            raise ValueError("boom")
    record = json.loads(mf.path().read_text())
    assert record["status"] == "error"
    assert "ValueError" in record["error"]


def test_add_manifest_arguments_parses(tmp_path):
    import argparse

    parser = argparse.ArgumentParser()
    add_manifest_arguments(parser)
    args = parser.parse_args([])
    assert args.manifest_dir == "outputs/manifests"
    assert args.no_manifest is False
    args = parser.parse_args(["--no-manifest", "--manifest-dir", str(tmp_path)])
    assert args.no_manifest is True
    assert args.manifest_dir == str(tmp_path)
