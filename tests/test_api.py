"""Tests for the FastAPI service (api.py) using an in-memory checkpoint.

A tiny GRU + scaler + calibration are written to a temp dir so the service can
be exercised end-to-end without the real artifacts.
"""

from __future__ import annotations

import json

import joblib
import numpy as np
import pytest
import torch
from sklearn.preprocessing import StandardScaler

from models import get_model

# The service layer needs the optional `serve` extra; skip cleanly without it.
pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

import api  # noqa: E402

INPUT_LEN = 14
HORIZON = 7


@pytest.fixture(scope="module")
def artifacts(tmp_path_factory):
    d = tmp_path_factory.mktemp("api_artifacts")

    torch.manual_seed(0)
    model_config = {
        "input_size": 1,
        "hidden1": 4,
        "hidden2": 4,
        "dropout": 0.0,
        "dense": 4,
        "output_size": HORIZON,
    }
    model = get_model("gru", **model_config)
    ckpt = {
        "model_type": "gru",
        "model_config": model_config,
        "state_dict": model.state_dict(),
        "input_len": INPUT_LEN,
        "horizon": HORIZON,
        "feature_config": None,
        "feature_columns": [],
        "feature_scaler_path": None,
    }
    model_path = d / "best_model.pt"
    torch.save(ckpt, model_path)

    scaler = StandardScaler().fit(np.linspace(1.0, 10.0, 100).reshape(-1, 1))
    scaler_path = d / "scaler.joblib"
    joblib.dump(scaler, scaler_path)

    calib = {
        "method": "power",
        "scale": "power",
        "gamma": 0.5,
        "levels": {"0.80": [0.3] * HORIZON, "0.90": [0.5] * HORIZON},
    }
    calib_path = d / "calibration.json"
    calib_path.write_text(json.dumps(calib))

    return model_path, scaler_path, calib_path


@pytest.fixture(scope="module")
def client(artifacts):
    model_path, scaler_path, calib_path = artifacts
    app = api.create_app(
        model_path=str(model_path),
        scaler_path=str(scaler_path),
        calibration_path=str(calib_path),
        device="cpu",
    )
    with TestClient(app) as c:
        yield c


def _history(n: int = INPUT_LEN) -> list[float]:
    return [3.0 + 0.5 * np.sin(i / 3.0) for i in range(n)]


def test_index(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "/forecast" in resp.json()["endpoints"]


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model"] == "gru"
    assert body["input_len"] == INPUT_LEN
    assert body["horizon"] == HORIZON
    assert body["calibration"] is True


def test_forecast_shape_and_intervals(client):
    resp = client.post("/forecast", json={"history_kwh": _history()})
    assert resp.status_code == 200
    body = resp.json()
    assert body["horizon"] == HORIZON
    assert len(body["forecast"]) == HORIZON
    assert [d["day"] for d in body["forecast"]] == list(range(1, HORIZON + 1))
    assert all(d["kwh"] >= 0 for d in body["forecast"])
    assert set(body["intervals"]) == {"80", "90"}
    assert body["forecast"][0]["lower_80"] <= body["forecast"][0]["upper_80"]
    assert body["total_kwh"] == pytest.approx(sum(d["kwh"] for d in body["forecast"]), abs=1e-3)


def test_forecast_custom_horizon(client):
    body = client.post("/forecast", json={"history_kwh": _history(), "horizon": 21}).json()
    assert body["horizon"] == 21
    assert len(body["forecast"]) == 21


def test_forecast_rejects_short_history(client):
    resp = client.post("/forecast", json={"history_kwh": [1.0, 2.0, 3.0]})
    assert resp.status_code == 422


def test_forecast_rejects_nonfinite(client):
    # Send raw JSON: NaN is not encodable by httpx's strict encoder but stdlib
    # json.loads accepts it, so this reaches the field validator.
    resp = client.post(
        "/forecast",
        content=b'{"history_kwh": [1.0, NaN, 3.0]}',
        headers={"content-type": "application/json"},
    )
    assert resp.status_code == 422


def test_bill_flat_tariff_math(client):
    body = client.post("/bill", json={"history_kwh": _history(), "rate": 7.0}).json()
    assert body["horizon"] == HORIZON
    assert len(body["daily"]) == HORIZON
    # Flat tariff: energy charge is exactly total kWh * rate.
    assert body["energy_charge_inr"] == pytest.approx(body["total_kwh"] * 7.0, abs=0.05)
    assert body["estimated_bill_inr"] == pytest.approx(
        body["energy_charge_inr"] + body["tax_inr"] + body["fixed_charge_inr"], abs=0.01
    )
    assert "80" in body["intervals"]
    assert body["intervals"]["80"]["lower_inr"] <= body["intervals"]["80"]["upper_inr"]


def test_bill_tax_and_fixed(client):
    body = client.post(
        "/bill",
        json={"history_kwh": _history(), "rate": 5.0, "tax_pct": 10.0, "fixed": 50.0},
    ).json()
    assert body["tax_inr"] == pytest.approx(body["energy_charge_inr"] * 0.10, abs=0.01)
    assert body["fixed_charge_inr"] == pytest.approx(50.0)


def test_bill_slab_tariff(client):
    body = client.post(
        "/bill",
        json={"history_kwh": _history(), "slab": "0-100:3.5,100-200:4.5,200-100000:7"},
    ).json()
    assert "slab" in body["tariff"]


def test_bill_bad_slab_is_422(client):
    resp = client.post("/bill", json={"history_kwh": _history(), "slab": "0-100"})
    assert resp.status_code == 422


def test_bill_unknown_preset_is_422(client):
    resp = client.post("/bill", json={"history_kwh": _history(), "preset": "does-not-exist"})
    assert resp.status_code == 422


def test_projection(client):
    body = client.post(
        "/projection", json={"history_kwh": _history(), "project_days": 30, "rate": 6.0}
    ).json()
    assert body["project_days"] == 30
    assert len(body["daily"]) == 30
    assert body["monthly_30d"]["days"] == 30
    assert body["three_month_90d"]["days"] == 30
    assert body["daily"][-1]["cumulative_cost_inr"] > body["daily"][0]["cumulative_cost_inr"]
