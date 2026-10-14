"""Tests for device selection, iAWE file detection and rolling forecast."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from sklearn.preprocessing import StandardScaler

from iawe_ingest import _is_main_meter
from inference import resolve_device
from predict import rolling_forecast


def test_resolve_device_cpu():
    assert resolve_device("cpu").type == "cpu"


def test_resolve_device_auto():
    assert resolve_device("auto").type in {"cpu", "cuda"}


def test_resolve_device_unknown():
    with pytest.raises(SystemExit):
        resolve_device("tpu")


@pytest.mark.parametrize("name", ["main_meter_trace.csv", "aggregate_trace.csv"])
def test_ac_substring_does_not_hide_main_meter(name):
    # Regression: substring "ac" inside "trace" used to flag these as appliances.
    assert _is_main_meter(name) is True


def test_plain_unknown_name_is_not_main_meter():
    assert _is_main_meter("trace.csv") is False


def test_main_meter_detection():
    assert _is_main_meter("main_meter.csv") is True
    assert _is_main_meter("aggregate.csv") is True
    assert _is_main_meter("1.csv") is True
    assert _is_main_meter("fridge.csv") is False
    assert _is_main_meter("air conditioner.csv") is False


def test_rolling_forecast_length_and_nonneg():
    class Identity(torch.nn.Module):
        def forward(self, x):
            return x[:, -1, 0:1].repeat(1, 7)

    rng = np.random.default_rng(0)
    data = 3.0 + rng.normal(0, 0.3, 60)
    scaler = StandardScaler().fit(data.reshape(-1, 1))
    out = rolling_forecast(
        Identity(),
        scaler,
        data,
        input_len=30,
        horizon=7,
        total_days=45,
        device=torch.device("cpu"),
    )
    assert len(out) == 45
    assert all(v >= 0.0 for v in out)
