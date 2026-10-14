"""Tests for split-conformal prediction intervals (calibration.py)."""

from __future__ import annotations

import numpy as np
import pytest

from calibration import (
    DEFAULT_METHOD,
    apply,
    conformal_quantile,
    coverage_report,
    empirical_coverage,
    fit,
    load,
    save,
    widths,
)


def test_conformal_quantile_is_conservative_order_statistic():
    resid = np.arange(1, 11, dtype=float)  # 1..10, n=10
    # ceil((10 + 1) * 0.8) = 9 -> 9th smallest = 9.0
    assert conformal_quantile(resid, 0.80) == 9.0
    # ceil(11 * 0.9) = 10 -> 10.0
    assert conformal_quantile(resid, 0.90) == 10.0
    # level low enough that k=1 -> smallest
    assert conformal_quantile(resid, 0.05) == 1.0


def test_conformal_quantile_clamps_and_validates():
    assert conformal_quantile([5.0], 0.99) == 5.0  # k clamped to n
    with pytest.raises(ValueError):
        conformal_quantile([], 0.9)
    with pytest.raises(ValueError):
        conformal_quantile([1.0, 2.0], 1.5)


def test_fit_shape_and_symmetry():
    rng = np.random.default_rng(0)
    y_true = 20.0 + rng.normal(0.0, 1.0, size=(200, 7))
    y_pred = np.full((200, 7), 20.0)
    calib = fit(y_true, y_pred, levels=(0.8, 0.9), scale="absolute")
    assert calib["horizon"] == 7
    assert calib["n_cal"] == 200
    assert calib["scale"] == "absolute"
    assert set(calib["levels"]) == {"0.80", "0.90"}
    assert len(calib["levels"]["0.80"]) == 7
    # wider nominal level -> wider (or equal) interval per step
    assert all(
        w90 >= w80
        for w80, w90 in zip(calib["levels"]["0.80"], calib["levels"]["0.90"], strict=True)
    )


def test_fit_records_scale_and_gamma():
    y_true = np.full((50, 3), 12.0)
    y_pred = np.full((50, 3), 10.0)
    calib = fit(y_true, y_pred, scale="power", gamma=0.5)
    assert calib["scale"] == "power"
    assert calib["gamma"] == 0.5
    assert calib["method"] == DEFAULT_METHOD


def test_fit_validates_inputs():
    with pytest.raises(ValueError):
        fit(np.zeros((0, 4)), np.zeros((0, 4)))
    with pytest.raises(ValueError):
        fit(np.zeros((5, 4)), np.zeros((5, 3)))  # shape mismatch
    with pytest.raises(ValueError):
        fit(np.zeros((5, 4)), np.zeros((5, 4)), scale="bogus")


def test_widths_extrapolate_beyond_horizon():
    calib = fit(np.full((100, 4), 2.0), np.zeros((100, 4)), levels=(0.9,), scale="absolute")
    w = widths(calib, 0.9, 8)
    assert list(w[:4]) == [2.0, 2.0, 2.0, 2.0]
    # beyond H the width grows as sqrt(step / H)
    assert w[4] == pytest.approx(2.0 * np.sqrt(5 / 4))
    assert w[7] > w[4]


def test_apply_broadcasts_2d_and_clips_lower():
    calib = fit(np.full((100, 3), 1.5), np.zeros((100, 3)), levels=(0.8,), scale="absolute")
    pred = np.array([[0.5, 10.0, 20.0], [1.0, 11.0, 21.0]])
    lower, upper = apply(pred, calib, 0.8)
    assert lower.shape == pred.shape
    assert (lower >= 0.0).all()
    assert lower[0, 0] == 0.0  # 0.5 - 1.5 clipped to 0
    np.testing.assert_allclose(upper, pred + 1.5)


def test_apply_power_scale_grows_with_level():
    # constant normalized score 0.1 at gamma=1 -> half-width = 0.1 * pred
    y_true = np.full((100, 3), 11.0)
    y_pred = np.full((100, 3), 10.0)
    calib = fit(y_true, y_pred, levels=(0.8,), scale="power", gamma=1.0)
    lower, upper = apply(np.array([[10.0, 20.0, 5.0]]), calib, 0.8)
    np.testing.assert_allclose(upper[0], [11.0, 22.0, 5.5])
    np.testing.assert_allclose(lower[0], [9.0, 18.0, 4.5])


def test_apply_1d_projection():
    calib = fit(np.full((50, 7), 2.0), np.zeros((50, 7)), levels=(0.9,), scale="absolute")
    lower, upper = apply(np.full(10, 5.0), calib, 0.9)
    assert lower.shape == (10,)
    np.testing.assert_allclose(lower[:7], 3.0)
    assert upper[9] > 7.0  # extrapolated curve widens


def test_coverage_report_near_nominal_on_gaussian_noise():
    rng = np.random.default_rng(1)
    n, horizon = 2000, 7
    base = 20.0  # a realistic positive consumption level, away from the 0 clip
    y_true = base + rng.normal(0.0, 1.0, size=(n, horizon))
    y_pred = np.full((n, horizon), base)
    y_cal = base + rng.normal(0.0, 1.0, size=(500, horizon))
    calib = fit(y_cal, np.full((500, horizon), base), levels=(0.8, 0.9), scale="absolute")
    report = coverage_report(calib, y_true, y_pred)
    assert abs(report["0.80"]["empirical"] - 0.80) < 0.05
    assert abs(report["0.90"]["empirical"] - 0.90) < 0.05


def test_normalized_conformal_covers_heteroscedastic_noise():
    # noise std grows as sqrt(level): power scale with gamma=0.5 stabilises it.
    rng = np.random.default_rng(7)
    n, horizon = 3000, 4
    level_cal = rng.uniform(2.0, 40.0, size=(n, horizon))
    y_cal = level_cal + rng.normal(0.0, 1.0, size=(n, horizon)) * np.sqrt(level_cal)
    level_eval = rng.uniform(2.0, 40.0, size=(n, horizon))
    y_eval = level_eval + rng.normal(0.0, 1.0, size=(n, horizon)) * np.sqrt(level_eval)
    calib = fit(y_cal, level_cal, levels=(0.8, 0.9), scale="power", gamma=0.5)
    report = coverage_report(calib, y_eval, level_eval)
    assert abs(report["0.80"]["empirical"] - 0.80) < 0.04
    assert abs(report["0.90"]["empirical"] - 0.90) < 0.03


def test_empirical_coverage_counts_edges_as_inside():
    y = np.array([1.0, 2.0, 3.0])
    lower = np.array([1.0, 0.0, 4.0])
    upper = np.array([1.0, 2.0, 5.0])
    assert empirical_coverage(y, lower, upper) == pytest.approx(2 / 3)


def test_save_load_round_trip(tmp_path):
    calib = fit(np.full((30, 5), 3.0), np.zeros((30, 5)), scale="absolute")
    path = save(calib, tmp_path / "cal.json")
    assert path.exists()
    assert load(path) == calib
