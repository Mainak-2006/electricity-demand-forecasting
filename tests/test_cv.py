"""Tests for the rolling-origin cross-validation splitter."""

from __future__ import annotations

import pytest

from cv import Fold, rolling_origin_splits


def test_default_folds_cover_series_end():
    folds = rolling_origin_splits(100, n_folds=5)
    assert len(folds) == 5
    assert folds[-1].test_end == 100
    # Folds are ordered and never overlap with a later train prefix.
    for a, b in zip(folds, folds[1:], strict=False):
        assert a.test_end <= b.test_start
        assert b.train_end == b.test_start


def test_expanding_train_grows():
    folds = rolling_origin_splits(120, n_folds=4, test_size=10, min_train_size=60)
    assert [f.train_start for f in folds] == [0, 0, 0, 0]
    assert [f.train_end for f in folds] == sorted(f.train_end for f in folds)
    assert folds[0].train_size < folds[-1].train_size


def test_sliding_window_keeps_fixed_train_size():
    folds = rolling_origin_splits(
        120, n_folds=4, test_size=10, min_train_size=60, expanding=False, train_window=30
    )
    assert {f.train_size for f in folds} == {30}
    assert folds[0].test_start == 60


def test_test_windows_are_contiguous_when_step_equals_test_size():
    folds = rolling_origin_splits(100, n_folds=4, test_size=20, min_train_size=20)
    assert [f.test_start for f in folds] == [20, 40, 60, 80]
    assert [f.test_end for f in folds] == [40, 60, 80, 100]


def test_min_train_size_pins_first_origin():
    folds = rolling_origin_splits(200, n_folds=5, test_size=10, min_train_size=150)
    assert folds[0].test_start == 150
    assert all(f.train_end == f.test_start for f in folds)


def test_explicit_test_size_and_step():
    folds = rolling_origin_splits(50, n_folds=3, test_size=5, min_train_size=10, step=10)
    assert [(f.test_start, f.test_end) for f in folds] == [(10, 15), (20, 25), (30, 35)]


def test_single_fold():
    folds = rolling_origin_splits(40, n_folds=1, test_size=10, min_train_size=30)
    assert len(folds) == 1
    assert folds[0].test_start == 30
    assert folds[0].test_end == 40


def test_fold_dataclass_properties():
    fold = Fold(index=0, train_start=0, train_end=30, test_start=30, test_end=37)
    assert fold.train_size == 30
    assert fold.test_size == 7


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_folds": 0},
        {"n_folds": 5, "test_size": 0},
        {"n_folds": 5, "test_size": 100, "min_train_size": 50},
        {"n_folds": 5, "test_size": 10, "min_train_size": 0},
    ],
)
def test_invalid_parameters_raise(kwargs):
    with pytest.raises(ValueError):
        rolling_origin_splits(100, **kwargs)


def test_too_short_series_raises():
    with pytest.raises(ValueError):
        rolling_origin_splits(2)
