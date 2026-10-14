"""BijliPredict AI - rolling-origin (walk-forward) cross-validation.

A rolling-origin splitter sweeps a fixed-size forecast window forward through a
time series. Each fold pairs a training prefix with the test window that
immediately follows it, so a fold never tests on observations its training
prefix already contained.

Folds are described by integer positions into the (cleaned) series, so they can
be applied to an array, a frame or a set of sliding-window indices without this
module knowing anything else.

Public API:
    Fold                        -> a single train/test position range
    rolling_origin_splits(...)  -> list[Fold] over a series of length n
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Fold:
    """Position ranges for one rolling-origin fold.

    All bounds are half-open ``[start, end)`` integer indices into a series of
    length ``n``. ``train`` is the fitting prefix; ``test`` is the evaluation
    window that follows it. With ``expanding=True`` ``train_start`` is always 0.
    """

    index: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int

    @property
    def train_size(self) -> int:
        return self.train_end - self.train_start

    @property
    def test_size(self) -> int:
        return self.test_end - self.test_start


def rolling_origin_splits(
    n: int,
    n_folds: int = 5,
    test_size: int | None = None,
    min_train_size: int | None = None,
    step: int | None = None,
    expanding: bool = True,
    train_window: int | None = None,
) -> list[Fold]:
    """Build ``n_folds`` rolling-origin folds over a series of length ``n``.

    Args:
        n: number of observations (days) in the series.
        n_folds: how many test windows to evaluate.
        test_size: length of each test window. Defaults to ``n // (n_folds+1)``
            of the whole series, or ``available // n_folds`` when
            `min_train_size` is pinned, so an initial train prefix remains.
        min_train_size: index where the *first* fold's test window starts (its
            training prefix is ``[0, min_train_size)``). Defaults to the largest
            value that still leaves room for all folds.
        step: how far the origin advances between folds. Defaults to an even
            spread so the final fold ends exactly at ``n`` (contiguous,
            non-overlapping blocks when ``step == test_size``).
        expanding: ``True`` grows the training prefix every fold; ``False`` uses
            a fixed ``train_window`` (defaults to ``min_train_size``) that slides.
        train_window: fixed training length when ``expanding`` is ``False``.

    Raises:
        ValueError: if the parameters cannot yield ``n_folds`` valid folds.
    """
    if n < 3:
        raise ValueError(f"Need at least 3 observations, got {n}.")
    if n_folds < 1:
        raise ValueError(f"n_folds must be >= 1, got {n_folds}.")

    if min_train_size is None:
        default_test = max(1, n // (n_folds + 1))
        min_train_size = max(default_test, n - n_folds * default_test)
    if min_train_size < 1:
        raise ValueError(f"min_train_size must be >= 1, got {min_train_size}.")

    available = n - min_train_size
    if test_size is None:
        test_size = max(1, available // n_folds)
    if test_size < 1:
        raise ValueError(f"test_size must be >= 1, got {test_size}.")

    if available < test_size:
        raise ValueError(
            f"Not enough data: {n} observations leave {available} after a "
            f"min_train_size of {min_train_size}, but each fold needs {test_size}."
        )

    if step is None:
        step = 0 if n_folds == 1 else max(1, (available - test_size) // (n_folds - 1))
    if step < 0:
        raise ValueError(f"step must be >= 0, got {step}.")

    window = train_window if train_window is not None else min_train_size
    folds: list[Fold] = []
    for i in range(n_folds):
        test_start = min_train_size + i * step
        test_end = test_start + test_size
        if test_end > n:
            raise ValueError(
                f"Fold {i} test window [{test_start}, {test_end}) exceeds series "
                f"length {n}; reduce n_folds/step/test_size."
            )
        train_end = test_start
        train_start = 0 if expanding else max(0, train_end - window)
        folds.append(
            Fold(
                index=i,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
            )
        )
    return folds


if __name__ == "__main__":  # tiny smoke test
    for f in rolling_origin_splits(100, n_folds=5, test_size=10):
        print(f)
