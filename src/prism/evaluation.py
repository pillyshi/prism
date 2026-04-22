from __future__ import annotations

from typing import Callable

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import get_scorer
from sklearn.model_selection import KFold, StratifiedKFold

from .models import FeatureMatrix
from .text_synthesis import _sample_feature_vectors

FeatureAugmentor = Callable[[np.ndarray, np.ndarray, int], tuple[np.ndarray, np.ndarray]]


def make_feature_augmentor(n: int = 1, seed: int | None = None) -> FeatureAugmentor:
    """Return an augmentor that samples synthetic feature vectors per fold.

    The returned callable fits a multivariate Gaussian to X_train, draws
    n_aug samples clipped to [0, 1], and samples labels from y_train with
    replacement to preserve the empirical class distribution.

    .. note::
        This is an **experimental** utility.

    Args:
        n: Default number of synthetic vectors (overridden by n_aug at call time).
        seed: Optional integer seed for reproducibility.

    Returns:
        A callable ``(X_train, y_train, n_aug) -> (X_aug, y_aug)``.
    """
    rng = np.random.default_rng(seed)

    def augmentor(
        X_train: np.ndarray,
        y_train: np.ndarray,
        n_aug: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        X_aug = _sample_feature_vectors(X_train, n=n_aug, rng=rng)
        idx = rng.integers(0, len(y_train), size=n_aug)
        y_aug = y_train[idx]
        return X_aug, y_aug

    return augmentor


def cross_val_score(
    estimator: BaseEstimator,
    matrix: FeatureMatrix,
    cv: int = 5,
    scoring: str | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """Cross-validate an estimator on a FeatureMatrix without augmentation.

    Args:
        estimator: An unfitted sklearn estimator.
        matrix: FeatureMatrix whose X and y are used directly.
        cv: Number of folds.
        scoring: sklearn scoring string. Defaults to ``"f1"`` for classification
            and ``"neg_mean_squared_error"`` for regression.
        seed: Random state for the fold splitter.

    Returns:
        Per-fold scores, shape ``(cv,)``.
    """
    if scoring is None:
        scoring = "f1" if matrix.mode == "classification" else "neg_mean_squared_error"
    splitter = _make_splitter(matrix.mode, cv, seed)
    scorer = get_scorer(scoring)
    X, y = matrix.X, matrix.y
    scores = []
    for train_idx, val_idx in splitter.split(X, y):
        est = clone(estimator)
        est.fit(X[train_idx], y[train_idx])
        scores.append(scorer(est, X[val_idx], y[val_idx]))
    return np.array(scores)


def cross_val_score_with_augmentation(
    estimator: BaseEstimator,
    matrix: FeatureMatrix,
    augmentor: FeatureAugmentor,
    n_aug: int = 1,
    cv: int = 5,
    scoring: str | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """Cross-validate an estimator with per-fold feature-vector augmentation.

    For each fold:

    1. Split into ``(X_train, y_train)`` and ``(X_val, y_val)``.
    2. Call ``augmentor(X_train, y_train, n_aug)`` to obtain synthetic rows.
    3. Concatenate: ``X_fit = [X_train; X_aug]``, ``y_fit = [y_train; y_aug]``.
    4. Clone and fit estimator on ``X_fit`` / ``y_fit``.
    5. Score on ``X_val`` / ``y_val`` (original data only — no leakage).

    .. warning::
        This is an **experimental** feature.

    Args:
        estimator: An unfitted sklearn estimator. Cloned fresh each fold.
        matrix: FeatureMatrix providing X, y, and mode.
        augmentor: Callable ``(X_train, y_train, n_aug) -> (X_aug, y_aug)``.
            Use :func:`make_feature_augmentor` to construct a standard one.
        n_aug: Number of synthetic rows to inject per fold.
        cv: Number of folds.
        scoring: sklearn scoring string. Defaults to ``"f1"`` for classification
            and ``"neg_mean_squared_error"`` for regression.
        seed: Random state for the fold splitter (independent of augmentor RNG).

    Returns:
        Per-fold scores, shape ``(cv,)``.
    """
    if scoring is None:
        scoring = "f1" if matrix.mode == "classification" else "neg_mean_squared_error"
    splitter = _make_splitter(matrix.mode, cv, seed)
    scorer = get_scorer(scoring)
    X, y = matrix.X, matrix.y
    scores = []
    for train_idx, val_idx in splitter.split(X, y):
        X_train, y_train = X[train_idx], y[train_idx]
        X_aug, y_aug = augmentor(X_train, y_train, n_aug)
        X_fit = np.concatenate([X_train, X_aug], axis=0)
        y_fit = np.concatenate([y_train, y_aug], axis=0)
        est = clone(estimator)
        est.fit(X_fit, y_fit)
        scores.append(scorer(est, X[val_idx], y[val_idx]))
    return np.array(scores)


def _make_splitter(
    mode: str, cv: int, seed: int | None
) -> StratifiedKFold | KFold:
    if mode == "classification":
        return StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
    return KFold(n_splits=cv, shuffle=True, random_state=seed)
