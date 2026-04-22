from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import StratifiedKFold

from prism.evaluation import (
    cross_val_score,
    cross_val_score_with_augmentation,
    make_feature_augmentor,
)
from prism.models import Axis, Feature, FeatureMatrix


def _make_matrix_clf(n_texts: int = 40, n_features: int = 4, seed: int = 0) -> FeatureMatrix:
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 1, (n_texts, n_features))
    y = np.array([1.0] * (n_texts // 2) + [-1.0] * (n_texts // 2))
    axis = Axis(hypothesis="Test axis.")
    features = [Feature(hypothesis=f"F{i}.", axis=axis) for i in range(n_features)]
    return FeatureMatrix(axis=axis, features=features, X=X, y=y, mode="classification")


def _make_matrix_reg(n_texts: int = 40, n_features: int = 4, seed: int = 0) -> FeatureMatrix:
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 1, (n_texts, n_features))
    y = rng.uniform(0, 1, n_texts)
    axis = Axis(hypothesis="Regression axis.")
    features = [Feature(hypothesis=f"F{i}.", axis=axis) for i in range(n_features)]
    return FeatureMatrix(axis=axis, features=features, X=X, y=y, mode="regression")


# --- make_feature_augmentor ---

def test_augmentor_output_shape() -> None:
    matrix = _make_matrix_clf()
    augmentor = make_feature_augmentor(seed=0)
    X_aug, y_aug = augmentor(matrix.X, matrix.y, 5)
    assert X_aug.shape == (5, matrix.X.shape[1])
    assert y_aug.shape == (5,)


def test_augmentor_x_clipped() -> None:
    matrix = _make_matrix_clf()
    augmentor = make_feature_augmentor(seed=0)
    X_aug, _ = augmentor(matrix.X, matrix.y, 20)
    assert X_aug.min() >= 0.0
    assert X_aug.max() <= 1.0


def test_augmentor_y_from_training_set() -> None:
    matrix = _make_matrix_clf()
    augmentor = make_feature_augmentor(seed=0)
    _, y_aug = augmentor(matrix.X, matrix.y, 20)
    for label in y_aug:
        assert label in matrix.y


def test_augmentor_reproducible() -> None:
    matrix = _make_matrix_clf()
    a1 = make_feature_augmentor(seed=42)
    a2 = make_feature_augmentor(seed=42)
    X1, y1 = a1(matrix.X, matrix.y, 10)
    X2, y2 = a2(matrix.X, matrix.y, 10)
    np.testing.assert_array_equal(X1, X2)
    np.testing.assert_array_equal(y1, y2)


# --- cross_val_score ---

def test_cross_val_score_shape_clf() -> None:
    matrix = _make_matrix_clf()
    scores = cross_val_score(LogisticRegression(), matrix, cv=5, seed=0)
    assert scores.shape == (5,)


def test_cross_val_score_shape_reg() -> None:
    matrix = _make_matrix_reg()
    scores = cross_val_score(Ridge(), matrix, cv=5, seed=0)
    assert scores.shape == (5,)


def test_cross_val_score_custom_scoring() -> None:
    matrix = _make_matrix_clf()
    scores = cross_val_score(LogisticRegression(), matrix, cv=3, scoring="accuracy", seed=0)
    assert scores.shape == (3,)
    assert all(0.0 <= s <= 1.0 for s in scores)


# --- cross_val_score_with_augmentation ---

def test_augmented_shape_clf() -> None:
    matrix = _make_matrix_clf()
    augmentor = make_feature_augmentor(seed=0)
    scores = cross_val_score_with_augmentation(
        LogisticRegression(), matrix, augmentor, n_aug=5, cv=5, seed=0
    )
    assert scores.shape == (5,)


def test_augmented_shape_reg() -> None:
    matrix = _make_matrix_reg()
    augmentor = make_feature_augmentor(seed=0)
    scores = cross_val_score_with_augmentation(
        Ridge(), matrix, augmentor, n_aug=5, cv=5, seed=0
    )
    assert scores.shape == (5,)


def test_augmented_no_data_leakage() -> None:
    matrix = _make_matrix_clf(n_texts=40, n_features=3, seed=1)
    seen_train: list[np.ndarray] = []

    def recording_augmentor(
        X_train: np.ndarray, y_train: np.ndarray, n: int
    ) -> tuple[np.ndarray, np.ndarray]:
        seen_train.append(X_train.copy())
        return np.zeros((n, X_train.shape[1])), np.zeros(n)

    cross_val_score_with_augmentation(
        LogisticRegression(), matrix, recording_augmentor, n_aug=2, cv=5, seed=0
    )
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    for fold_i, (_, val_idx) in enumerate(splitter.split(matrix.X, matrix.y)):
        for val_row in matrix.X[val_idx]:
            for train_row in seen_train[fold_i]:
                assert not np.allclose(val_row, train_row)


def test_augmented_estimator_cloned_per_fold() -> None:
    matrix = _make_matrix_clf()
    fit_calls: list[int] = [0]

    class CountingEstimator(LogisticRegression):
        def fit(self, X, y, **kw):  # type: ignore[override]
            fit_calls[0] += 1
            return super().fit(X, y, **kw)

    augmentor = make_feature_augmentor(seed=0)
    cross_val_score_with_augmentation(
        CountingEstimator(), matrix, augmentor, n_aug=2, cv=5, seed=0
    )
    assert fit_calls[0] == 5


def test_augmented_n_aug_passed_correctly() -> None:
    matrix = _make_matrix_clf()
    received_n: list[int] = []

    def recording_augmentor(
        X_train: np.ndarray, y_train: np.ndarray, n: int
    ) -> tuple[np.ndarray, np.ndarray]:
        received_n.append(n)
        return np.zeros((n, X_train.shape[1])), np.zeros(n)

    cross_val_score_with_augmentation(
        LogisticRegression(), matrix, recording_augmentor, n_aug=7, cv=4, seed=0
    )
    assert all(n == 7 for n in received_n)
    assert len(received_n) == 4
