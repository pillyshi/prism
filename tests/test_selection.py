"""Unit tests for FeatureSelector."""
import numpy as np
import pytest

from prism import Feature, FeatureDependency, FeatureSelector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_features(n: int) -> list[Feature]:
    return [Feature(hypothesis=f"Feature {i}.") for i in range(n)]


def _make_X(n_texts: int = 30, n_features: int = 4, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).uniform(0, 1, (n_texts, n_features))


# ---------------------------------------------------------------------------
# fit
# ---------------------------------------------------------------------------

def test_fit_returns_self():
    X = _make_X()
    features = _make_features(4)
    selector = FeatureSelector()
    result = selector.fit(X, features)
    assert result is selector


def test_fit_populates_dependencies():
    X = _make_X(n_features=4)
    features = _make_features(4)
    selector = FeatureSelector().fit(X, features)
    assert len(selector.dependencies_) == 4


def test_fit_dependency_features_match():
    X = _make_X(n_features=3)
    features = _make_features(3)
    selector = FeatureSelector().fit(X, features)
    for dep, feat in zip(selector.dependencies_, features):
        assert dep.feature == feat


def test_fit_r2_in_range():
    X = _make_X(n_features=4)
    features = _make_features(4)
    selector = FeatureSelector().fit(X, features)
    for dep in selector.dependencies_:
        assert 0.0 <= dep.r2 <= 1.0


def test_fit_redundant_feature_has_high_r2():
    rng = np.random.default_rng(0)
    X_base = rng.uniform(0, 1, (50, 2))
    # feature 2 = feature 0 + small noise → highly predictable
    X_redundant = X_base[:, [0]] + rng.normal(0, 0.01, (50, 1))
    X = np.hstack([X_base, X_redundant])
    features = _make_features(3)
    selector = FeatureSelector().fit(X, features)
    assert selector.dependencies_[2].r2 > 0.9


def test_fit_independent_feature_has_low_r2():
    rng = np.random.default_rng(1)
    # All features are independent uniform noise
    X = rng.uniform(0, 1, (50, 4))
    features = _make_features(4)
    selector = FeatureSelector().fit(X, features)
    for dep in selector.dependencies_:
        assert dep.r2 < 0.5


def test_fit_single_feature_r2_zero():
    X = _make_X(n_features=1)
    features = _make_features(1)
    selector = FeatureSelector().fit(X, features)
    assert len(selector.dependencies_) == 1
    assert selector.dependencies_[0].r2 == 0.0
    assert selector.dependencies_[0].predictors == []


def test_fit_predictors_are_subset_of_features():
    X = _make_X(n_features=4)
    features = _make_features(4)
    selector = FeatureSelector().fit(X, features)
    for dep in selector.dependencies_:
        for predictor in dep.predictors:
            assert predictor in features
        assert dep.feature not in dep.predictors


def test_fit_coef_length_matches_predictors():
    X = _make_X(n_features=4)
    features = _make_features(4)
    selector = FeatureSelector().fit(X, features)
    for dep in selector.dependencies_:
        assert len(dep.coef) == len(dep.predictors)


# ---------------------------------------------------------------------------
# transform
# ---------------------------------------------------------------------------

def test_transform_keeps_independent_features():
    rng = np.random.default_rng(0)
    X_base = rng.uniform(0, 1, (50, 2))
    X_redundant = X_base[:, [0]] + rng.normal(0, 0.01, (50, 1))
    X = np.hstack([X_base, X_redundant])
    features = _make_features(3)
    selector = FeatureSelector(r2_threshold=0.9).fit(X, features)
    X2, features2 = selector.transform(X, features)
    # feature 0 and feature 2 predict each other → both have high R², both removed
    # feature 1 is independent → must be kept
    assert features[1] in features2
    assert features[2] not in features2


def test_transform_returns_correct_columns():
    rng = np.random.default_rng(2)
    X = rng.uniform(0, 1, (50, 3))
    features = _make_features(3)
    selector = FeatureSelector(r2_threshold=0.9).fit(X, features)
    X2, features2 = selector.transform(X, features)
    keep = [i for i, dep in enumerate(selector.dependencies_) if dep.r2 <= 0.9]
    np.testing.assert_array_equal(X2, X[:, keep])
    assert features2 == [features[i] for i in keep]


def test_transform_high_threshold_keeps_all():
    X = _make_X(n_features=3)
    features = _make_features(3)
    selector = FeatureSelector(r2_threshold=1.0).fit(X, features)
    X2, features2 = selector.transform(X, features)
    assert X2.shape == X.shape
    assert features2 == features


def test_transform_low_threshold_removes_all():
    X = _make_X(n_features=3)
    features = _make_features(3)
    selector = FeatureSelector(r2_threshold=-1.0).fit(X, features)
    X2, features2 = selector.transform(X, features)
    assert X2.shape == (X.shape[0], 0)
    assert features2 == []


def test_transform_single_feature_always_kept():
    X = _make_X(n_features=1)
    features = _make_features(1)
    selector = FeatureSelector(r2_threshold=0.9).fit(X, features)
    X2, features2 = selector.transform(X, features)
    assert X2.shape == (X.shape[0], 1)
    assert features2 == features
