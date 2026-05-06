"""Unit tests for evaluate_fit and evaluate_generation."""
import numpy as np
import pytest

from prism import (
    FitEvaluation,
    GenerationEvaluation,
    evaluate_fit,
    evaluate_generation,
)


# ---------------------------------------------------------------------------
# evaluate_fit — output shape
# ---------------------------------------------------------------------------

def test_evaluate_fit_output_shapes():
    rng = np.random.default_rng(0)
    X_orig = rng.uniform(0, 1, (20, 4))
    X_sampled = rng.uniform(0, 1, (15, 4))
    result = evaluate_fit(X_orig, X_sampled)
    assert isinstance(result, FitEvaluation)
    assert result.wasserstein.shape == (4,)
    assert result.mean_diff.shape == (4,)
    assert result.std_diff.shape == (4,)


def test_evaluate_fit_identical_returns_near_zero():
    X = np.random.default_rng(1).uniform(0, 1, (20, 3))
    result = evaluate_fit(X, X)
    np.testing.assert_allclose(result.wasserstein, np.zeros(3), atol=1e-10)
    np.testing.assert_allclose(result.mean_diff, np.zeros(3), atol=1e-10)
    np.testing.assert_allclose(result.std_diff, np.zeros(3), atol=1e-10)


def test_evaluate_fit_mean_diff_sign():
    X_orig = np.full((10, 2), 0.3)
    X_sampled = np.full((10, 2), 0.7)
    result = evaluate_fit(X_orig, X_sampled)
    assert np.all(result.mean_diff > 0)


def test_evaluate_fit_std_diff_sign():
    rng = np.random.default_rng(2)
    X_orig = np.full((20, 1), 0.5)
    X_sampled = rng.uniform(0, 1, (20, 1))
    result = evaluate_fit(X_orig, X_sampled)
    assert result.std_diff[0] > 0


def test_evaluate_fit_different_row_counts():
    rng = np.random.default_rng(3)
    X_orig = rng.uniform(0, 1, (30, 2))
    X_sampled = rng.uniform(0, 1, (10, 2))
    result = evaluate_fit(X_orig, X_sampled)
    assert result.wasserstein.shape == (2,)


def test_evaluate_fit_zero_features():
    X_orig = np.empty((10, 0))
    X_sampled = np.empty((10, 0))
    result = evaluate_fit(X_orig, X_sampled)
    assert result.wasserstein.shape == (0,)
    assert result.mean_diff.shape == (0,)
    assert result.std_diff.shape == (0,)


def test_evaluate_fit_wasserstein_non_negative():
    rng = np.random.default_rng(4)
    X_orig = rng.uniform(0, 1, (20, 5))
    X_sampled = rng.uniform(0, 1, (20, 5))
    result = evaluate_fit(X_orig, X_sampled)
    assert np.all(result.wasserstein >= 0)


# ---------------------------------------------------------------------------
# evaluate_generation — output shape and values
# ---------------------------------------------------------------------------

def test_evaluate_generation_output_shapes():
    rng = np.random.default_rng(5)
    X_sampled = rng.uniform(0, 1, (15, 3))
    X_scored = rng.uniform(0, 1, (15, 3))
    result = evaluate_generation(X_sampled, X_scored)
    assert isinstance(result, GenerationEvaluation)
    assert result.wasserstein.shape == (3,)
    assert result.mae.shape == (3,)


def test_evaluate_generation_identical_returns_near_zero():
    X = np.random.default_rng(6).uniform(0, 1, (20, 3))
    result = evaluate_generation(X, X)
    np.testing.assert_allclose(result.wasserstein, np.zeros(3), atol=1e-10)
    np.testing.assert_allclose(result.mae, np.zeros(3), atol=1e-10)


def test_evaluate_generation_mae_values():
    X_sampled = np.zeros((10, 2))
    X_scored = np.column_stack([np.full(10, 0.2), np.full(10, 0.5)])
    result = evaluate_generation(X_sampled, X_scored)
    np.testing.assert_allclose(result.mae[0], 0.2, atol=1e-10)
    np.testing.assert_allclose(result.mae[1], 0.5, atol=1e-10)


def test_evaluate_generation_row_mismatch_raises():
    X_sampled = np.zeros((10, 3))
    X_scored = np.zeros((8, 3))
    with pytest.raises(ValueError, match="same number of rows"):
        evaluate_generation(X_sampled, X_scored)


def test_evaluate_generation_wasserstein_non_negative():
    rng = np.random.default_rng(7)
    X_sampled = rng.uniform(0, 1, (15, 4))
    X_scored = rng.uniform(0, 1, (15, 4))
    result = evaluate_generation(X_sampled, X_scored)
    assert np.all(result.wasserstein >= 0)


def test_evaluate_generation_zero_features():
    X_sampled = np.empty((10, 0))
    X_scored = np.empty((10, 0))
    result = evaluate_generation(X_sampled, X_scored)
    assert result.wasserstein.shape == (0,)
    assert result.mae.shape == (0,)


