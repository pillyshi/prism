from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FitEvaluation:
    wasserstein: np.ndarray  # (n_features,) per-feature W1 distance
    mean_diff: np.ndarray    # (n_features,) sampled_mean - orig_mean (signed)
    std_diff: np.ndarray     # (n_features,) sampled_std - orig_std (signed)


@dataclass
class GenerationEvaluation:
    wasserstein: np.ndarray  # (n_features,)
    mae: np.ndarray          # (n_features,) mean |X_sampled - X_scored|


def _wasserstein1d(a: np.ndarray, b: np.ndarray) -> float:
    n = max(len(a), len(b))
    q = np.linspace(0.0, 1.0, n)
    return float(np.mean(np.abs(np.quantile(a, q) - np.quantile(b, q))))


def evaluate_fit(X_orig: np.ndarray, X_sampled: np.ndarray) -> FitEvaluation:
    """Evaluate how well the fitted Gaussian reproduces the original distribution.

    Args:
        X_orig: Feature matrix of original texts, shape (n_orig, n_features).
        X_sampled: Feature vectors sampled from the fitted Gaussian, shape (n_sampled, n_features).

    Returns:
        FitEvaluation with per-feature metrics.
    """
    n_features = X_orig.shape[1]
    wasserstein = np.array(
        [_wasserstein1d(X_orig[:, i], X_sampled[:, i]) for i in range(n_features)]
    )
    mean_diff = X_sampled.mean(axis=0) - X_orig.mean(axis=0)
    std_diff = X_sampled.std(axis=0) - X_orig.std(axis=0)
    return FitEvaluation(wasserstein=wasserstein, mean_diff=mean_diff, std_diff=std_diff)


def evaluate_generation(X_sampled: np.ndarray, X_scored: np.ndarray) -> GenerationEvaluation:
    """Evaluate how faithfully the LLM reproduced the target feature vectors.

    Args:
        X_sampled: Feature vectors passed to the LLM, shape (n, n_features).
        X_scored: NLI scores of generated texts, shape (n, n_features).

    Returns:
        GenerationEvaluation with per-feature metrics.

    Raises:
        ValueError: If X_sampled and X_scored have different numbers of rows.
    """
    if X_sampled.shape[0] != X_scored.shape[0]:
        raise ValueError(
            f"X_sampled and X_scored must have the same number of rows, "
            f"got {X_sampled.shape[0]} and {X_scored.shape[0]}"
        )
    n_features = X_sampled.shape[1]
    wasserstein = np.array(
        [_wasserstein1d(X_sampled[:, i], X_scored[:, i]) for i in range(n_features)]
    )
    mae = np.mean(np.abs(X_sampled - X_scored), axis=0)
    return GenerationEvaluation(wasserstein=wasserstein, mae=mae)
