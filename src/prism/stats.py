from __future__ import annotations

import numpy as np


def sample_feature_vectors(X: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Fit multivariate Gaussian to X rows and draw n samples, clipped to [0, 1]."""
    n_texts, n_features = X.shape
    if n_texts < 2:
        mean = X.mean(axis=0) if n_texts == 1 else np.full(n_features, 0.5)
        return np.clip(np.tile(mean, (n, 1)), 0.0, 1.0)
    mean = X.mean(axis=0)
    cov = np.atleast_2d(np.cov(X.T)) + np.eye(n_features) * 1e-6
    return np.clip(rng.multivariate_normal(mean=mean, cov=cov, size=n), 0.0, 1.0)
