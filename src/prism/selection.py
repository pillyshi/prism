from __future__ import annotations

import numpy as np
from sklearn.linear_model import Lasso

from .models import Feature, FeatureDependency

_COEF_THRESHOLD = 1e-4


class FeatureSelector:
    """Analyzes feature dependencies and removes redundant features.

    For each feature i, fits a Lasso model predicting X[:,i] from all other
    features. A high R² indicates the feature is redundant (well-explained by others).

    Usage::

        selector = FeatureSelector().fit(X, features)
        print(selector.dependencies_)          # inspection
        X2, features2 = selector.transform(X, features)
    """

    def __init__(self, r2_threshold: float = 0.9) -> None:
        self.r2_threshold = r2_threshold
        self.dependencies_: list[FeatureDependency] = []

    def fit(self, X: np.ndarray, features: list[Feature]) -> FeatureSelector:
        """Fit dependency models for each feature.

        Args:
            X: Feature matrix of shape (n_texts, n_features).
            features: Features corresponding to columns of X.

        Returns:
            self, for chaining.
        """
        n_features = X.shape[1]
        self.dependencies_ = []
        for i in range(n_features):
            if n_features == 1:
                self.dependencies_.append(
                    FeatureDependency(feature=features[i], r2=0.0, predictors=[], coef=[])
                )
                continue
            others = [j for j in range(n_features) if j != i]
            X_others = X[:, others]
            y_i = X[:, i]
            model = Lasso(alpha=0.01, max_iter=5000)
            model.fit(X_others, y_i)
            r2 = float(model.score(X_others, y_i))
            predictors = [features[j] for j, c in zip(others, model.coef_) if abs(c) > _COEF_THRESHOLD]
            coef = [float(c) for c in model.coef_ if abs(c) > _COEF_THRESHOLD]
            self.dependencies_.append(
                FeatureDependency(feature=features[i], r2=r2, predictors=predictors, coef=coef)
            )
        return self

    def transform(
        self, X: np.ndarray, features: list[Feature]
    ) -> tuple[np.ndarray, list[Feature]]:
        """Return X and features with redundant features removed.

        A feature is considered redundant if its R² (from fit()) exceeds
        r2_threshold, meaning it can be well-predicted from other features.

        Args:
            X: Feature matrix of shape (n_texts, n_features).
            features: Features corresponding to columns of X.

        Returns:
            Tuple of (X_reduced, features_reduced).
        """
        keep = [i for i, dep in enumerate(self.dependencies_) if dep.r2 <= self.r2_threshold]
        if not keep:
            return np.empty((X.shape[0], 0)), []
        return X[:, keep], [features[i] for i in keep]
