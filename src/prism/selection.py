from __future__ import annotations

import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler

from .models import FeatureMatrix, FittedPredictor, SelectionResult


class LassoSelector:
    """Selects predictive features per axis using Lasso regression."""

    def __init__(self, cv: int = 5, max_iter: int = 5000, coef_threshold: float = 1e-4) -> None:
        self.cv = cv
        self.max_iter = max_iter
        self.coef_threshold = coef_threshold

    def select(self, matrix: FeatureMatrix) -> tuple[SelectionResult, FittedPredictor]:
        """Fit LassoCV on the feature matrix and return the selected features and fitted predictor.

        Args:
            matrix: FeatureMatrix with X (n_texts, n_features) and y (n_texts,).

        Returns:
            Tuple of (SelectionResult, FittedPredictor).
        """
        X, y = matrix.X, matrix.y

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = LassoCV(cv=self.cv, max_iter=self.max_iter)
        model.fit(X_scaled, y)

        # Convert coefficients back to original (unscaled) space
        coef_original = model.coef_ / scaler.scale_

        mask = np.abs(coef_original) > self.coef_threshold
        selected_features = [f for f, m in zip(matrix.features, mask) if m]
        selected_coef = coef_original[mask].tolist()

        result = SelectionResult(
            axis=matrix.axis,
            selected_features=selected_features,
            coef=selected_coef,
        )
        predictor = FittedPredictor(axis=matrix.axis, scaler=scaler, model=model)
        return result, predictor
