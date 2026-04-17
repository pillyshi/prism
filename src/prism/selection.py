from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

from .models import FeatureMatrix, FittedPredictor, SelectionResult


_ALPHA_GRID = np.logspace(-4, 0, 20)


class FeatureSelector:
    """Selects predictive features per axis using L1-regularized SGD models.

    Classification mode: SGDClassifier(loss="log_loss", penalty="l1"), scoring="f1".
    Regression mode: SGDRegressor(penalty="l1"), scoring="neg_mean_squared_error".
    Alpha is tuned via GridSearchCV.
    """

    def __init__(
        self,
        cv: int = 5,
        max_iter: int = 5000,
        coef_threshold: float = 1e-4,
        alpha_grid: np.ndarray | None = None,
    ) -> None:
        self.cv = cv
        self.max_iter = max_iter
        self.coef_threshold = coef_threshold
        self.alpha_grid = alpha_grid if alpha_grid is not None else _ALPHA_GRID

    def select(self, matrix: FeatureMatrix) -> tuple[SelectionResult, FittedPredictor]:
        """Fit an L1-penalized model on the feature matrix and return selected features.

        Args:
            matrix: FeatureMatrix with X, y, and mode.

        Returns:
            Tuple of (SelectionResult, FittedPredictor).
        """
        X, y = matrix.X, matrix.y

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        estimator, param_grid, scoring = self._build_estimator(matrix.mode)
        search = GridSearchCV(
            estimator,
            param_grid,
            cv=self.cv,
            scoring=scoring,
            refit=True,
        )
        search.fit(X_scaled, y)

        best_model = search.best_estimator_
        # SGDClassifier.coef_ is (1, n_features) for binary; SGDRegressor.coef_ is (n_features,)
        coef_scaled = best_model.coef_[0] if matrix.mode == "classification" else best_model.coef_
        coef_original = coef_scaled / scaler.scale_

        mask = np.abs(coef_original) > self.coef_threshold
        result = SelectionResult(
            axis=matrix.axis,
            selected_features=[f for f, m in zip(matrix.features, mask) if m],
            coef=coef_original[mask].tolist(),
            cv_score=float(search.best_score_),
            cv_scoring=scoring,
        )
        return result, FittedPredictor(axis=matrix.axis, scaler=scaler, model=best_model)

    def _build_estimator(self, mode: str) -> tuple[Any, dict[str, list], str]:
        param_grid: dict[str, list] = {"alpha": self.alpha_grid.tolist()}
        if mode == "classification":
            return (
                SGDClassifier(loss="log_loss", penalty="l1", max_iter=self.max_iter, random_state=42),
                param_grid,
                "f1",
            )
        return (
            SGDRegressor(penalty="l1", max_iter=self.max_iter, random_state=42),
            param_grid,
            "neg_mean_squared_error",
        )
