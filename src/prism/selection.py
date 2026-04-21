from __future__ import annotations

import numpy as np
from sklearn.linear_model import LassoCV, LogisticRegressionCV

from .models import FeatureMatrix, FittedPredictor, SelectionResult


class FeatureSelector:
    """Selects predictive features per axis using L1-regularized models with built-in CV.

    Classification mode: LogisticRegressionCV(penalty="elasticnet", l1_ratio=1), scoring="f1".
    Regression mode: LassoCV, scoring="neg_mean_squared_error".
    """

    def __init__(
        self,
        cv: int = 5,
        max_iter: int = 5000,
        coef_threshold: float = 1e-4,
    ) -> None:
        self.cv = cv
        self.max_iter = max_iter
        self.coef_threshold = coef_threshold

    def select(self, matrix: FeatureMatrix) -> tuple[SelectionResult, FittedPredictor]:
        """Fit an L1-penalized model on the feature matrix and return selected features.

        Args:
            matrix: FeatureMatrix with X, y, and mode.

        Returns:
            Tuple of (SelectionResult, FittedPredictor).
        """
        X, y = matrix.X, matrix.y

        cv_scoring = "f1" if matrix.mode == "classification" else "neg_mean_squared_error"

        if matrix.mode == "classification" and len(np.unique(y)) < 2:
            return (
                SelectionResult(axis=matrix.axis, selected_features=[], coef=[], cv_scoring=cv_scoring),
                FittedPredictor(axis=matrix.axis, model=None),
            )

        model = self._build_estimator(matrix.mode)
        model.fit(X, y)

        # LogisticRegressionCV.coef_ is (1, n_features) for binary; LassoCV.coef_ is (n_features,)
        coef = model.coef_[0] if matrix.mode == "classification" else model.coef_

        if matrix.mode == "classification":
            cv_score = float(model.scores_[max(model.scores_)].mean(axis=0).max())
        else:
            cv_score = float(-np.min(model.mse_path_.mean(axis=1)))

        mask = np.abs(coef) > self.coef_threshold
        result = SelectionResult(
            axis=matrix.axis,
            selected_features=[f for f, m in zip(matrix.features, mask) if m],
            coef=coef[mask].tolist(),
            cv_score=cv_score,
            cv_scoring=cv_scoring,
        )
        return result, FittedPredictor(axis=matrix.axis, model=model)

    def _build_estimator(self, mode: str) -> LogisticRegressionCV | LassoCV:
        if mode == "classification":
            return LogisticRegressionCV(
                penalty="elasticnet", l1_ratios=(1,), solver="saga", scoring="f1",
                cv=self.cv, max_iter=self.max_iter, random_state=42,
            )
        return LassoCV(cv=self.cv, max_iter=self.max_iter)
