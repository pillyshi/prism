from __future__ import annotations

import numpy as np
from sklearn.linear_model import LassoCV, LogisticRegressionCV
from sklearn.preprocessing import StandardScaler

from .models import FeatureMatrix, FittedPredictor, SelectionResult


class FeatureSelector:
    """Selects predictive features per axis using L1-regularized models with built-in CV.

    Classification mode: LogisticRegressionCV(penalty="l1", solver="saga"), scoring="f1".
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

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        cv_scoring = "f1" if matrix.mode == "classification" else "neg_mean_squared_error"

        if matrix.mode == "classification" and len(np.unique(y)) < 2:
            return (
                SelectionResult(axis=matrix.axis, selected_features=[], coef=[], cv_scoring=cv_scoring),
                FittedPredictor(axis=matrix.axis, scaler=scaler, model=None),
            )

        model = self._build_estimator(matrix.mode)
        model.fit(X_scaled, y)

        # LogisticRegressionCV.coef_ is (1, n_features) for binary; LassoCV.coef_ is (n_features,)
        coef_scaled = model.coef_[0] if matrix.mode == "classification" else model.coef_
        coef_original = coef_scaled / scaler.scale_

        if matrix.mode == "classification":
            # scores_ is keyed by class label (float); pick the positive class
            cv_score = float(model.scores_[max(model.scores_)].mean(axis=0).max())
        else:
            cv_score = float(-np.min(model.mse_path_.mean(axis=1)))

        mask = np.abs(coef_original) > self.coef_threshold
        result = SelectionResult(
            axis=matrix.axis,
            selected_features=[f for f, m in zip(matrix.features, mask) if m],
            coef=coef_original[mask].tolist(),
            cv_score=cv_score,
            cv_scoring=cv_scoring,
        )
        return result, FittedPredictor(axis=matrix.axis, scaler=scaler, model=model)

    def _build_estimator(self, mode: str) -> LogisticRegressionCV | LassoCV:
        if mode == "classification":
            return LogisticRegressionCV(
                l1_ratios=(1,), solver="saga", scoring="f1",
                cv=self.cv, max_iter=self.max_iter, random_state=42,
            )
        return LassoCV(cv=self.cv, max_iter=self.max_iter)
