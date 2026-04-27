from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np


@dataclass(frozen=True)
class Triple:
    """Shared structure for Axis and Feature."""

    hypothesis: str  # e.g. "This text has a positive emotional tone."


@dataclass(frozen=True)
class Axis(Triple):
    """A binary partition of a text collection with a natural language label."""


@dataclass(frozen=True)
class Feature(Triple):
    """A fine-grained property that explains why a text falls on one side of an axis."""

    axis: Axis


@dataclass(frozen=True)
class CollectionFeature:
    """A property characterizing a text collection as a whole (no axis)."""

    hypothesis: str


@dataclass(frozen=True)
class NamedFeature:
    """Display-layer wrapper: a human-readable name assigned to a finalized feature."""

    name: str
    feature: Feature


@dataclass
class AxisLabels:
    """Labels for a text collection under one axis.

    Classification mode: +1.0 / -1.0. Regression mode: raw NLI scores in [0, 1].
    """

    axis: Axis
    labels: list[float]  # classification: +1.0/-1.0, regression: [0, 1]


@dataclass
class FeatureScores:
    """Scores for one feature across all texts."""

    feature: Feature
    scores: list[float]  # NLI: continuous [0,1] / QA: discrete {0.0, 1.0}


@dataclass
class FeatureMatrix:
    """Full feature matrix for one axis, ready for feature selection."""

    axis: Axis
    features: list[Feature]
    X: np.ndarray  # shape (n_texts, n_features)
    y: np.ndarray  # shape (n_texts,); classification: +1.0/-1.0, regression: [0, 1]
    mode: Literal["classification", "regression"] = "regression"


@dataclass
class SelectionResult:
    """Features selected by L1-regularized model for one axis."""

    axis: Axis
    selected_features: list[Feature]
    coef: list[float]
    cv_score: float = 0.0      # best CV score
    cv_scoring: str = ""       # scoring metric name (e.g. "f1", "neg_mean_squared_error")


@dataclass
class FittedPredictor:
    """Trained estimator for one axis."""

    axis: Axis
    model: Any    # sklearn LogisticRegressionCV or LassoCV
