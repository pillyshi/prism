from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Triple:
    """Shared triple structure for Axis and Feature."""

    name: str
    question: str    # e.g. "Does this text express a positive emotional tone?"
    hypothesis: str  # e.g. "This text has a positive emotional tone."


@dataclass(frozen=True)
class Axis(Triple):
    """A binary partition of a text collection with a natural language label."""


@dataclass(frozen=True)
class Feature(Triple):
    """A fine-grained property that explains why a text falls on one side of an axis."""

    axis: Axis


@dataclass
class AxisLabels:
    """Binary labels for a text collection under one axis."""

    axis: Axis
    labels: list[int]  # +1 (positive) / -1 (negative), parallel to texts


@dataclass
class FeatureScores:
    """Scores for one feature across all texts."""

    feature: Feature
    scores: list[float]  # NLI: continuous [0,1] / QA: discrete {0.0, 1.0}


@dataclass
class FeatureMatrix:
    """Full feature matrix for one axis, ready for Lasso."""

    axis: Axis
    features: list[Feature]
    X: np.ndarray  # shape (n_texts, n_features)
    y: np.ndarray  # shape (n_texts,), axis labels as float (+1.0 / -1.0)


@dataclass
class SelectionResult:
    """Features selected by Lasso for one axis."""

    axis: Axis
    selected_features: list[Feature]
    coef: list[float]  # Lasso coefficients (original scale)


@dataclass
class FittedPredictor:
    """Trained StandardScaler + LassoCV for one axis."""

    axis: Axis
    scaler: Any   # sklearn StandardScaler
    model: Any    # sklearn LassoCV
