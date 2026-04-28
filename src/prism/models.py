from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Feature:
    """A property of a text, expressed as a natural language hypothesis for NLI scoring."""

    hypothesis: str  # e.g. "This text expresses satisfaction with the product."


@dataclass(frozen=True)
class NamedFeature:
    """Display-layer wrapper: a human-readable name assigned to a feature."""

    name: str
    feature: Feature


@dataclass
class FeatureDependency:
    """Dependency of one feature on the others, fitted by linear regression."""

    feature: Feature
    r2: float                      # R² of X[:,i] ~ X[:,j≠i]
    predictors: list[Feature]      # features with non-zero Lasso coefficients
    coef: list[float]              # corresponding coefficients


@dataclass
class FitResult:
    """Result of fitting a linear model (y ~ X) on a feature matrix."""

    features: list[Feature]
    coef: list[float]
    intercept: float
    score: float                   # R² for regression, accuracy for classification
    scoring: str                   # "r2" or "accuracy"
