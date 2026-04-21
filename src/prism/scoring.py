from __future__ import annotations

import numpy as np

from .models import Feature, FeatureScores
from .nli import NLIModel


class NLIScorer:
    """Score texts against features using an NLI model (continuous scores)."""

    def __init__(self, model: NLIModel) -> None:
        self._model = model

    def score(self, texts: list[str], features: list[Feature]) -> list[FeatureScores]:
        """Score all texts against all features.

        Returns one FeatureScores per feature with scores parallel to texts.
        """
        results: list[FeatureScores] = []
        for feature in features:
            hypotheses = [feature.hypothesis] * len(texts)
            scores = self._model.score(texts, hypotheses)
            results.append(FeatureScores(feature=feature, scores=scores.tolist()))
        return results
