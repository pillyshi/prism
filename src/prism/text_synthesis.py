from __future__ import annotations

import numpy as np

from .llm import BaseLLMClient
from .models import FeatureMatrix
from .prompts import text_synthesis as prompts

_BOOL_THRESHOLD = 0.5


class TextSynthesizer:
    """Samples feature vectors from a FeatureMatrix and generates synthetic texts."""

    def __init__(self, llm: BaseLLMClient) -> None:
        self._llm = llm

    def synthesize(
        self,
        matrix: FeatureMatrix,
        n: int = 1,
        language: str | None = None,
        rng: np.random.Generator | None = None,
    ) -> list[str]:
        """Generate n synthetic texts consistent with the feature distribution.

        Args:
            matrix: FeatureMatrix for a single axis.
            n: Number of texts to generate.
            language: If specified, instruct the LLM to respond in this language.
            rng: Optional numpy Generator for reproducibility.
        """
        if rng is None:
            rng = np.random.default_rng()
        samples = _sample_feature_vectors(matrix.X, n=n, rng=rng)
        results: list[str] = []
        for sample in samples:
            conditions = [
                (feat.hypothesis, float(score), float(score) > _BOOL_THRESHOLD)
                for feat, score in zip(matrix.features, sample)
            ]
            user_msg = prompts.build_user_message(
                axis_hypothesis=matrix.axis.hypothesis,
                conditions=conditions,
                language=language,
            )
            messages = [
                {"role": "system", "content": prompts.SYSTEM},
                {"role": "user", "content": user_msg},
            ]
            results.append(self._llm.complete(messages))
        return results


def _sample_feature_vectors(X: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Fit multivariate Gaussian to X rows and draw n samples, clipped to [0, 1]."""
    n_texts, n_features = X.shape
    if n_texts < 2:
        mean = X.mean(axis=0) if n_texts == 1 else np.full(n_features, 0.5)
        return np.clip(np.tile(mean, (n, 1)), 0.0, 1.0)
    mean = X.mean(axis=0)
    cov = np.atleast_2d(np.cov(X.T)) + np.eye(n_features) * 1e-6
    return np.clip(rng.multivariate_normal(mean=mean, cov=cov, size=n), 0.0, 1.0)
