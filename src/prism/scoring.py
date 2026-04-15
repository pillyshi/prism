from __future__ import annotations

import json

import numpy as np

from .models import Feature, FeatureScores
from .llm import LLMClient
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


class QAScorer:
    """Score texts against features using an LLM (discrete Yes/No scores).

    Batches all features for each text into a single LLM call to reduce cost.
    """

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    def score(self, texts: list[str], features: list[Feature]) -> list[FeatureScores]:
        """Score all texts against all features.

        Returns one FeatureScores per feature with scores parallel to texts.
        """
        # scores[feature_idx][text_idx]
        all_scores: list[list[float]] = [[] for _ in features]

        for text in texts:
            answers = self._score_text(text, features)
            for i, answer in enumerate(answers):
                all_scores[i].append(answer)

        return [
            FeatureScores(feature=feature, scores=all_scores[i])
            for i, feature in enumerate(features)
        ]

    def _score_text(self, text: str, features: list[Feature]) -> list[float]:
        """Ask the LLM to answer all feature questions for a single text."""
        questions_block = "\n".join(
            f"{i + 1}. {f.question}" for i, f in enumerate(features)
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "Answer each question about the given text with \"yes\" or \"no\". "
                    "Respond with JSON only: {\"answers\": [\"yes\", \"no\", ...]}"
                ),
            },
            {
                "role": "user",
                "content": f'Text: "{text}"\n\nQuestions:\n{questions_block}',
            },
        ]
        result = self._llm.complete_json(messages)
        answers = result.get("answers", [])
        # Pad with 0.0 if the model returns fewer answers than expected
        scores: list[float] = []
        for i in range(len(features)):
            if i < len(answers):
                scores.append(1.0 if str(answers[i]).lower().strip() == "yes" else 0.0)
            else:
                scores.append(0.0)
        return scores
