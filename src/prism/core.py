from __future__ import annotations

import random
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge

from .llm import BaseLLMClient, LLMClient, LangChainLLMClient
from .models import Feature, FitResult, NamedFeature
from .naming import FeatureNamer
from .nli import NLIModel
from .sampling import sample_texts_within_budget
from .prompts import collection_description as prompts

_PROMPT_OVERHEAD = 500


class Prism:
    """Facade for the Prism pipeline.

    Example::

        prism = Prism(llm="gpt-4o", nli_model="cross-encoder/nli-deberta-v3-large")
        features = prism.generate_features(texts, n=20)
        X = prism.score(texts, features)
        result = prism.fit(X, y, features)
    """

    def __init__(
        self,
        llm: str | Any = "gpt-4o",
        nli_model: str = "cross-encoder/nli-deberta-v3-large",
        api_key: str | None = None,
    ) -> None:
        if isinstance(llm, str):
            self._llm: BaseLLMClient = LLMClient(model=llm, api_key=api_key)
        elif isinstance(llm, BaseLLMClient):
            self._llm = llm
        else:
            self._llm = LangChainLLMClient(model=llm)
        self._nli_model = NLIModel(model_name=nli_model)
        self._namer = FeatureNamer(llm=self._llm)

    def generate_features(
        self,
        texts: list[str],
        n: int = 20,
        context_limit: int = 100_000,
        seed: int | None = None,
        language: str | None = None,
    ) -> list[Feature]:
        """Generate features that characterize the text collection.

        Args:
            texts: Text collection to characterize.
            n: Number of features to generate.
            context_limit: Max tokens available in the LLM context window.
            seed: Random seed for reproducible text sampling.
            language: If specified, generate feature hypotheses in this language.

        Returns:
            List of Feature objects.
        """
        rng = random.Random(seed)
        budget = context_limit - _PROMPT_OVERHEAD
        sampled = sample_texts_within_budget(texts, budget, self._llm.count_tokens, rng=rng)
        messages = [
            {"role": "system", "content": prompts.SYSTEM},
            {"role": "user", "content": prompts.build_user_message(sampled, n=n, language=language)},
        ]
        result = self._llm.complete_json(messages)
        return [Feature(hypothesis=item["hypothesis"]) for item in result.get("features", [])]

    def score(self, texts: list[str], features: list[Feature]) -> np.ndarray:
        """Score texts against features using NLI.

        Args:
            texts: Text collection to score.
            features: Features to score against.

        Returns:
            np.ndarray of shape (n_texts, n_features) with entailment scores in [0, 1].
        """
        if not features:
            return np.empty((len(texts), 0))
        columns = []
        for feature in features:
            hypotheses = [feature.hypothesis] * len(texts)
            scores = self._nli_model.score(texts, hypotheses)
            columns.append(scores)
        return np.column_stack(columns)

    def name_features(
        self,
        features: list[Feature],
        language: str | None = None,
    ) -> list[NamedFeature]:
        """Assign human-readable names to features.

        Args:
            features: Features to name.
            language: If specified, generate names in this language.

        Returns:
            NamedFeature list parallel to the input.
        """
        return self._namer.name_features(features, language=language)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        features: list[Feature],
    ) -> FitResult:
        """Fit a linear model (y ~ X) and return an interpretable report.

        Automatically selects classification (LogisticRegression) when y has
        exactly 2 unique values, otherwise uses Ridge regression.

        Args:
            X: Feature matrix of shape (n_texts, n_features).
            y: Target values of shape (n_texts,).
            features: Features corresponding to columns of X.

        Returns:
            FitResult with per-feature coefficients and model score.
        """
        is_classification = len(np.unique(y)) == 2
        if is_classification:
            model = LogisticRegression(max_iter=1000)
            model.fit(X, y)
            coef = model.coef_[0].tolist()
            intercept = float(model.intercept_[0])
            score = float(model.score(X, y))
            scoring = "accuracy"
        else:
            model = Ridge()
            model.fit(X, y)
            coef = model.coef_.tolist()
            intercept = float(model.intercept_)
            score = float(model.score(X, y))
            scoring = "r2"
        return FitResult(
            features=list(features),
            coef=coef,
            intercept=intercept,
            score=score,
            scoring=scoring,
        )
