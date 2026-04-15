from __future__ import annotations

import random
from typing import Literal

import numpy as np

from .llm import BaseLLMClient
from .models import Axis, AxisLabels
from .nli import NLIModel
from .sampling import sample_texts_within_budget
from .prompts import axis_discovery as prompts


_PROMPT_OVERHEAD = 500  # reserved tokens for the prompt template itself


class AxisDiscoverer:
    """Discovers binary axes from a text collection using an LLM."""

    def __init__(self, llm: BaseLLMClient, nli_model: NLIModel | None = None) -> None:
        self._llm = llm
        self._nli_model = nli_model

    def discover(
        self,
        texts: list[str],
        n: int,
        context_limit: int = 100_000,
        seed: int | None = None,
    ) -> list[Axis]:
        """Generate n axes from a random sample of texts.

        Args:
            texts: Full text collection.
            n: Number of axes to discover.
            context_limit: Max tokens available in the LLM context window.
            seed: Random seed for reproducible sampling.

        Returns:
            List of Axis objects.
        """
        rng = random.Random(seed)
        token_budget = context_limit - _PROMPT_OVERHEAD
        sampled = sample_texts_within_budget(
            texts,
            token_budget=token_budget,
            tokenizer_fn=self._llm.count_tokens,
            rng=rng,
        )

        messages = [
            {"role": "system", "content": prompts.SYSTEM},
            {"role": "user", "content": prompts.build_user_message(sampled, n)},
        ]
        result = self._llm.complete_json(messages)
        return [
            Axis(
                name=item["name"],
                question=item["question"],
                hypothesis=item["hypothesis"],
            )
            for item in result.get("axes", [])
        ]

    def label(
        self,
        texts: list[str],
        axes: list[Axis],
        method: Literal["nli", "qa"] = "nli",
        threshold: float = 0.5,
    ) -> list[AxisLabels]:
        """Label texts as positive (+1) or negative (-1) for each axis.

        Args:
            texts: Full text collection.
            axes: Axes to label against.
            method: Scoring method ("nli" or "qa").
            threshold: Score threshold for positive label (NLI only).

        Returns:
            One AxisLabels per axis, parallel to texts.
        """
        if method == "nli":
            return self._label_nli(texts, axes, threshold)
        raise NotImplementedError(f"Labeling with method='{method}' is not yet supported.")

    def _label_nli(
        self, texts: list[str], axes: list[Axis], threshold: float
    ) -> list[AxisLabels]:
        if self._nli_model is None:
            raise ValueError("nli_model must be provided to use NLI labeling.")
        results: list[AxisLabels] = []
        for axis in axes:
            hypotheses = [axis.hypothesis] * len(texts)
            scores = self._nli_model.score(texts, hypotheses)
            labels = [1 if s >= threshold else -1 for s in scores]
            results.append(AxisLabels(axis=axis, labels=labels))
        return results
