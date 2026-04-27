from __future__ import annotations

import random

from .llm import BaseLLMClient
from .models import CollectionFeature
from .sampling import sample_texts_within_budget
from .prompts import collection_description as prompts


_PROMPT_OVERHEAD = 500


class CollectionDescriber:
    """Generates features that characterize a text collection as a whole."""

    def __init__(self, llm: BaseLLMClient) -> None:
        self._llm = llm

    def describe(
        self,
        texts: list[str],
        n_features: int = 10,
        context_limit: int = 100_000,
        seed: int | None = None,
        language: str | None = None,
    ) -> list[CollectionFeature]:
        """Generate features characterizing the collection as a whole.

        Args:
            texts: Text collection to characterize.
            n_features: Number of features to generate.
            context_limit: Max tokens available in the LLM context window.
            seed: Random seed for reproducible text sampling.
            language: If specified, generate feature hypotheses in this language.

        Returns:
            List of CollectionFeature objects.
        """
        rng = random.Random(seed)
        budget = context_limit - _PROMPT_OVERHEAD
        sampled = sample_texts_within_budget(texts, budget, self._llm.count_tokens, rng=rng)

        messages = [
            {"role": "system", "content": prompts.SYSTEM},
            {
                "role": "user",
                "content": prompts.build_user_message(sampled, n=n_features, language=language),
            },
        ]
        result = self._llm.complete_json(messages)
        return [
            CollectionFeature(hypothesis=item["hypothesis"])
            for item in result.get("features", [])
        ]
