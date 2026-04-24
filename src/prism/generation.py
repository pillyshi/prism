from __future__ import annotations

import random

from .llm import BaseLLMClient
from .models import Axis, AxisLabels, Feature
from .sampling import sample_texts_within_budget
from .prompts import feature_generation as prompts


_PROMPT_OVERHEAD = 800  # reserved tokens for the prompt template itself


class FeatureGenerator:
    """Generates discriminative features for an axis using contrastive LLM prompting."""

    def __init__(self, llm: BaseLLMClient) -> None:
        self._llm = llm

    def generate(
        self,
        texts: list[str],
        axis_labels: AxisLabels,
        n_features: int = 10,
        context_limit: int = 100_000,
        seed: int | None = None,
        language: str | None = None,
    ) -> list[Feature]:
        """Generate features that discriminate positive from negative examples.

        Args:
            texts: Full text collection, parallel to axis_labels.labels.
            axis_labels: Binary labels for each text under the target axis.
            n_features: Number of features to generate.
            context_limit: Max tokens available in the LLM context window.
            seed: Random seed for reproducible sampling.

        Returns:
            List of Feature objects associated with the axis.
        """
        axis = axis_labels.axis
        rng = random.Random(seed)

        positive_texts = [t for t, l in zip(texts, axis_labels.labels) if l >= 0.5]
        negative_texts = [t for t, l in zip(texts, axis_labels.labels) if l < 0.5]

        # Split budget evenly between positive and negative sides
        half_budget = (context_limit - _PROMPT_OVERHEAD) // 2
        tokenizer_fn = self._llm.count_tokens

        sampled_positive = sample_texts_within_budget(
            positive_texts, half_budget, tokenizer_fn, rng=random.Random(rng.random())
        )
        sampled_negative = sample_texts_within_budget(
            negative_texts, half_budget, tokenizer_fn, rng=random.Random(rng.random())
        )

        messages = [
            {"role": "system", "content": prompts.SYSTEM},
            {
                "role": "user",
                "content": prompts.build_user_message(
                    axis_hypothesis=axis.hypothesis,
                    positive_texts=sampled_positive,
                    negative_texts=sampled_negative,
                    n=n_features,
                    language=language,
                ),
            },
        ]
        result = self._llm.complete_json(messages)
        return [
            Feature(hypothesis=item["hypothesis"], axis=axis)
            for item in result.get("features", [])
        ]
