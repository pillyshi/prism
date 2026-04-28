from __future__ import annotations

from .llm import BaseLLMClient
from .models import Feature, NamedFeature
from .prompts import feature_naming as prompts


class FeatureNamer:
    """Assigns human-readable names to features via a single LLM call."""

    def __init__(self, llm: BaseLLMClient) -> None:
        self._llm = llm

    def name_features(
        self,
        features: list[Feature],
        language: str | None = None,
    ) -> list[NamedFeature]:
        """Generate short labels for a list of features.

        Args:
            features: Features to name.
            language: If specified, generate labels in this language.

        Returns:
            NamedFeature list parallel to the input.
        """
        if not features:
            return []
        hypotheses = [f.hypothesis for f in features]
        messages = [
            {"role": "system", "content": prompts.SYSTEM},
            {"role": "user", "content": prompts.build_user_message(hypotheses, language=language)},
        ]
        result = self._llm.complete_json(messages)
        names: list[str] = result.get("names", [])
        named: list[NamedFeature] = []
        for i, feature in enumerate(features):
            name = names[i] if i < len(names) else feature.hypothesis[:40]
            named.append(NamedFeature(name=name, feature=feature))
        return named
