from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .llm import BaseLLMClient
from .models import Feature
from .prompts import collection_synthesis as prompts

_LEVEL_LABELS: dict[int, list[str]] = {
    2: ["NO", "YES"],
    3: ["LOW", "MED", "HIGH"],
}


def _format_score(score: float, n_levels: int | None) -> str:
    if n_levels is None:
        return f"{score:.2f}"
    bucket = min(int(score * n_levels), n_levels - 1)
    labels = _LEVEL_LABELS.get(n_levels)
    if labels:
        return labels[bucket]
    return str(bucket + 1)


class TextSynthesizer:
    """Fits a length distribution and synthesizes texts from a user-provided feature matrix.

    Usage::

        synthesizer = TextSynthesizer().fit(features, lengths=lengths)
        synthesizer.save("synthesizer.json")

        synthesizer = TextSynthesizer.load("synthesizer.json")
        # X_sampled: user-created feature matrix (e.g. via SMOTE)
        texts = synthesizer.synthesize(X_sampled, llm=llm_client)
    """

    def fit(
        self,
        features: list[Feature],
        lengths: np.ndarray | None = None,
    ) -> TextSynthesizer:
        """Store feature definitions and optionally fit a log-normal length distribution.

        Args:
            features: Feature definitions corresponding to columns of X at synthesis time.
            lengths: Character counts of shape (n_texts,). If provided, a log-normal
                marginal distribution is fitted so that synthesize() can sample a
                target length for each generated text.

        Returns:
            self, for chaining.
        """
        self._features = list(features)

        if lengths is not None and len(lengths) > 0:
            log_lengths = np.log(np.clip(lengths, 1, None).astype(float))
            self._log_length_mean = float(np.mean(log_lengths))
            self._log_length_std = float(np.std(log_lengths))
            self._has_length = True
        else:
            self._log_length_mean = 0.0
            self._log_length_std = 0.0
            self._has_length = False

        return self

    def save(self, path: str | Path) -> None:
        """Serialize fitted state to a JSON file."""
        data = {
            "features": [{"hypothesis": f.hypothesis} for f in self._features],
            "has_length": self._has_length,
            "log_length_mean": self._log_length_mean,
            "log_length_std": self._log_length_std,
        }
        Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> TextSynthesizer:
        """Deserialize from a JSON file produced by save()."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        obj = cls.__new__(cls)
        obj._features = [Feature(hypothesis=f["hypothesis"]) for f in data["features"]]
        obj._has_length = data.get("has_length", False)
        obj._log_length_mean = float(data.get("log_length_mean", 0.0))
        obj._log_length_std = float(data.get("log_length_std", 0.0))
        return obj

    def synthesize(
        self,
        X: np.ndarray,
        *,
        llm: BaseLLMClient,
        language: str | None = None,
        n_levels: int | None = None,
        threshold: float | None = None,
        seed: int | None = None,
    ) -> list[str]:
        """Synthesize texts from a given feature matrix.

        Args:
            X: Feature matrix of shape (n_texts, n_features).
            llm: LLM client used for text generation.
            language: If specified, instruct the LLM to respond in this language.
            n_levels: Discretize scores into k levels (None=continuous, 2=YES/NO,
                3=LOW/MED/HIGH, k>=4=numeric labels). Default: None.
            threshold: Only include features with score >= threshold in the prompt.
                Default: None (include all features).
            seed: Random seed for length sampling. Default: None.

        Returns:
            List of len(X) generated texts.
        """
        rng = np.random.default_rng(seed)
        results: list[str] = []
        for sample in X:
            conditions: list[tuple[str, str]] = []
            for feature, score in zip(self._features, sample):
                if threshold is not None and score < threshold:
                    continue
                conditions.append((feature.hypothesis, _format_score(float(score), n_levels)))
            if self._has_length:
                log_len = rng.normal(self._log_length_mean, self._log_length_std)
                length: int | None = max(1, int(np.exp(log_len)))
            else:
                length = None
            user_msg = prompts.build_user_message(conditions, language=language, length=length)
            messages = [
                {"role": "system", "content": prompts.SYSTEM},
                {"role": "user", "content": user_msg},
            ]
            results.append(llm.complete(messages))
        return results
