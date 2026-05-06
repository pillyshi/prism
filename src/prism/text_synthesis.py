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
    """Fits a feature distribution and synthesizes texts from a given feature matrix.

    Usage::

        synthesizer = TextSynthesizer().fit(X, features)
        synthesizer.save("synthesizer.json")

        synthesizer = TextSynthesizer.load("synthesizer.json")
        texts = synthesizer.synthesize(X_new, llm=llm_client)
    """

    def fit(
        self,
        X: np.ndarray,
        features: list[Feature],
        lengths: np.ndarray | None = None,
    ) -> TextSynthesizer:
        """Estimate the joint distribution of feature scores and text length from X.

        Args:
            X: Feature matrix of shape (n_texts, n_features).
            features: Features corresponding to columns of X.
            lengths: Character counts of shape (n_texts,). If provided,
                log(lengths) is appended as an extra dimension and fitted
                jointly with features so that their correlation is captured.

        Returns:
            self, for chaining.
        """
        self._features = list(features)
        n_texts, n_features = X.shape

        if lengths is not None and len(lengths) > 0:
            log_lengths = np.log(np.clip(lengths, 1, None).astype(float))
            X_aug = np.column_stack([X, log_lengths]) if n_features > 0 else log_lengths.reshape(-1, 1)
            self._has_length = True
        else:
            X_aug = X
            self._has_length = False

        n_aug = X_aug.shape[1]

        if n_texts == 0:
            self._mean = np.full(n_aug, 0.5)
            self._cov = np.eye(n_aug)
        elif n_texts == 1:
            self._mean = X_aug[0].copy()
            self._cov = np.eye(n_aug)
        else:
            self._mean = X_aug.mean(axis=0)
            self._cov = np.atleast_2d(np.cov(X_aug.T)) + np.eye(n_aug) * 1e-6

        return self

    def save(self, path: str | Path) -> None:
        """Serialize fitted state to a JSON file."""
        data = {
            "features": [{"hypothesis": f.hypothesis} for f in self._features],
            "mean": self._mean.tolist(),
            "cov": self._cov.tolist(),
            "has_length": self._has_length,
        }
        Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> TextSynthesizer:
        """Deserialize from a JSON file produced by save()."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        features = [Feature(hypothesis=f["hypothesis"]) for f in data["features"]]
        mean = np.array(data["mean"])
        cov = np.array(data["cov"])
        n = len(features)
        has_length = data.get("has_length", False)
        n_aug = n + (1 if has_length else 0)
        if mean.shape != (n_aug,) or cov.shape != (n_aug, n_aug):
            raise ValueError(
                f"Serialized shape mismatch: expected ({n_aug},) and ({n_aug},{n_aug}), "
                f"got {mean.shape} and {cov.shape}"
            )
        obj = cls.__new__(cls)
        obj._features = features
        obj._mean = mean
        obj._cov = cov
        obj._has_length = has_length
        return obj

    def synthesize(
        self,
        X: np.ndarray,
        *,
        llm: BaseLLMClient,
        lengths: np.ndarray | None = None,
        language: str | None = None,
        n_levels: int | None = None,
        threshold: float | None = None,
    ) -> list[str]:
        """Synthesize texts from a given feature matrix.

        Args:
            X: Feature matrix of shape (n_texts, n_features).
            llm: LLM client used for text generation.
            lengths: Target character counts of shape (n_texts,). If provided,
                each generated text is prompted to match the corresponding length.
            language: If specified, instruct the LLM to respond in this language.
            n_levels: Discretize scores into k levels (None=continuous, 2=YES/NO,
                3=LOW/MED/HIGH, k>=4=numeric labels). Default: None.
            threshold: Only include features with score >= threshold in the prompt.
                Default: None (include all features).

        Returns:
            List of len(X) generated texts.
        """
        results: list[str] = []
        for i, sample in enumerate(X):
            conditions: list[tuple[str, str]] = []
            for feature, score in zip(self._features, sample):
                if threshold is not None and score < threshold:
                    continue
                conditions.append((feature.hypothesis, _format_score(float(score), n_levels)))
            length = int(lengths[i]) if lengths is not None else None
            user_msg = prompts.build_user_message(conditions, language=language, length=length)
            messages = [
                {"role": "system", "content": prompts.SYSTEM},
                {"role": "user", "content": user_msg},
            ]
            results.append(llm.complete(messages))
        return results
