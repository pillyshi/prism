from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .llm import BaseLLMClient
from .models import CollectionFeature
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


class CollectionSynthesizer:
    """Samples from a fitted feature distribution and synthesizes new texts.

    Usage::

        synthesizer = CollectionSynthesizer()
        synthesizer.fit(X, features)
        synthesizer.save("collection.json")

        synthesizer = CollectionSynthesizer.load("collection.json")
        texts = synthesizer.sample(n=5, llm=llm_client)
    """

    def fit(
        self,
        X: np.ndarray,
        features: list[CollectionFeature],
        lengths: np.ndarray | None = None,
    ) -> CollectionSynthesizer:
        """Estimate the multivariate Gaussian distribution from X.

        Args:
            X: Feature matrix of shape (n_texts, n_features).
            features: CollectionFeatures corresponding to columns of X.
            lengths: Character counts of shape (n_texts,). If provided, a
                log-normal distribution is fitted and used during sampling.

        Returns:
            self, for chaining.
        """
        self._features = list(features)
        n_texts, n_features = X.shape
        if n_texts == 0:
            self._mean = np.full(n_features, 0.5)
            self._cov = np.eye(n_features)
        elif n_texts == 1:
            self._mean = X[0].copy()
            self._cov = np.eye(n_features)
        else:
            self._mean = X.mean(axis=0)
            self._cov = np.atleast_2d(np.cov(X.T)) + np.eye(n_features) * 1e-6

        if lengths is not None and len(lengths) > 0:
            log_lengths = np.log(np.clip(lengths, 1, None).astype(float))
            self._len_mu: float | None = float(log_lengths.mean())
            self._len_sigma: float | None = float(log_lengths.std()) if len(lengths) > 1 else 0.3
        else:
            self._len_mu = None
            self._len_sigma = None

        return self

    def save(self, path: str | Path) -> None:
        """Serialize fitted state to a JSON file."""
        data = {
            "features": [{"hypothesis": f.hypothesis} for f in self._features],
            "mean": self._mean.tolist(),
            "cov": self._cov.tolist(),
            "len_mu": self._len_mu,
            "len_sigma": self._len_sigma,
        }
        Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> CollectionSynthesizer:
        """Deserialize from a JSON file produced by save()."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        features = [CollectionFeature(hypothesis=f["hypothesis"]) for f in data["features"]]
        mean = np.array(data["mean"])
        cov = np.array(data["cov"])
        n = len(features)
        if mean.shape != (n,) or cov.shape != (n, n):
            raise ValueError(
                f"Serialized shape mismatch: expected ({n},) and ({n},{n}), "
                f"got {mean.shape} and {cov.shape}"
            )
        obj = cls.__new__(cls)
        obj._features = features
        obj._mean = mean
        obj._cov = cov
        obj._len_mu = data.get("len_mu")
        obj._len_sigma = data.get("len_sigma")
        return obj

    def sample(
        self,
        n: int,
        *,
        llm: BaseLLMClient,
        language: str | None = None,
        n_levels: int | None = None,
        threshold: float | None = None,
        rng: np.random.Generator | None = None,
    ) -> list[str]:
        """Sample n feature vectors and synthesize texts.

        Args:
            n: Number of texts to generate.
            llm: LLM client used for text generation.
            language: If specified, instruct the LLM to respond in this language.
            n_levels: Discretize scores into k levels (None=continuous, 2=YES/NO,
                3=LOW/MED/HIGH, k>=4=numeric labels). Default: None.
            threshold: Only include features with score >= threshold in the prompt.
                Default: None (include all features).
            rng: Optional numpy Generator for reproducibility.

        Returns:
            List of n generated texts.
        """
        if rng is None:
            rng = np.random.default_rng()

        n_features = len(self._features)
        if n_features == 0:
            samples = np.empty((n, 0))
        else:
            samples = np.clip(
                rng.multivariate_normal(mean=self._mean, cov=self._cov, size=n),
                0.0,
                1.0,
            )

        results: list[str] = []
        for sample in samples:
            conditions: list[tuple[str, str]] = []
            for feature, score in zip(self._features, sample):
                if threshold is not None and score < threshold:
                    continue
                conditions.append((feature.hypothesis, _format_score(float(score), n_levels)))

            length: int | None = None
            if self._len_mu is not None:
                sigma = max(self._len_sigma or 0.0, 1e-6)
                length = max(1, int(round(rng.lognormal(self._len_mu, sigma))))

            user_msg = prompts.build_user_message(conditions, language=language, length=length)
            messages = [
                {"role": "system", "content": prompts.SYSTEM},
                {"role": "user", "content": user_msg},
            ]
            results.append(llm.complete(messages))
        return results
