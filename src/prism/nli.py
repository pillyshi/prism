from __future__ import annotations

import numpy as np


class NLIModel:
    """Wrapper around a sentence-transformers CrossEncoder for NLI scoring.

    The model returns three logits per pair: [contradiction, neutral, entailment].
    We apply softmax and return the entailment probability as the score.
    """

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        self._model = CrossEncoder(model_name)

    def score(self, texts: list[str], hypotheses: list[str]) -> np.ndarray:
        """Score (text, hypothesis) pairs.

        Args:
            texts: List of n texts.
            hypotheses: List of n hypotheses, parallel to texts.

        Returns:
            np.ndarray of shape (n,) with entailment probabilities in [0, 1].
        """
        pairs = list(zip(texts, hypotheses))
        logits = self._model.predict(pairs)  # shape (n, 3) or (n,) depending on model
        logits = np.array(logits)
        if logits.ndim == 1:
            # Binary model — return sigmoid scores directly
            return 1.0 / (1.0 + np.exp(-logits))
        return _softmax(logits)[:, 2]  # entailment column


def _softmax(x: np.ndarray) -> np.ndarray:
    """Row-wise softmax."""
    x = x - x.max(axis=1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / exp_x.sum(axis=1, keepdims=True)
