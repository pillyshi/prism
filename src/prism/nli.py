from __future__ import annotations

import numpy as np


class NLIModel:
    """Wrapper around a sentence-transformers CrossEncoder for NLI scoring.

    The model returns three logits per pair. We apply sigmoid to the entailment
    logit and zero out pairs where the model does not predict entailment.

    Args:
        model_name: sentence-transformers CrossEncoder model name.
        entailment_idx: Column index of the entailment class in the model output.
            Defaults to 0, which matches ``cross-encoder/nli-deberta-v3-large``
            (label order: [entailment, neutral, contradiction]).
    """

    def __init__(self, model_name: str, entailment_idx: int = 0) -> None:
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        self.entailment_idx = entailment_idx
        self._model = CrossEncoder(model_name)

    def score(self, texts: list[str], hypotheses: list[str]) -> np.ndarray:
        """Score (text, hypothesis) pairs.

        Args:
            texts: List of n texts.
            hypotheses: List of n hypotheses, parallel to texts.

        Returns:
            np.ndarray of shape (n,) with values in [0, 1]. Pairs where the
            model does not predict entailment are scored as 0.
        """
        pairs = list(zip(texts, hypotheses))
        logits = self._model.predict(pairs)  # shape (n, 3) or (n,) depending on model
        logits = np.array(logits)
        if logits.ndim == 1:
            # Binary model — return sigmoid scores directly
            return 1.0 / (1.0 + np.exp(-logits))
        entail_logits = logits[:, self.entailment_idx]
        scores = 1.0 / (1.0 + np.exp(-entail_logits))
        scores[logits.argmax(axis=1) != self.entailment_idx] = 0.0
        return scores
