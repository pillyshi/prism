from __future__ import annotations

import numpy as np

from .models import Axis
from .nli import NLIModel


class AxisMerger:
    """Merges axes from multiple discovery runs using NLI entailment."""

    def __init__(self, nli_model: NLIModel, threshold: float = 0.5) -> None:
        self._nli_model = nli_model
        self._threshold = threshold

    def merge(self, axes_per_run: list[list[Axis]]) -> list[Axis]:
        """Consolidate axes from multiple runs into a unified, non-redundant set.

        Two axes are merged when either hypothesis entails the other above threshold.

        Args:
            axes_per_run: One list of Axis objects per run.

        Returns:
            Merged list of Axis objects (one representative per equivalence class).
        """
        all_axes: list[Axis] = [ax for run in axes_per_run for ax in run]
        n = len(all_axes)
        if n == 0:
            return []
        if n == 1:
            return list(all_axes)

        hypotheses = [ax.hypothesis for ax in all_axes]

        # Batch both directions in a single NLI call each
        h_i = [hypotheses[i] for i in range(n) for j in range(n) if i != j]
        h_j = [hypotheses[j] for i in range(n) for j in range(n) if i != j]
        scores_flat = self._nli_model.score(h_i, h_j)

        # Build n×n score matrix (diagonal = 0)
        score_matrix = np.zeros((n, n))
        idx = 0
        for i in range(n):
            for j in range(n):
                if i != j:
                    score_matrix[i, j] = scores_flat[idx]
                    idx += 1

        # Bidirectional entailment: merge if either direction exceeds threshold
        similar = (score_matrix > self._threshold) | (score_matrix.T > self._threshold)

        # Union-Find
        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x: int, y: int) -> None:
            parent[find(x)] = find(y)

        for i in range(n):
            for j in range(i + 1, n):
                if similar[i, j]:
                    union(i, j)

        # Collect one representative per equivalence class (first encountered)
        seen: set[int] = set()
        result: list[Axis] = []
        for i, ax in enumerate(all_axes):
            root = find(i)
            if root not in seen:
                seen.add(root)
                result.append(ax)
        return result
