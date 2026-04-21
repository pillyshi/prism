"""Unit tests for NLI-based AxisMerger."""
from unittest.mock import MagicMock

import numpy as np

from prism import Axis, AxisMerger


def _make_axis(hypothesis: str) -> Axis:
    return Axis(hypothesis=hypothesis)


def _make_nli_model(scores: dict[tuple[str, str], float]) -> MagicMock:
    """Mock NLIModel that returns entailment scores for (premise, hypothesis) pairs."""
    nli = MagicMock()

    def score_fn(texts: list[str], hypotheses: list[str]) -> np.ndarray:
        return np.array([scores.get((t, h), 0.0) for t, h in zip(texts, hypotheses)])

    nli.score.side_effect = score_fn
    return nli


def test_merge_deduplicates_equivalent_axes():
    h1 = "This text has a positive tone."
    h2 = "This text expresses positive sentiment."
    # h1 entails h2 above threshold → should merge
    nli = _make_nli_model({(h1, h2): 0.9, (h2, h1): 0.8})

    merger = AxisMerger(nli_model=nli, threshold=0.5)
    merged = merger.merge([[_make_axis(h1)], [_make_axis(h2)]])

    assert len(merged) == 1


def test_merge_preserves_distinct_axes():
    h1 = "This text has a positive tone."
    h2 = "This text discusses technical topics."
    # Low entailment in both directions → should not merge
    nli = _make_nli_model({(h1, h2): 0.1, (h2, h1): 0.1})

    merger = AxisMerger(nli_model=nli, threshold=0.5)
    merged = merger.merge([[_make_axis(h1)], [_make_axis(h2)]])

    assert len(merged) == 2
    hypotheses = {a.hypothesis for a in merged}
    assert hypotheses == {h1, h2}


def test_merge_returns_axis_objects():
    h = "This text discusses topic A."
    nli = _make_nli_model({})

    merger = AxisMerger(nli_model=nli, threshold=0.5)
    merged = merger.merge([[_make_axis(h)]])

    assert len(merged) == 1
    assert all(isinstance(a, Axis) for a in merged)


def test_merge_empty_input():
    nli = _make_nli_model({})
    merger = AxisMerger(nli_model=nli, threshold=0.5)
    merged = merger.merge([])
    assert merged == []


def test_merge_empty_runs():
    nli = _make_nli_model({})
    merger = AxisMerger(nli_model=nli, threshold=0.5)
    merged = merger.merge([[], []])
    assert merged == []


def test_merge_single_axis():
    h = "This text is written in first person."
    nli = _make_nli_model({})
    merger = AxisMerger(nli_model=nli, threshold=0.5)
    merged = merger.merge([[_make_axis(h)]])
    assert len(merged) == 1
    assert merged[0].hypothesis == h
