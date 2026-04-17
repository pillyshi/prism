"""Unit tests for AxisMerger."""
from unittest.mock import MagicMock

import pytest

from prism import Axis, AxisMerger


def _make_axis(name: str) -> Axis:
    return Axis(
        name=name,
        question=f"Does this text have {name}?",
        hypothesis=f"This text has {name}.",
    )


def _make_llm(response: dict) -> MagicMock:
    llm = MagicMock()
    llm.complete_json.return_value = response
    return llm


def test_merge_deduplicates_equivalent_axes():
    axes_run1 = [_make_axis("positive tone")]
    axes_run2 = [_make_axis("positive emotional tone")]
    llm = _make_llm({"merged_axes": [
        {"name": "positive tone", "question": "Does this text express a positive tone?", "hypothesis": "This text expresses a positive tone."}
    ]})

    merger = AxisMerger(llm=llm)
    merged = merger.merge([axes_run1, axes_run2])

    assert len(merged) == 1
    assert merged[0].name == "positive tone"


def test_merge_preserves_distinct_axes():
    axes_run1 = [_make_axis("positive tone")]
    axes_run2 = [_make_axis("factual content")]
    llm = _make_llm({"merged_axes": [
        {"name": "positive tone", "question": "q1", "hypothesis": "h1"},
        {"name": "factual content", "question": "q2", "hypothesis": "h2"},
    ]})

    merger = AxisMerger(llm=llm)
    merged = merger.merge([axes_run1, axes_run2])

    assert len(merged) == 2
    names = {a.name for a in merged}
    assert names == {"positive tone", "factual content"}


def test_merge_returns_axis_objects():
    axes = [_make_axis("topic A")]
    llm = _make_llm({"merged_axes": [
        {"name": "topic A", "question": "Does this text discuss topic A?", "hypothesis": "This text discusses topic A."}
    ]})

    merger = AxisMerger(llm=llm)
    merged = merger.merge([axes])

    assert all(isinstance(a, Axis) for a in merged)


def test_merge_empty_response():
    llm = _make_llm({"merged_axes": []})
    merger = AxisMerger(llm=llm)
    merged = merger.merge([[_make_axis("A")]])
    assert merged == []


def test_merge_passes_all_runs_to_llm():
    axes_run1 = [_make_axis("A")]
    axes_run2 = [_make_axis("B")]
    axes_run3 = [_make_axis("C")]
    llm = _make_llm({"merged_axes": []})

    merger = AxisMerger(llm=llm)
    merger.merge([axes_run1, axes_run2, axes_run3])

    call_args = llm.complete_json.call_args
    user_content = call_args[0][0][1]["content"]
    assert "Run 1" in user_content
    assert "Run 2" in user_content
    assert "Run 3" in user_content
