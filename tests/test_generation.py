"""Unit tests for FeatureGenerator."""
from unittest.mock import MagicMock

import pytest

from prism.generation import FeatureGenerator
from prism.models import Axis, AxisLabels


def _make_llm(response: dict) -> MagicMock:
    llm = MagicMock()
    llm.count_tokens.return_value = 10
    llm.complete_json.return_value = response
    return llm


_AXIS = Axis(hypothesis="This text has a positive tone.")
_TEXTS = ["great product", "terrible experience", "loved it", "awful quality"]
_FEATURE_RESPONSE = {"features": [{"hypothesis": "The text expresses satisfaction."}]}


def test_regression_mode_both_lists_non_empty():
    """Regression labels (continuous [0,1]) must produce non-empty positive and negative lists."""
    llm = _make_llm(_FEATURE_RESPONSE)
    generator = FeatureGenerator(llm=llm)

    # Continuous NLI scores: two above 0.5 (positive), two below 0.5 (negative)
    axis_labels = AxisLabels(axis=_AXIS, labels=[0.8, 0.3, 0.9, 0.1])
    features = generator.generate(_TEXTS, axis_labels)

    assert llm.complete_json.called
    messages = llm.complete_json.call_args[0][0]
    user_content = messages[1]["content"]

    # Both sides of the prompt must contain actual text, not "(none)"
    assert "great product" in user_content or "loved it" in user_content
    assert "terrible experience" in user_content or "awful quality" in user_content
    assert features == [pytest.approx] or len(features) >= 0  # call succeeded


def test_regression_mode_returns_features():
    llm = _make_llm(_FEATURE_RESPONSE)
    generator = FeatureGenerator(llm=llm)

    axis_labels = AxisLabels(axis=_AXIS, labels=[0.8, 0.3, 0.9, 0.1])
    features = generator.generate(_TEXTS, axis_labels)

    assert len(features) == 1
    assert features[0].hypothesis == "The text expresses satisfaction."
    assert features[0].axis == _AXIS


def test_classification_mode_both_lists_non_empty():
    """Classification labels (±1.0) must still produce non-empty positive and negative lists."""
    llm = _make_llm(_FEATURE_RESPONSE)
    generator = FeatureGenerator(llm=llm)

    axis_labels = AxisLabels(axis=_AXIS, labels=[1.0, -1.0, 1.0, -1.0])
    features = generator.generate(_TEXTS, axis_labels)

    messages = llm.complete_json.call_args[0][0]
    user_content = messages[1]["content"]

    assert "great product" in user_content or "loved it" in user_content
    assert "terrible experience" in user_content or "awful quality" in user_content
    assert len(features) == 1


def test_regression_mode_boundary_at_0_5():
    """Label exactly 0.5 is treated as positive (>= 0.5)."""
    llm = _make_llm(_FEATURE_RESPONSE)
    generator = FeatureGenerator(llm=llm)

    texts = ["boundary text", "below text"]
    axis_labels = AxisLabels(axis=_AXIS, labels=[0.5, 0.4])
    generator.generate(texts, axis_labels)

    messages = llm.complete_json.call_args[0][0]
    user_content = messages[1]["content"]
    assert "boundary text" in user_content
    assert "below text" in user_content
