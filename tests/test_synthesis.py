"""Unit tests for TextSynthesizer and supporting utilities."""
from unittest.mock import MagicMock

import numpy as np
import pytest

from prism import Axis, Feature, FeatureMatrix
from prism.prompts.text_synthesis import build_user_message
from prism.text_synthesis import TextSynthesizer, _sample_feature_vectors


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_matrix(X: np.ndarray) -> FeatureMatrix:
    axis = Axis(hypothesis="This text has a positive tone.")
    features = [
        Feature(hypothesis=f"Feature {i}.", axis=axis)
        for i in range(X.shape[1])
    ]
    y = np.zeros(X.shape[0])
    return FeatureMatrix(axis=axis, features=features, X=X, y=y)


def _make_llm(responses: list[str]) -> MagicMock:
    llm = MagicMock()
    llm.complete.side_effect = responses
    return llm


# ---------------------------------------------------------------------------
# _sample_feature_vectors
# ---------------------------------------------------------------------------

def test_sample_shape():
    X = np.random.default_rng(0).uniform(0, 1, (20, 5))
    rng = np.random.default_rng(42)
    out = _sample_feature_vectors(X, n=3, rng=rng)
    assert out.shape == (3, 5)


def test_sample_clipped_to_unit_interval():
    X = np.random.default_rng(0).uniform(0, 1, (20, 5))
    rng = np.random.default_rng(0)
    out = _sample_feature_vectors(X, n=100, rng=rng)
    assert out.min() >= 0.0
    assert out.max() <= 1.0


def test_sample_reproducible_with_same_seed():
    X = np.random.default_rng(0).uniform(0, 1, (20, 5))
    out1 = _sample_feature_vectors(X, n=5, rng=np.random.default_rng(7))
    out2 = _sample_feature_vectors(X, n=5, rng=np.random.default_rng(7))
    np.testing.assert_array_equal(out1, out2)


def test_sample_single_text():
    X = np.array([[0.2, 0.8, 0.5]])
    out = _sample_feature_vectors(X, n=4, rng=np.random.default_rng(0))
    assert out.shape == (4, 3)
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_sample_no_texts():
    X = np.empty((0, 3))
    out = _sample_feature_vectors(X, n=2, rng=np.random.default_rng(0))
    assert out.shape == (2, 3)
    np.testing.assert_array_equal(out, np.full((2, 3), 0.5))


def test_sample_single_feature():
    X = np.array([[0.1], [0.9], [0.5], [0.4]])
    out = _sample_feature_vectors(X, n=3, rng=np.random.default_rng(0))
    assert out.shape == (3, 1)
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_sample_zero_variance_feature():
    # All texts have the same score for one feature — near-singular covariance.
    X = np.column_stack([
        np.full(10, 0.5),
        np.random.default_rng(0).uniform(0, 1, 10),
    ])
    out = _sample_feature_vectors(X, n=3, rng=np.random.default_rng(0))
    assert out.shape == (3, 2)


# ---------------------------------------------------------------------------
# build_user_message
# ---------------------------------------------------------------------------

def test_build_user_message_contains_axis():
    msg = build_user_message(
        axis_hypothesis="This text has a positive tone.",
        conditions=[("Feature A.", 0.85, True)],
    )
    assert "This text has a positive tone." in msg


def test_build_user_message_true_false_labels():
    msg = build_user_message(
        axis_hypothesis="Axis.",
        conditions=[
            ("High feature.", 0.85, True),
            ("Low feature.", 0.12, False),
        ],
    )
    assert "TRUE" in msg
    assert "FALSE" in msg
    assert "0.85" in msg
    assert "0.12" in msg


def test_build_user_message_language_instruction():
    msg = build_user_message(
        axis_hypothesis="Axis.",
        conditions=[("Feature.", 0.5, True)],
        language="Japanese",
    )
    assert "Japanese" in msg


def test_build_user_message_no_language_instruction():
    msg = build_user_message(
        axis_hypothesis="Axis.",
        conditions=[("Feature.", 0.5, True)],
    )
    assert "Respond in" not in msg


# ---------------------------------------------------------------------------
# TextSynthesizer
# ---------------------------------------------------------------------------

def test_synthesize_returns_n_texts():
    X = np.random.default_rng(0).uniform(0, 1, (10, 3))
    matrix = _make_matrix(X)
    llm = _make_llm(["text A", "text B", "text C"])

    synthesizer = TextSynthesizer(llm=llm)
    results = synthesizer.synthesize(matrix, n=3, rng=np.random.default_rng(0))

    assert results == ["text A", "text B", "text C"]
    assert llm.complete.call_count == 3


def test_synthesize_passes_language_to_prompt():
    X = np.random.default_rng(0).uniform(0, 1, (10, 2))
    matrix = _make_matrix(X)
    llm = _make_llm(["テキスト"])

    synthesizer = TextSynthesizer(llm=llm)
    synthesizer.synthesize(matrix, n=1, language="Japanese", rng=np.random.default_rng(0))

    call_args = llm.complete.call_args[0][0]  # messages list
    user_content = call_args[1]["content"]
    assert "Japanese" in user_content


def test_synthesize_n_zero_returns_empty():
    X = np.random.default_rng(0).uniform(0, 1, (10, 2))
    matrix = _make_matrix(X)
    llm = _make_llm([])

    synthesizer = TextSynthesizer(llm=llm)
    results = synthesizer.synthesize(matrix, n=0, rng=np.random.default_rng(0))

    assert results == []
    llm.complete.assert_not_called()
