"""Unit tests for TextSynthesizer."""
import json
import re
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from prism import Feature, TextSynthesizer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_features(n: int) -> list[Feature]:
    return [Feature(hypothesis=f"Feature {i}.") for i in range(n)]


def _make_llm(responses: list[str]) -> MagicMock:
    llm = MagicMock()
    llm.complete.side_effect = responses
    return llm


def _fit(X: np.ndarray | None = None, n_features: int = 3) -> TextSynthesizer:
    if X is None:
        X = np.random.default_rng(0).uniform(0, 1, (10, n_features))
    features = _make_features(X.shape[1])
    s = TextSynthesizer()
    s.fit(X, features)
    return s


# ---------------------------------------------------------------------------
# fit
# ---------------------------------------------------------------------------

def test_fit_stores_distribution():
    X = np.random.default_rng(0).uniform(0, 1, (10, 4))
    s = _fit(X)
    assert s._mean.shape == (4,)
    assert s._cov.shape == (4, 4)
    assert len(s._features) == 4


def test_fit_returns_self():
    s = TextSynthesizer()
    result = s.fit(np.ones((5, 2)), _make_features(2))
    assert result is s


def test_fit_edge_single_text():
    X = np.array([[0.2, 0.8, 0.5]])
    s = _fit(X)
    assert s._mean.shape == (3,)
    np.testing.assert_array_equal(s._cov, np.eye(3))


def test_fit_edge_empty():
    X = np.empty((0, 3))
    s = _fit(X)
    assert s._mean.shape == (3,)
    np.testing.assert_allclose(s._mean, np.full(3, 0.5))
    np.testing.assert_array_equal(s._cov, np.eye(3))


# ---------------------------------------------------------------------------
# save / load
# ---------------------------------------------------------------------------

def test_save_load_roundtrip(tmp_path: Path):
    X = np.random.default_rng(42).uniform(0, 1, (20, 3))
    features = _make_features(3)
    s = TextSynthesizer()
    s.fit(X, features)
    path = tmp_path / "col.json"
    s.save(path)

    loaded = TextSynthesizer.load(path)
    assert [f.hypothesis for f in loaded._features] == [f.hypothesis for f in features]
    np.testing.assert_allclose(loaded._mean, s._mean)
    np.testing.assert_allclose(loaded._cov, s._cov)


def test_save_creates_valid_json(tmp_path: Path):
    s = _fit()
    path = tmp_path / "col.json"
    s.save(path)
    data = json.loads(path.read_text())
    assert "features" in data
    assert "mean" in data
    assert "cov" in data


def test_load_raises_on_shape_mismatch(tmp_path: Path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({
        "features": [{"hypothesis": "A."}, {"hypothesis": "B."}],
        "mean": [0.5],
        "cov": [[1.0]],
    }))
    with pytest.raises(ValueError, match="shape mismatch"):
        TextSynthesizer.load(path)


# ---------------------------------------------------------------------------
# synthesize
# ---------------------------------------------------------------------------

def _get_user_content(llm: MagicMock, call_index: int = 0) -> str:
    messages = llm.complete.call_args_list[call_index][0][0]
    return messages[1]["content"]


def test_synthesize_returns_n_texts():
    s = _fit()
    X = np.random.default_rng(0).uniform(0, 1, (3, 3))
    llm = _make_llm(["t1", "t2", "t3"])
    results = s.synthesize(X, llm=llm)
    assert results == ["t1", "t2", "t3"]
    assert llm.complete.call_count == 3


def test_synthesize_empty_x_returns_empty():
    s = _fit()
    X = np.empty((0, 3))
    llm = _make_llm([])
    results = s.synthesize(X, llm=llm)
    assert results == []
    llm.complete.assert_not_called()


def test_synthesize_n_levels_none_continuous():
    s = _fit(np.full((5, 2), 0.7))
    X = np.full((1, 2), 0.7)
    llm = _make_llm(["t"])
    s.synthesize(X, llm=llm)
    content = _get_user_content(llm)
    assert re.search(r"→ 0\.\d{2}", content)


def test_synthesize_n_levels_2_yes_no():
    s = _fit(np.full((5, 2), 0.8))
    X = np.full((1, 2), 0.8)
    llm = _make_llm(["t"])
    s.synthesize(X, llm=llm, n_levels=2)
    content = _get_user_content(llm)
    assert "YES" in content or "NO" in content


def test_synthesize_n_levels_3_low_med_high():
    s = _fit(np.full((5, 2), 0.5))
    X = np.full((1, 2), 0.5)
    llm = _make_llm(["t"])
    s.synthesize(X, llm=llm, n_levels=3)
    content = _get_user_content(llm)
    assert any(label in content for label in ("LOW", "MED", "HIGH"))


def test_synthesize_n_levels_4_numeric():
    s = _fit(np.full((5, 1), 0.9))
    X = np.full((1, 1), 0.9)
    llm = _make_llm(["t"])
    s.synthesize(X, llm=llm, n_levels=4)
    content = _get_user_content(llm)
    assert re.search(r"→ \d+", content)


def test_synthesize_threshold_filters_low_features():
    features = [
        Feature(hypothesis="Always low feature."),
        Feature(hypothesis="Always high feature."),
    ]
    X_fit = np.column_stack([np.full(10, 0.1), np.full(10, 0.9)])
    s = TextSynthesizer()
    s.fit(X_fit, features)
    X = np.array([[0.1, 0.9]])
    llm = _make_llm(["t"])
    s.synthesize(X, llm=llm, threshold=0.5)
    content = _get_user_content(llm)
    assert "Always low feature." not in content
    assert "Always high feature." in content


def test_synthesize_threshold_all_filtered_no_crash():
    s = _fit(np.full((5, 2), 0.1))
    X = np.full((1, 2), 0.1)
    llm = _make_llm(["t"])
    results = s.synthesize(X, llm=llm, threshold=0.9)
    assert len(results) == 1
    content = _get_user_content(llm)
    assert "no specific conditions" in content


def test_synthesize_language_in_prompt():
    s = _fit()
    X = np.random.default_rng(0).uniform(0, 1, (1, 3))
    llm = _make_llm(["テキスト"])
    s.synthesize(X, llm=llm, language="Japanese")
    content = _get_user_content(llm)
    assert "Japanese" in content


def test_synthesize_lengths_in_prompt():
    s = _fit()
    X = np.random.default_rng(0).uniform(0, 1, (1, 3))
    llm = _make_llm(["t"])
    s.synthesize(X, llm=llm, lengths=np.array([200]))
    content = _get_user_content(llm)
    assert "approximately" in content
    assert "characters" in content


def test_synthesize_no_lengths_no_length_in_prompt():
    s = _fit()
    X = np.random.default_rng(0).uniform(0, 1, (1, 3))
    llm = _make_llm(["t"])
    s.synthesize(X, llm=llm)
    content = _get_user_content(llm)
    assert "characters" not in content
