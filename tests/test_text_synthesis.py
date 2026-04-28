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
# sample — count and basic behavior
# ---------------------------------------------------------------------------

def test_sample_returns_n_texts():
    s = _fit()
    llm = _make_llm(["t1", "t2", "t3"])
    results = s.sample(3, llm=llm, rng=np.random.default_rng(0))
    assert results == ["t1", "t2", "t3"]
    assert llm.complete.call_count == 3


def test_sample_n_zero_returns_empty():
    s = _fit()
    llm = _make_llm([])
    results = s.sample(0, llm=llm)
    assert results == []
    llm.complete.assert_not_called()


# ---------------------------------------------------------------------------
# sample — n_levels discretization
# ---------------------------------------------------------------------------

def _get_user_content(llm: MagicMock, call_index: int = 0) -> str:
    messages = llm.complete.call_args_list[call_index][0][0]
    return messages[1]["content"]


def test_sample_n_levels_none_continuous():
    s = _fit(np.full((5, 2), 0.7))
    llm = _make_llm(["t"])
    s.sample(1, llm=llm, rng=np.random.default_rng(0))
    content = _get_user_content(llm)
    assert re.search(r"→ 0\.\d{2}", content)


def test_sample_n_levels_2_yes_no():
    s = _fit(np.full((5, 2), 0.8))
    llm = _make_llm(["t"])
    s.sample(1, llm=llm, n_levels=2, rng=np.random.default_rng(0))
    content = _get_user_content(llm)
    assert "YES" in content or "NO" in content


def test_sample_n_levels_3_low_med_high():
    s = _fit(np.full((5, 2), 0.5))
    llm = _make_llm(["t"])
    s.sample(1, llm=llm, n_levels=3, rng=np.random.default_rng(0))
    content = _get_user_content(llm)
    assert any(label in content for label in ("LOW", "MED", "HIGH"))


def test_sample_n_levels_4_numeric():
    s = _fit(np.full((5, 1), 0.9))
    llm = _make_llm(["t"])
    s.sample(1, llm=llm, n_levels=4, rng=np.random.default_rng(0))
    content = _get_user_content(llm)
    assert re.search(r"→ \d+", content)


# ---------------------------------------------------------------------------
# sample — threshold filtering
# ---------------------------------------------------------------------------

def test_sample_threshold_filters_low_features():
    features = [
        Feature(hypothesis="Always low feature."),
        Feature(hypothesis="Always high feature."),
    ]
    X = np.column_stack([np.full(10, 0.1), np.full(10, 0.9)])
    s = TextSynthesizer()
    s.fit(X, features)
    s._mean = np.array([0.1, 0.9])
    s._cov = np.eye(2) * 1e-9

    llm = _make_llm(["t"])
    s.sample(1, llm=llm, threshold=0.5, rng=np.random.default_rng(0))
    content = _get_user_content(llm)
    assert "Always low feature." not in content
    assert "Always high feature." in content


def test_sample_threshold_all_filtered_no_crash():
    s = _fit(np.full((5, 2), 0.1))
    s._mean = np.array([0.1, 0.1])
    s._cov = np.eye(2) * 1e-9
    llm = _make_llm(["t"])
    results = s.sample(1, llm=llm, threshold=0.9, rng=np.random.default_rng(0))
    assert len(results) == 1
    content = _get_user_content(llm)
    assert "no specific conditions" in content


# ---------------------------------------------------------------------------
# sample — language
# ---------------------------------------------------------------------------

def test_sample_language_in_prompt():
    s = _fit()
    llm = _make_llm(["テキスト"])
    s.sample(1, llm=llm, language="Japanese", rng=np.random.default_rng(0))
    content = _get_user_content(llm)
    assert "Japanese" in content


# ---------------------------------------------------------------------------
# fit / sample — lengths
# ---------------------------------------------------------------------------

def test_fit_with_lengths_stores_joint():
    X = np.random.default_rng(0).uniform(0, 1, (10, 3))
    lengths = np.array([100, 200, 150, 300, 250, 180, 120, 90, 400, 220])
    s = TextSynthesizer()
    s.fit(X, _make_features(3), lengths=lengths)
    assert s._has_length is True
    assert s._mean.shape == (4,)
    assert s._cov.shape == (4, 4)
    assert s._mean[-1] == pytest.approx(np.log(lengths.astype(float)).mean())


def test_fit_without_lengths():
    s = _fit()
    assert s._has_length is False
    assert s._mean.shape == (3,)


def test_fit_single_length_uses_identity_cov():
    X = np.array([[0.5, 0.5]])
    s = TextSynthesizer()
    s.fit(X, _make_features(2), lengths=np.array([200]))
    assert s._has_length is True
    assert s._mean.shape == (3,)
    np.testing.assert_array_equal(s._cov, np.eye(3))


def test_sample_length_in_prompt_when_fitted():
    X = np.full((5, 2), 0.5)
    lengths = np.full(5, 200)
    s = TextSynthesizer()
    s.fit(X, _make_features(2), lengths=lengths)
    llm = _make_llm(["t"])
    s.sample(1, llm=llm, rng=np.random.default_rng(0))
    content = _get_user_content(llm)
    assert "approximately" in content
    assert "characters" in content


def test_sample_no_length_in_prompt_when_not_fitted():
    s = _fit()
    llm = _make_llm(["t"])
    s.sample(1, llm=llm, rng=np.random.default_rng(0))
    content = _get_user_content(llm)
    assert "characters" not in content


def test_save_load_roundtrip_with_lengths(tmp_path):
    X = np.random.default_rng(0).uniform(0, 1, (10, 3))
    lengths = np.array([100, 200, 150, 300, 250, 180, 120, 90, 400, 220])
    s = TextSynthesizer()
    s.fit(X, _make_features(3), lengths=lengths)
    path = tmp_path / "col.json"
    s.save(path)

    loaded = TextSynthesizer.load(path)
    assert loaded._has_length == s._has_length
    np.testing.assert_allclose(loaded._mean, s._mean)
    np.testing.assert_allclose(loaded._cov, s._cov)


def test_load_old_format_without_lengths(tmp_path):
    """Files saved before length support must load without error."""
    path = tmp_path / "old.json"
    path.write_text(json.dumps({
        "features": [{"hypothesis": "A."}],
        "mean": [0.5],
        "cov": [[1.0]],
    }))
    loaded = TextSynthesizer.load(path)
    assert loaded._has_length is False


# ---------------------------------------------------------------------------
# sample_with_vectors
# ---------------------------------------------------------------------------

def test_sample_with_vectors_return_type():
    s = _fit(n_features=3)
    llm = _make_llm(["t1", "t2"])
    texts, X_sampled = s.sample_with_vectors(2, llm=llm, rng=np.random.default_rng(0))
    assert isinstance(texts, list)
    assert isinstance(X_sampled, np.ndarray)


def test_sample_with_vectors_matrix_shape():
    s = _fit(n_features=3)
    llm = _make_llm(["t"] * 5)
    _, X_sampled = s.sample_with_vectors(5, llm=llm, rng=np.random.default_rng(0))
    assert X_sampled.shape == (5, 3)


def test_sample_with_vectors_matrix_clipped():
    s = _fit()
    llm = _make_llm(["t"] * 10)
    _, X_sampled = s.sample_with_vectors(10, llm=llm, rng=np.random.default_rng(0))
    assert np.all(X_sampled >= 0.0)
    assert np.all(X_sampled <= 1.0)


def test_sample_with_vectors_reproducible():
    s = _fit()
    responses = ["a", "b", "c"]
    texts1, X1 = s.sample_with_vectors(3, llm=_make_llm(responses), rng=np.random.default_rng(99))
    texts2, X2 = s.sample_with_vectors(3, llm=_make_llm(responses), rng=np.random.default_rng(99))
    assert texts1 == texts2
    np.testing.assert_array_equal(X1, X2)


def test_sample_with_vectors_n_zero():
    s = _fit(n_features=3)
    llm = _make_llm([])
    texts, X_sampled = s.sample_with_vectors(0, llm=llm)
    assert texts == []
    assert X_sampled.shape == (0, 3)


def test_sample_with_vectors_zero_features():
    X = np.empty((5, 0))
    s = TextSynthesizer()
    s.fit(X, [])
    llm = _make_llm(["t"] * 3)
    texts, X_sampled = s.sample_with_vectors(3, llm=llm, rng=np.random.default_rng(0))
    assert len(texts) == 3
    assert X_sampled.shape == (3, 0)
