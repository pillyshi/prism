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


def _fit(n_features: int = 3) -> TextSynthesizer:
    return TextSynthesizer().fit(_make_features(n_features))


# ---------------------------------------------------------------------------
# fit
# ---------------------------------------------------------------------------

def test_fit_returns_self():
    s = TextSynthesizer()
    result = s.fit(_make_features(2))
    assert result is s


def test_fit_stores_features():
    features = _make_features(4)
    s = TextSynthesizer().fit(features)
    assert len(s._features) == 4
    assert not s._has_length


def test_fit_with_lengths_sets_has_length():
    lengths = np.array([100, 200, 300])
    s = TextSynthesizer().fit(_make_features(2), lengths=lengths)
    assert s._has_length
    assert s._log_length_mean == pytest.approx(np.mean(np.log(lengths)))
    assert s._log_length_std == pytest.approx(np.std(np.log(lengths)))


def test_fit_with_empty_lengths_no_has_length():
    s = TextSynthesizer().fit(_make_features(2), lengths=np.array([]))
    assert not s._has_length


def test_fit_with_single_length():
    s = TextSynthesizer().fit(_make_features(2), lengths=np.array([200]))
    assert s._has_length
    assert s._log_length_std == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# save / load
# ---------------------------------------------------------------------------

def test_save_load_roundtrip(tmp_path: Path):
    features = _make_features(3)
    lengths = np.array([100, 200, 150, 300])
    s = TextSynthesizer().fit(features, lengths=lengths)
    path = tmp_path / "synth.json"
    s.save(path)

    loaded = TextSynthesizer.load(path)
    assert [f.hypothesis for f in loaded._features] == [f.hypothesis for f in features]
    assert loaded._has_length == s._has_length
    assert loaded._log_length_mean == pytest.approx(s._log_length_mean)
    assert loaded._log_length_std == pytest.approx(s._log_length_std)


def test_save_creates_valid_json(tmp_path: Path):
    s = _fit()
    path = tmp_path / "synth.json"
    s.save(path)
    data = json.loads(path.read_text())
    assert "features" in data
    assert "has_length" in data
    assert "log_length_mean" in data
    assert "log_length_std" in data


def test_save_load_roundtrip_no_lengths(tmp_path: Path):
    features = _make_features(3)
    s = TextSynthesizer().fit(features)
    path = tmp_path / "synth.json"
    s.save(path)

    loaded = TextSynthesizer.load(path)
    assert not loaded._has_length


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
    s = _fit(2)
    X = np.full((1, 2), 0.7)
    llm = _make_llm(["t"])
    s.synthesize(X, llm=llm)
    content = _get_user_content(llm)
    assert re.search(r"→ 0\.\d{2}", content)


def test_synthesize_n_levels_2_yes_no():
    s = _fit(2)
    X = np.full((1, 2), 0.8)
    llm = _make_llm(["t"])
    s.synthesize(X, llm=llm, n_levels=2)
    content = _get_user_content(llm)
    assert "YES" in content or "NO" in content


def test_synthesize_n_levels_3_low_med_high():
    s = _fit(2)
    X = np.full((1, 2), 0.5)
    llm = _make_llm(["t"])
    s.synthesize(X, llm=llm, n_levels=3)
    content = _get_user_content(llm)
    assert any(label in content for label in ("LOW", "MED", "HIGH"))


def test_synthesize_n_levels_4_numeric():
    s = _fit(1)
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
    s = TextSynthesizer().fit(features)
    X = np.array([[0.1, 0.9]])
    llm = _make_llm(["t"])
    s.synthesize(X, llm=llm, threshold=0.5)
    content = _get_user_content(llm)
    assert "Always low feature." not in content
    assert "Always high feature." in content


def test_synthesize_threshold_all_filtered_no_crash():
    s = _fit(2)
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


def test_fit_with_lengths_samples_length_in_prompt():
    lengths = np.full(10, 200)
    s = TextSynthesizer().fit(_make_features(2), lengths=lengths)
    X = np.full((1, 2), 0.5)
    llm = _make_llm(["t"])
    s.synthesize(X, llm=llm, seed=0)
    content = _get_user_content(llm)
    assert "approximately" in content
    assert "characters" in content


def test_fit_without_lengths_no_length_in_prompt():
    s = _fit()
    X = np.random.default_rng(0).uniform(0, 1, (1, 3))
    llm = _make_llm(["t"])
    s.synthesize(X, llm=llm)
    content = _get_user_content(llm)
    assert "characters" not in content


def test_synthesize_explicit_lengths_override():
    lengths_fit = np.full(10, 200)
    s = TextSynthesizer().fit(_make_features(2), lengths=lengths_fit)
    X = np.full((1, 2), 0.5)
    llm = _make_llm(["t"])
    s.synthesize(X, llm=llm, lengths=np.array([999]))
    content = _get_user_content(llm)
    assert "999" in content


def test_synthesize_seed_produces_same_lengths():
    lengths = np.array([100, 200, 300, 400, 500])
    s = TextSynthesizer().fit(_make_features(2), lengths=lengths)
    X = np.full((3, 2), 0.5)
    llm1 = _make_llm(["a", "b", "c"])
    llm2 = _make_llm(["a", "b", "c"])
    s.synthesize(X, llm=llm1, seed=42)
    s.synthesize(X, llm=llm2, seed=42)
    contents1 = [_get_user_content(llm1, i) for i in range(3)]
    contents2 = [_get_user_content(llm2, i) for i in range(3)]
    assert contents1 == contents2
