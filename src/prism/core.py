from __future__ import annotations

from typing import Any, Literal

import numpy as np

from .discovery import AxisDiscoverer
from .generation import FeatureGenerator
from .llm import BaseLLMClient, LLMClient, LangChainLLMClient
from .merging import AxisMerger
from .models import Axis, AxisLabels, Feature, FeatureMatrix, FittedPredictor, NamedFeature, SelectionResult
from .naming import FeatureNamer
from .nli import NLIModel
from .scoring import NLIScorer
from .selection import FeatureSelector


class Prism:
    """Facade for the full Prism pipeline.

    Example::

        prism = Prism(llm="gpt-4o", nli_model="cross-encoder/nli-deberta-v3-large")
        axes = prism.discover_axes(texts, n=20)
        features = prism.generate_features(texts, axes)
        matrices = prism.score(texts, features, method="nli")
        results = prism.select(matrices)
    """

    def __init__(
        self,
        llm: str | Any = "gpt-4o",
        nli_model: str = "cross-encoder/nli-deberta-v3-large",
        api_key: str | None = None,
        mode: Literal["classification", "regression"] = "regression",
    ) -> None:
        if isinstance(llm, str):
            self._llm: BaseLLMClient = LLMClient(model=llm, api_key=api_key)
        else:
            self._llm = LangChainLLMClient(model=llm)
        self._nli_model = NLIModel(model_name=nli_model)
        self._discoverer = AxisDiscoverer(llm=self._llm, nli_model=self._nli_model)
        self._generator = FeatureGenerator(llm=self._llm)
        self._merger = AxisMerger(nli_model=self._nli_model)
        self._nli_scorer = NLIScorer(model=self._nli_model)
        self._namer = FeatureNamer(llm=self._llm)
        self._mode = mode
        self._selector = FeatureSelector()

    def merge_axes(self, axes_per_run: list[list[Axis]]) -> list[Axis]:
        """Merge axes from multiple discovery runs using LLM-based consolidation."""
        return self._merger.merge(axes_per_run)

    def discover_axes(
        self,
        texts: list[str],
        n: int = 20,
        context_limit: int = 100_000,
        seed: int | None = None,
        language: str | None = None,
    ) -> list[Axis]:
        """Stage 1a: Discover n axes from the text collection.

        Args:
            language: If specified, axes are generated in this language (e.g. "Japanese").
        """
        return self._discoverer.discover(texts, n=n, context_limit=context_limit, seed=seed, language=language)

    def label_axes(
        self,
        texts: list[str],
        axes: list[Axis],
        method: Literal["nli"] = "nli",
        threshold: float = 0.5,
    ) -> list[AxisLabels]:
        """Stage 1b: Label texts for each axis."""
        return self._discoverer.label(texts, axes, method=method, threshold=threshold, mode=self._mode)

    def generate_features(
        self,
        texts: list[str],
        axes: list[Axis],
        n_features: int = 10,
        context_limit: int = 100_000,
        seed: int | None = None,
        axes_labels: list[AxisLabels] | None = None,
        language: str | None = None,
    ) -> dict[Axis, list[Feature]]:
        """Stage 2: Generate discriminative features for each axis.

        Internally labels texts per axis before feature generation.
        If axes_labels is provided, NLI labeling is skipped entirely.

        Args:
            language: If specified, features are generated in this language (e.g. "Japanese").
        """
        if axes_labels is None:
            axis_labels_list = self.label_axes(texts, axes)
        else:
            axis_labels_list = axes_labels
        features_by_axis: dict[Axis, list[Feature]] = {}
        for axis_labels in axis_labels_list:
            features = self._generator.generate(
                texts,
                axis_labels,
                n_features=n_features,
                context_limit=context_limit,
                seed=seed,
                language=language,
            )
            features_by_axis[axis_labels.axis] = features
        return features_by_axis

    def score(
        self,
        texts: list[str],
        features_by_axis: dict[Axis, list[Feature]],
        axes_labels: list[AxisLabels] | None = None,
    ) -> dict[Axis, FeatureMatrix]:
        """Stage 3: Score texts against features, returning one FeatureMatrix per axis.

        Args:
            texts: Full text collection.
            features_by_axis: Output of generate_features().
            axes_labels: Pre-computed axis labels. If None, they are computed internally.
        """
        if axes_labels is None:
            axes = list(features_by_axis.keys())
            axes_labels = self.label_axes(texts, axes)

        labels_map = {al.axis: al for al in axes_labels}

        matrices: dict[Axis, FeatureMatrix] = {}
        for axis, features in features_by_axis.items():
            feature_scores = self._nli_scorer.score(texts, features)
            X = np.column_stack([fs.scores for fs in feature_scores])
            y = np.array(labels_map[axis].labels, dtype=float)
            matrices[axis] = FeatureMatrix(axis=axis, features=features, X=X, y=y, mode=self._mode)

        return matrices

    def select(
        self, matrices: dict[Axis, FeatureMatrix]
    ) -> tuple[dict[Axis, SelectionResult], dict[Axis, FittedPredictor]]:
        """Stage 4: Select predictive features per axis using L1-regularized SGD.

        Returns:
            Tuple of (results, predictors), both keyed by Axis.
        """
        results: dict[Axis, SelectionResult] = {}
        predictors: dict[Axis, FittedPredictor] = {}
        for matrix in matrices.values():
            result, predictor = self._selector.select(matrix)
            results[matrix.axis] = result
            predictors[matrix.axis] = predictor
        return results, predictors

    def name_features(
        self,
        features_by_axis: dict[Axis, list[Feature]],
        language: str | None = None,
    ) -> dict[Axis, list[NamedFeature]]:
        """Assign human-readable names to finalized features (display layer).

        Typically called after select() on the selected features.

        Args:
            features_by_axis: Features to name, keyed by axis.
            language: If specified, generate names in this language.

        Returns:
            Named features keyed by axis.
        """
        return {
            axis: self._namer.name_features(features, language=language)
            for axis, features in features_by_axis.items()
        }
