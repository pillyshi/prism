from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import joblib

from .models import Axis, AxisLabels, Feature, FittedPredictor, SelectionResult


def _sanitize_filename(name: str) -> str:
    """Convert an axis name to a safe filename."""
    return re.sub(r"[^\w\-]", "_", name).strip("_")


# --- Serialization helpers ---

def _axis_to_dict(axis: Axis) -> dict[str, str]:
    return {"name": axis.name, "question": axis.question, "hypothesis": axis.hypothesis}


def _feature_to_dict(feature: Feature) -> dict[str, Any]:
    return {
        "name": feature.name,
        "question": feature.question,
        "hypothesis": feature.hypothesis,
        "axis": _axis_to_dict(feature.axis),
    }


def _axis_from_dict(d: dict[str, str]) -> Axis:
    return Axis(name=d["name"], question=d["question"], hypothesis=d["hypothesis"])


def _feature_from_dict(d: dict[str, Any]) -> Feature:
    return Feature(
        name=d["name"],
        question=d["question"],
        hypothesis=d["hypothesis"],
        axis=_axis_from_dict(d["axis"]),
    )


# --- Public API ---

def save_session(
    output_dir: str | Path,
    axes: list[Axis],
    features_by_axis: dict[Axis, list[Feature]],
    results: dict[Axis, SelectionResult],
    predictors: dict[Axis, FittedPredictor],
    axis_labels: list[AxisLabels] | None = None,
) -> None:
    """Save a Prism session to disk.

    Creates:
      {output_dir}/session.json          — human-readable JSON
      {output_dir}/predictors/*.joblib   — one file per axis
    """
    output_dir = Path(output_dir)
    predictors_dir = output_dir / "predictors"
    output_dir.mkdir(parents=True, exist_ok=True)
    predictors_dir.mkdir(exist_ok=True)

    # Build JSON payload
    session: dict[str, Any] = {
        "axes": [_axis_to_dict(a) for a in axes],
        "features_by_axis": {
            a.name: [_feature_to_dict(f) for f in feats]
            for a, feats in features_by_axis.items()
        },
        "axis_labels": (
            [
                {"axis": _axis_to_dict(al.axis), "labels": al.labels}
                for al in axis_labels
            ]
            if axis_labels is not None
            else None
        ),
        "selection_results": [
            {
                "axis": _axis_to_dict(r.axis),
                "selected_features": [_feature_to_dict(f) for f in r.selected_features],
                "coef": r.coef,
                "cv_score": r.cv_score,
                "cv_scoring": r.cv_scoring,
            }
            for r in results.values()
        ],
    }

    with open(output_dir / "session.json", "w", encoding="utf-8") as f:
        json.dump(session, f, ensure_ascii=False, indent=2)

    # Save predictors
    for axis, predictor in predictors.items():
        filename = _sanitize_filename(axis.name) + ".joblib"
        joblib.dump(predictor, predictors_dir / filename)


def load_session(output_dir: str | Path) -> dict[str, Any]:
    """Load a Prism session from disk.

    Returns a dict with keys:
      axes              : list[Axis]
      features_by_axis  : dict[Axis, list[Feature]]
      axis_labels       : list[AxisLabels] | None
      results           : dict[Axis, SelectionResult]
      predictors        : dict[Axis, FittedPredictor]
    """
    output_dir = Path(output_dir)

    with open(output_dir / "session.json", encoding="utf-8") as f:
        session = json.load(f)

    axes = [_axis_from_dict(d) for d in session["axes"]]
    axis_by_name = {a.name: a for a in axes}

    features_by_axis: dict[Axis, list[Feature]] = {
        axis_by_name[axis_name]: [_feature_from_dict(fd) for fd in feat_list]
        for axis_name, feat_list in session["features_by_axis"].items()
    }

    axis_labels: list[AxisLabels] | None = None
    if session.get("axis_labels") is not None:
        axis_labels = [
            AxisLabels(axis=_axis_from_dict(d["axis"]), labels=d["labels"])
            for d in session["axis_labels"]
        ]

    results: dict[Axis, SelectionResult] = {}
    for d in session["selection_results"]:
        axis = _axis_from_dict(d["axis"])
        result = SelectionResult(
            axis=axis,
            selected_features=[_feature_from_dict(fd) for fd in d["selected_features"]],
            coef=d["coef"],
            cv_score=d.get("cv_score", 0.0),
            cv_scoring=d.get("cv_scoring", ""),
        )
        results[axis] = result

    # Load predictors
    predictors: dict[Axis, FittedPredictor] = {}
    predictors_dir = output_dir / "predictors"
    for axis in axes:
        filename = _sanitize_filename(axis.name) + ".joblib"
        path = predictors_dir / filename
        if path.exists():
            predictor = joblib.load(path)
            predictors[axis] = predictor

    return {
        "axes": axes,
        "features_by_axis": features_by_axis,
        "axis_labels": axis_labels,
        "results": results,
        "predictors": predictors,
    }
