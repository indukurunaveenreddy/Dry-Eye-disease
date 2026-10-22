"""Tests for the cross-validation harness and metric helpers.

The harness is compared against scikit-learn's own ``cross_val_score`` so a
regression in the out-of-fold bookkeeping cannot pass silently.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from ded import metrics


@pytest.fixture
def dataset() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    n = 160
    features = rng.normal(size=(n, 4))
    targets = (features[:, 0] + features[:, 1] * 0.5 > 0).astype(int)
    return features, targets


def _model(seed: int = 0) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=5000, random_state=seed)),
        ]
    )


def test_single_repeat_matches_sklearn(dataset) -> None:
    features, targets = dataset
    reference = cross_val_score(
        _model(),
        features,
        targets,
        cv=StratifiedKFold(5, shuffle=True, random_state=42),
        scoring="accuracy",
    ).mean()

    result = metrics.cross_validate(
        _model(), features, targets, n_splits=5, n_repeats=1, random_state=42
    )

    assert result["cv"]["summary"]["accuracy"] == pytest.approx(reference, abs=1e-9)


def test_repeats_do_not_collapse_predictions(dataset) -> None:
    """Guards the bug where NaN folds defaulted every row to class 0."""
    features, targets = dataset

    result = metrics.cross_validate(
        _model(), features, targets, n_splits=5, n_repeats=4, random_state=42
    )
    per_round = result["cv"]["summary"]["accuracy"]

    assert per_round > 0.85
    assert result["oof_metrics"]["accuracy"] > 0.85
    assert set(result["cv"]["summary"]) == {"accuracy", "std"}
    assert result["cv"]["summary"]["std"]["accuracy"] >= 0.0


def test_repeated_cv_reports_every_fold(dataset) -> None:
    features, targets = dataset
    result = metrics.cross_validate(
        _model(), features, targets, n_splits=5, n_repeats=3, random_state=0
    )
    assert result["cv"]["n_splits"] == 5
    assert result["cv"]["n_repeats"] == 3
    assert result["cv"]["summary"]["accuracy"] > 0.85


def test_out_of_fold_probabilities_sum_to_one(dataset) -> None:
    features, targets = dataset
    result = metrics.cross_validate(_model(), features, targets, n_splits=5, random_state=1)
    totals = result["oof_probabilities"].sum(axis=1)
    assert np.allclose(totals, 1.0, atol=1e-6)


def test_out_of_fold_covers_every_sample(dataset) -> None:
    features, targets = dataset
    result = metrics.cross_validate(_model(), features, targets, n_splits=5, n_repeats=2)
    assert len(result["oof_predictions"]) == len(targets)


def test_classification_metrics_perfect_prediction() -> None:
    targets = np.array([0, 0, 1, 1])
    result = metrics.classification_metrics(targets, targets, labels=[0, 1])
    assert result["accuracy"] == 1.0
    assert result["macro_f1"] == 1.0
    assert result["confusion_matrix"] == [[2, 0], [0, 2]]


def test_classification_metrics_single_class_prediction() -> None:
    targets = np.array([0, 0, 1, 1])
    result = metrics.classification_metrics(targets, np.zeros(4, dtype=int), labels=[0, 1])
    assert result["balanced_accuracy"] == pytest.approx(0.5)


def test_binary_metrics_includes_roc_auc(dataset) -> None:
    features, targets = dataset
    scores = np.where(targets == 1, 0.9, 0.1)
    counts = [int((targets == value).sum()) for value in (0, 1)]
    result = metrics.binary_metrics(targets, targets, y_score=scores)
    assert result["roc_auc"] == 1.0
    assert result["confusion_matrix"] == [[counts[0], 0], [0, counts[1]]]


def test_to_builtin_handles_numpy_scalars() -> None:
    payload = {"a": np.int64(3), "b": np.float64(1.5), "c": np.bool_(True)}
    converted = metrics.to_builtin(payload)
    assert converted == {"a": 3, "b": 1.5, "c": True}
    assert isinstance(converted["a"], int)


def test_to_builtin_converts_nested_nan() -> None:
    converted = metrics.to_builtin({"x": [np.array([1.0, np.nan]), float("nan")]})
    assert converted["x"][0] == [1.0, None]


def test_save_and_load_json_roundtrip(tmp_path) -> None:
    destination = tmp_path / "nested" / "payload.json"
    metrics.save_json({"model": "rf", "score": np.float64(0.5), "n": np.int64(7)}, destination)
    loaded = metrics.load_json(destination)
    assert loaded == {"model": "rf", "score": 0.5, "n": 7}


def test_cross_validate_handles_estimators_without_predict_proba() -> None:
    """SVC(probability=False) has no predict_proba; folds must still score."""
    rng = np.random.default_rng(3)
    features = rng.normal(size=(60, 3))
    targets = (features[:, 0] > 0).astype(int)
    model = SVC(kernel="linear", random_state=0)

    result = metrics.cross_validate(model, features, targets, n_splits=3, random_state=0)

    assert result["oof_metrics"]["accuracy"] > 0.9
    assert set(np.unique(result["oof_predictions"])) <= {0, 1}


def test_forest_baseline_separates_well(dataset) -> None:
    features, targets = dataset
    model = RandomForestClassifier(n_estimators=50, random_state=0)
    result = metrics.cross_validate(model, features, targets, n_splits=5, random_state=0)
    assert result["oof_metrics"]["accuracy"] > 0.9
