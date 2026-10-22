"""Shared evaluation helpers and artifact serialisation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def to_builtin(value: Any) -> Any:
    """Convert numpy/pandas scalars into JSON-safe Python objects."""
    if isinstance(value, dict):
        return {str(key): to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return to_builtin(value.tolist())
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[int] | None = None,
    class_names: list[str] | None = None,
) -> dict[str, Any]:
    """Accuracy, macro F1 and a full per-class report for one split."""
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": matrix,
        "labels": labels,
        "class_names": class_names,
        "report": report,
    }


def binary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray | None = None,
) -> dict[str, Any]:
    """Metrics for the two-class dry-eye versus healthy task."""
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }
    if y_score is not None and len(np.unique(y_true)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
    return metrics


def cross_validate(
    estimator,
    features: np.ndarray,
    targets: np.ndarray,
    n_splits: int = 5,
    n_repeats: int = 1,
    random_state: int = 42,
    scoring: str = "accuracy",
) -> dict[str, Any]:
    """Repeated stratified cross-validation with out-of-fold predictions."""
    from sklearn.base import clone
    from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold

    if n_repeats > 1:
        splitter: Any = RepeatedStratifiedKFold(
            n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
        )
        n_rounds = n_splits * n_repeats
    else:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        n_rounds = n_splits

    n_classes = int(targets.max()) + 1
    oof_sum = np.zeros((n_rounds, len(targets), n_classes), dtype=np.float64)
    oof_count = np.zeros((n_rounds, len(targets)), dtype=np.float64)
    round_test_indices: list[np.ndarray] = []

    for fold_index, (train_index, test_index) in enumerate(splitter.split(features, targets)):
        round_test_indices.append(test_index)
        model = clone(estimator)
        if hasattr(model, "random_state"):
            model.random_state = random_state + fold_index
        model.fit(features[train_index], targets[train_index])

        if hasattr(model, "predict_proba"):
            fold_probabilities = np.asarray(model.predict_proba(features[test_index]))
        else:
            fold_probabilities = np.zeros((len(test_index), n_classes))
            predictions = np.asarray(model.predict(features[test_index]), dtype=int)
            fold_probabilities[np.arange(len(test_index)), predictions] = 1.0

        oof_sum[fold_index, test_index] = fold_probabilities
        oof_count[fold_index, test_index] = 1.0

    averaged = oof_sum.sum(axis=0) / np.maximum(
        oof_count.sum(axis=0)[:, None], 1.0
    )
    predicted = averaged.argmax(axis=1)
    scores = [scoring] if isinstance(scoring, str) else list(scoring)

    per_round: dict[str, list[float]] = {}
    for round_index in range(n_rounds):
        covered = round_test_indices[round_index]
        round_pred = oof_sum[round_index, covered, :].argmax(axis=1)
        round_true = targets[covered]
        for score in scores:
            per_round.setdefault(score, []).append(
                classification_metrics(round_true, round_pred)[score]
            )

    summary = {score: float(np.mean(values)) for score, values in per_round.items()}
    summary["std"] = {score: float(np.std(values)) for score, values in per_round.items()}

    return {
        "cv": {
            "n_splits": n_splits,
            "n_repeats": n_repeats,
            "scoring": scores,
            "summary": summary,
        },
        "oof_probabilities": averaged,
        "oof_predictions": predicted,
        "oof_metrics": classification_metrics(targets, predicted),
    }


def save_json(payload: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(to_builtin(payload), indent=2), encoding="utf-8")


def load_json(source: Path) -> dict[str, Any]:
    return json.loads(Path(source).read_text(encoding="utf-8"))


def summarise_dataframe(frame: pd.DataFrame) -> pd.DataFrame:
    """Numeric describe block used by the EDA report."""
    return frame.describe(include=[np.number]).T.round(3)
