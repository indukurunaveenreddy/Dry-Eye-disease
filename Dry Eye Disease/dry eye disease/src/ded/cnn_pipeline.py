"""Shared training and evaluation loop for the image classification models."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.metrics import classification_report

from . import config, metrics


@dataclass
class TrainingOutcome:
    """Everything a trained CNN produces, ready to serialise."""

    model: Any
    history: dict[str, list[float]]
    train_metrics: dict[str, Any]
    test_metrics: dict[str, Any]
    counts: dict[str, int]
    seconds: float
    class_names: list[str]
    extra: dict[str, Any] = field(default_factory=dict)


def keras_callbacks(model: Any, patience: int = 8) -> list[Any]:
    """Early stopping plus plateau learning-rate decay."""
    from tensorflow import keras

    return [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=0,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=max(2, patience // 2),
            min_lr=1e-6,
            verbose=0,
        ),
    ]


def train_cnn(
    model: Any,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    class_names: list[str],
    class_weight: dict[int, float] | None,
    epochs: int = 20,
    batch_size: int = 32,
    validation_split: float = 0.15,
    patience: int = 8,
    seed: int = config.RANDOM_STATE,
    verbose: int = 2,
) -> TrainingOutcome:
    """Fit a Keras classifier and score it on the held-out split."""
    callbacks = keras_callbacks(model, patience=patience)
    started = time.time()

    history = model.fit(
        x_train,
        y_train,
        validation_split=validation_split,
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=verbose,
    )
    elapsed = time.time() - started

    return TrainingOutcome(
        model=model,
        history={key: [float(value) for value in values] for key, values in history.history.items()},
        train_metrics=_score(model, x_train, y_train, class_names),
        test_metrics=_score(model, x_test, y_test, class_names),
        counts={
            name: int((y_train == index).sum()) for index, name in enumerate(class_names)
        },
        seconds=round(elapsed, 1),
        class_names=class_names,
    )


def _score(model: Any, features: np.ndarray, targets: np.ndarray, class_names: list[str]) -> dict[str, Any]:
    probabilities = model.predict(features, verbose=0)
    predicted = np.asarray(probabilities).argmax(axis=1)
    result = metrics.classification_metrics(
        targets,
        predicted,
        labels=list(range(len(class_names))),
        class_names=class_names,
    )
    result["mean_confidence"] = float(np.asarray(probabilities).max(axis=1).mean())
    result["report_text"] = classification_report(
        targets, predicted, target_names=class_names, zero_division=0
    )
    return result


def outcome_payload(outcome: TrainingOutcome, task: str, **extra: Any) -> dict[str, Any]:
    """Bundle an outcome plus arbitrary extras into a JSON-safe payload."""
    payload = {
        "task": task,
        "class_names": outcome.class_names,
        "train_counts": outcome.counts,
        "training_seconds": outcome.seconds,
        "epochs_run": len(outcome.history.get("loss", [])),
        "test": outcome.test_metrics,
        "train": outcome.train_metrics,
        "history": outcome.history,
    }
    payload.update(extra)
    return metrics.to_builtin(payload)


def augment(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Light augmentation: brightness jitter plus a small translation."""
    out = image
    brightness = rng.uniform(0.85, 1.15)
    out = np.clip(out * brightness, 0.0, 1.0)

    shift_y, shift_x = rng.integers(-3, 4, size=2)
    out = np.roll(out, (shift_y, shift_x), axis=(0, 1))
    if shift_y > 0:
        out[:shift_y] = 0
    elif shift_y < 0:
        out[shift_y:] = 0
    if shift_x > 0:
        out[:, :shift_x] = 0
    elif shift_x < 0:
        out[:, shift_x:] = 0
    return out


def augmented_copy(
    features: np.ndarray,
    targets: np.ndarray,
    rng: np.random.Generator,
    factor: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Augment the training set, preserving each sample's class label.

    Returns the concatenated features and the matching label vector, where
    every augmented image inherits the label of the sample it was derived
    from.
    """
    extra = int(round(len(features) * factor))
    if extra <= 0:
        return features, targets

    picks = rng.integers(0, len(features), size=extra)
    produced = np.stack([augment(features[index], rng) for index in picks])
    features_out = np.concatenate([features, produced], axis=0).astype(np.float32)
    targets_out = np.concatenate([targets, targets[picks]]).astype(np.int32)
    return features_out, targets_out
