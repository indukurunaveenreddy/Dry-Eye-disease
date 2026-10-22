"""Symptom-based models: classical baselines plus a regularised MLP."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .. import config

SEED = config.RANDOM_STATE


def _preprocessor(scale: bool = True) -> Pipeline:
    steps: list[tuple[str, Any]] = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scale", StandardScaler()))
    return Pipeline(steps)


def build_classical_models(seed: int = SEED) -> dict[str, Pipeline]:
    """Interpretable baselines for the survey-derived features."""
    return {
        "logistic_regression": Pipeline(
            [
                ("prep", _preprocessor()),
                ("clf", LogisticRegression(max_iter=5000, C=1.0, random_state=seed)),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("prep", _preprocessor(scale=False)),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=500,
                        max_depth=None,
                        min_samples_leaf=2,
                        class_weight="balanced",
                        random_state=seed,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "hist_gradient_boosting": Pipeline(
            [
                ("prep", _preprocessor(scale=False)),
                (
                    "clf",
                    HistGradientBoostingClassifier(
                        max_iter=300,
                        learning_rate=0.06,
                        max_leaf_nodes=15,
                        l2_regularization=1.0,
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "svm_rbf": Pipeline(
            [
                ("prep", _preprocessor()),
                (
                    "clf",
                    SVC(
                        kernel="rbf",
                        C=3.0,
                        gamma="scale",
                        probability=True,
                        class_weight="balanced",
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "mlp_sklearn": Pipeline(
            [
                ("prep", _preprocessor()),
                (
                    "clf",
                    MLPClassifier(
                        hidden_layer_sizes=(64, 32),
                        activation="relu",
                        alpha=1e-2,
                        learning_rate_init=1e-3,
                        max_iter=1200,
                        early_stopping=True,
                        n_iter_no_change=30,
                        random_state=seed,
                    ),
                ),
            ]
        ),
    }


def build_keras_mlp(
    n_features: int,
    n_classes: int,
    seed: int = SEED,
    hidden: tuple[int, ...] = (96, 48, 24),
    dropout: float = 0.35,
) -> Any:
    """Regularised dense network used as the headline symptom classifier.

    Imported lazily so the pure-sklearn path stays usable without TensorFlow.
    """
    from tensorflow import keras
    from tensorflow.keras import layers

    keras.utils.set_random_seed(seed)

    inputs = keras.Input(shape=(n_features,), name="features")
    x = inputs
    for width in hidden:
        x = layers.BatchNormalization()(x)
        x = layers.Dense(width, activation="relu")(x)
        x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(n_classes, activation="softmax", name="probability")(x)

    model = keras.Model(inputs, outputs, name="ded_symptom_mlp")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def keras_mlp_sklearn_wrapper(
    n_features: int,
    n_classes: int,
    seed: int = SEED,
    max_epochs: int = 300,
) -> Pipeline:
    """Expose the Keras MLP through the sklearn estimator API.

    Uses the sklearn wrapper so the same cross-validation harness, imputer and
    scaler apply to every model in the comparison table.
    """

    return Pipeline(
        [
            ("prep", _preprocessor()),
            ("clf", KerasClassifierWrapper(n_features, n_classes, seed, max_epochs)),
        ]
    )


class KerasClassifierWrapper:
    """Minimal sklearn-compatible wrapper around a Keras model.

    Every constructor argument is reported by ``get_params`` and accepted by
    ``__init__`` so scikit-learn's ``clone`` can rebuild it inside
    cross-validation. TensorFlow is imported lazily inside ``fit`` rather than
    held on the instance, because a module handle cannot be cloned.
    """

    def __init__(
        self,
        n_features: int,
        n_classes: int,
        seed: int = SEED,
        max_epochs: int = 200,
    ) -> None:
        self.n_features = n_features
        self.n_classes = n_classes
        self.seed = seed
        self.max_epochs = max_epochs
        self.model_: Any = None
        self.classes_ = np.arange(n_classes)

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {
            "n_features": self.n_features,
            "n_classes": self.n_classes,
            "seed": self.seed,
            "max_epochs": self.max_epochs,
        }

    def set_params(self, **params: Any) -> KerasClassifierWrapper:
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def fit(self, features: np.ndarray, targets: np.ndarray) -> KerasClassifierWrapper:
        from tensorflow import keras

        model = build_keras_mlp(self.n_features, self.n_classes, self.seed)
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=25, restore_best_weights=True, verbose=0
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=10, min_lr=1e-5, verbose=0
            ),
        ]
        model.fit(
            np.asarray(features, dtype="float32"),
            np.asarray(targets, dtype="int32"),
            validation_split=0.2,
            epochs=self.max_epochs,
            batch_size=32,
            callbacks=callbacks,
            verbose=0,
        )
        self.model_ = model
        return self

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        probabilities = self.model_.predict(
            np.asarray(features, dtype="float32"), verbose=0
        )
        return np.asarray(probabilities)

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self.predict_proba(features).argmax(axis=1)
