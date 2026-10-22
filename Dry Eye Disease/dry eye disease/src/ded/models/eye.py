"""Compact MobileNet-style network for binary eye-disease classification."""

from __future__ import annotations

from typing import Any

import numpy as np

from .. import config

DEPTHWISE_MULTIPLIER = 1


def depthwise_separable_block(
    layers: Any,
    inputs: Any,
    filters: int,
    strides: int = 1,
    dropout: float = 0.0,
) -> Any:
    """Inverted-residual style block: pointwise -> depthwise -> pointwise."""
    x = layers.Conv2D(
        inputs.shape[-1] * DEPTHWISE_MULTIPLIER,
        kernel_size=1,
        padding="same",
        use_bias=False,
    )(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu6")(x)

    x = layers.DepthwiseConv2D(
        kernel_size=3,
        strides=strides,
        padding="same",
        use_bias=False,
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu6")(x)

    if dropout > 0:
        x = layers.Dropout(dropout)(x)

    x = layers.Conv2D(filters, kernel_size=1, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    return layers.Activation("relu6")(x)


def build_eye_cnn(
    image_size: int = config.EYE_IMAGE_SIZE,
    n_classes: int = 2,
    dropout: float = 0.5,
    learning_rate: float = 1e-3,
    seed: int = config.RANDOM_STATE,
    class_weight: dict[int, float] | None = None,
    width_multiplier: float = 1.0,
) -> Any:
    """Lightweight depthwise-separable CNN, trained from scratch.

    Kept deliberately small so it trains on CPU and can run live in the
    Streamlit app. Architecture mirrors MobileNet's separation of spatial and
    channel mixing rather than copying its large pretrained weights.

    ``width_multiplier`` scales every stage so the capacity can be reduced for
    small datasets; the eye corpus has fewer than a thousand images, where a
    wide network memorises the training split instead of generalising.
    """
    from tensorflow import keras
    from tensorflow.keras import layers

    keras.utils.set_random_seed(seed)
    layers_module = layers

    def width(filters: int) -> int:
        return max(8, int(filters * width_multiplier))

    inputs = keras.Input(shape=(image_size, image_size, 1), name="eye_image")
    x = layers_module.Conv2D(
        width(32), kernel_size=3, strides=2, padding="same", use_bias=False
    )(inputs)
    x = layers_module.BatchNormalization()(x)
    x = layers_module.Activation("relu6")(x)

    for filters, strides, dropout_rate in (
        (64, 1, 0.0),
        (128, 2, 0.1),
        (128, 1, 0.1),
        (256, 2, 0.2),
        (256, 1, 0.2),
        (512, 2, 0.3),
    ):
        x = depthwise_separable_block(
            layers_module, x, width(filters), strides, dropout_rate
        )

    x = layers_module.GlobalAveragePooling2D()(x)
    x = layers_module.Dense(width(128), activation="relu6")(x)
    x = layers_module.Dropout(dropout)(x)
    x = layers_module.Dense(width(64), activation="relu6")(x)
    outputs = layers_module.Dense(n_classes, activation="softmax", name="probability")(x)

    model = keras.Model(inputs, outputs, name="ded_eye_mobilenet")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.class_weight = class_weight  # type: ignore[attr-defined]
    return model


def compute_class_weights(targets: np.ndarray) -> dict[int, float]:
    """Inverse-frequency weights so the minority class is not ignored."""
    counts = np.bincount(np.asarray(targets, dtype=int), minlength=2)
    total = counts.sum()
    weights = total / (len(counts) * np.clip(counts, 1, None))
    return {index: float(value) for index, value in enumerate(weights)}
