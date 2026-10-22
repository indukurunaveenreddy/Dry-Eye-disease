"""Tests for the model architectures and training utilities."""

from __future__ import annotations

import numpy as np
import pytest

from ded import config
from ded.models.blink import (
    VGG19_BLOCKS,
    architecture_summary,
    build_blink_cnn,
    compute_blink_class_weights,
)
from ded.models.eye import build_eye_cnn, compute_class_weights
from ded.models.symptom import build_classical_models, build_keras_mlp


@pytest.fixture(scope="module")
def eye_model():
    return build_eye_cnn(image_size=32, n_classes=2)


@pytest.fixture(scope="module")
def blink_model():
    return build_blink_cnn(image_size=32, n_classes=3)


def test_eye_cnn_output_shape(eye_model) -> None:
    output = eye_model.predict(np.zeros((2, 32, 32, 1), dtype=np.float32), verbose=0)
    assert output.shape == (2, 2)
    assert np.allclose(output.sum(axis=1), 1.0, atol=1e-5)


def test_eye_cnn_is_compact(eye_model) -> None:
    assert eye_model.count_params() < 1_000_000


def test_blink_cnn_output_shape(blink_model) -> None:
    output = blink_model.predict(np.zeros((2, 32, 32, 1), dtype=np.float32), verbose=0)
    assert output.shape == (2, 3)
    assert np.allclose(output.sum(axis=1), 1.0, atol=1e-5)


def test_blink_cnn_is_smaller_than_reference_vgg19(blink_model) -> None:
    summary = architecture_summary()
    assert blink_model.count_params() < summary["reference_vgg19_parameters"]
    assert summary["total_convolutions"] == 16


def test_vgg19_block_topology() -> None:
    assert VGG19_BLOCKS[0] == (64, 2)
    assert len(VGG19_BLOCKS) == 5
    assert sum(repeats for _, repeats in VGG19_BLOCKS) == 16


def test_class_weights_favour_minority_class() -> None:
    targets = np.array([0] * 90 + [1] * 10)
    weights = compute_class_weights(targets)
    assert weights[1] > weights[0]


def test_class_weights_handles_single_class() -> None:
    """Weights follow sklearn's balanced formula n / (k * count).

    With only class 0 observed the formula still divides by the configured
    class count of two, so class 0 gets 0.5 and the empty class 1 is never
    referenced during fitting.
    """
    weights = compute_class_weights(np.zeros(20, dtype=int))
    assert set(weights) == {0, 1}
    assert weights[0] == pytest.approx(0.5)


def test_class_weights_balance_total_mass() -> None:
    """Each class should contribute equal total weight to the loss."""
    targets = np.array([0] * 90 + [1] * 10)
    weights = compute_class_weights(targets)
    counts = np.array([90, 10], dtype=float)
    mass = np.array([weights[0] * counts[0], weights[1] * counts[1]])
    assert mass[0] == pytest.approx(mass[1])


def test_blink_class_weights_scale_with_imbalance() -> None:
    targets = np.array([0] * 3000 + [1] * 3000 + [2] * 3000 + [3] * 100 + [4] * 100)
    weights = compute_blink_class_weights(targets, n_classes=5)
    assert weights[3] > weights[0]


def test_symptom_mlp_output_shape() -> None:
    model = build_keras_mlp(n_features=29, n_classes=4)
    output = model.predict(np.zeros((3, 29), dtype=np.float32), verbose=0)
    assert output.shape == (3, 4)
    assert np.allclose(output.sum(axis=1), 1.0, atol=1e-5)


def test_classical_models_all_define_preprocessing() -> None:
    for name, model in build_classical_models().items():
        assert hasattr(model, "named_steps"), name
        assert "prep" in model.named_steps, name
        assert "clf" in model.named_steps, name


def test_classical_models_predict_proba() -> None:
    features = np.random.default_rng(0).normal(size=(40, 6))
    targets = np.array([0] * 20 + [1] * 20)
    for name, model in build_classical_models().items():
        model.fit(features, targets)
        probabilities = model.predict_proba(features)
        assert probabilities.shape == (40, 2), name
        assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5), name


def test_eye_class_order_is_not_then_affected() -> None:
    assert config.EYE_CLASSES == {"Not": 0, "Affected": 1}
    assert config.EYE_CLASS_NAMES == ("Not Affected", "Affected")


def test_min_sample_threshold_excludes_partial_fold() -> None:
    assert config.BLINK_MIN_SAMPLES > 10
