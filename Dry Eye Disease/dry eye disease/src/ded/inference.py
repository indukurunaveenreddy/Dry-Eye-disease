"""Inference helpers shared by the Streamlit app and the tests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import config, survey


def symptom_feature_frame(
    answers: dict[str, Any],
    age: int,
    gender: str,
    academic_year: str | None = None,
) -> pd.DataFrame:
    """Turn raw questionnaire answers into the model's feature frame.

    Builds a one-row frame and reuses :func:`ded.survey.build_features` so the
    column order and encoding can never drift from training.
    """
    row: dict[str, Any] = {
        "Age": age,
        "Gender": gender,
        config.SCREEN_TIME_COLUMN: answers["screen_time"],
        config.DEVICE_COLUMN: answers["device"],
    }
    if academic_year is not None:
        row["Academic Year"] = f"{academic_year} BPTh"
    for column in config.FREQUENCY_COLUMNS:
        row[column] = answers["frequency"][column]
    return survey.build_features(pd.DataFrame([row]))


def symptom_feature_row(answers: dict[str, Any], age: int, gender: str) -> np.ndarray:
    """Feature vector matching the column order used during training."""
    return symptom_feature_frame(answers, age, gender).to_numpy(dtype=np.float64)


@dataclass
class SymptomPrediction:
    """Result of scoring a questionnaire submission."""

    severity_label: str
    severity_index: int
    severity_confidence: float
    severity_probabilities: dict[str, float]
    dedu_label: str
    dedu_confidence: float
    dedu_probabilities: dict[str, float]
    osdi: float
    osdi_band: str
    model_names: dict[str, str]


class SymptomPredictor:
    """Loads the refitted symptom models and scores new respondents."""

    def __init__(self, path: Path | None = None) -> None:
        import joblib

        self.path = path or (config.ARTIFACTS / "symptom_models.joblib")
        if not self.path.exists():
            raise FileNotFoundError(
                f"Symptom models not found at {self.path}. Run scripts/train_symptom.py first."
            )
        payload = joblib.load(self.path)
        self.severity_model = payload["severity_model"]
        self.binary_model = payload["binary_model"]
        self.feature_names: list[str] = list(payload["feature_names"])
        self.severity_classes: list[str] = list(payload["severity_classes"])

    def predict(
        self,
        answers: dict[str, Any],
        age: int,
        gender: str,
        academic_year: str | None = None,
    ) -> SymptomPrediction:
        frame = symptom_feature_frame(answers, age, gender, academic_year)
        vector = frame.to_numpy(dtype=np.float64)

        severity_probabilities = np.asarray(self.severity_model.predict_proba(vector))[0]
        binary_probabilities = np.asarray(self.binary_model.predict_proba(vector))[0]

        severity_index = int(severity_probabilities.argmax())
        severity_label = self.severity_classes[severity_index]
        binary_index = int(binary_probabilities.argmax())

        osdi_value = float(frame["osdi"].iloc[0])

        return SymptomPrediction(
            severity_label=severity_label,
            severity_index=severity_index,
            severity_confidence=float(severity_probabilities.max()),
            severity_probabilities={
                name: float(severity_probabilities[index])
                for index, name in enumerate(self.severity_classes)
                if index < len(severity_probabilities)
            },
            dedu_label="Dry Eye" if binary_index == 1 else "Healthy",
            dedu_confidence=float(binary_probabilities.max()),
            dedu_probabilities={
                "Healthy": float(binary_probabilities[0]),
                "Dry Eye": float(binary_probabilities[1]),
            },
            osdi=osdi_value,
            osdi_band=survey.severity_from_osdi(osdi_value),
            model_names={
                "severity": type(self.severity_model.named_steps["clf"]).__name__
                if hasattr(self.severity_model, "named_steps")
                else type(self.severity_model).__name__,
                "binary": type(self.binary_model.named_steps["clf"]).__name__
                if hasattr(self.binary_model, "named_steps")
                else type(self.binary_model).__name__,
            },
        )


class EyeImagePredictor:
    """Scores an uploaded eye photograph as affected or unaffected.

    Dispatches to whichever model won the held-out comparison recorded in
    ``eye_metrics.json``. On the class-capped subset a linear model on raw
    pixels beats the convolutional network, so the choice is made from data
    rather than assumed.
    """

    def __init__(self, path: Path | None = None) -> None:
        self.metrics_path = config.ARTIFACTS / "eye_metrics.json"
        self.selected = "cnn"
        if self.metrics_path.exists():
            from .metrics import load_json

            self.selected = load_json(self.metrics_path).get("selected_model", "cnn")

        if self.selected == "baseline":
            import joblib

            self.baseline = joblib.load(config.ARTIFACTS / "eye_baseline.joblib")
            self.model = None
            return

        self.path = path or (config.ARTIFACTS / "eye_cnn.keras")
        if not self.path.exists():
            raise FileNotFoundError(
                f"Eye model not found at {self.path}. Run scripts/train_eye.py first."
            )
        import tensorflow as tf

        self.model = tf.keras.models.load_model(self.path)
        self.baseline = None

    @property
    def description(self) -> str:
        if self.selected == "baseline":
            return "logistic regression on standardised pixels"
        return "MobileNet-style depthwise separable CNN"

    def predict_array(self, image: np.ndarray) -> tuple[str, float, dict[str, float]]:
        array = np.asarray(image, dtype=np.float32)
        if array.ndim == 3:
            array = array[None, ...]

        if self.baseline is not None:
            probabilities = self.baseline.predict_proba(array.reshape(len(array), -1))[0]
        else:
            probabilities = np.asarray(self.model.predict(array, verbose=0))[0]

        index = int(probabilities.argmax())
        return (
            config.EYE_CLASS_NAMES[index],
            float(probabilities[index]),
            {name: float(probabilities[i]) for i, name in enumerate(config.EYE_CLASS_NAMES)},
        )

    def predict_file(self, source: Path) -> tuple[str, float, dict[str, float]]:
        from .images import load_image

        array = load_image(Path(source), config.EYE_IMAGE_SIZE, grayscale=True)
        if array is None:
            raise ValueError(f"Could not read image: {source}")
        return self.predict_array(array)


class BlinkPredictor:
    """Scores an eye crop into a gaze direction or open/closed state."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or (config.ARTIFACTS / "blink_cnn.keras")
        if not self.path.exists():
            raise FileNotFoundError(
                f"Blink model not found at {self.path}. Run scripts/train_blink.py first."
            )
        import tensorflow as tf

        self.model = tf.keras.models.load_model(self.path)
        self.class_names = self._infer_class_names()

    def _infer_class_names(self) -> list[str]:
        metrics_path = config.ARTIFACTS / "blink_metrics.json"
        if metrics_path.exists():
            from .metrics import load_json

            stored = load_json(metrics_path).get("class_names")
            if stored:
                return list(stored)
        return list(config.BLINK_CLASS_NAMES)

    def predict_array(self, image: np.ndarray) -> tuple[str, float, dict[str, float]]:
        probabilities = np.asarray(
            self.model.predict(np.asarray(image, dtype=np.float32)[None, ...], verbose=0)
        )[0]
        index = int(probabilities.argmax())
        names = self.class_names
        return (
            names[index],
            float(probabilities[index]),
            {name: float(probabilities[i]) for i, name in enumerate(names)},
        )

    def predict_file(self, source: Path) -> tuple[str, float, dict[str, float]]:
        from .images import load_image

        array = load_image(Path(source), config.BLINK_IMAGE_SIZE, grayscale=True)
        if array is None:
            raise ValueError(f"Could not read image: {source}")
        return self.predict_array(array)
