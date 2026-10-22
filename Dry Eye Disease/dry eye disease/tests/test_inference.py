"""Tests for the inference layer used by the Streamlit dashboard.

These guard the column-contract between the questionnaire form and the model:
the feature builder must tolerate a submission that omits optional fields.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ded import config, inference, survey


def _answers(max_frequency: bool = True) -> dict[str, object]:
    value = "all the time" if max_frequency else "None of the time"
    return {
        "screen_time": ">8 hours",
        "device": "Smartphone, Laptop",
        "frequency": dict.fromkeys(config.FREQUENCY_COLUMNS, value),
    }


def test_feature_frame_works_without_academic_year() -> None:
    """A submission with no academic year must still build a full frame."""
    frame = inference.symptom_feature_frame(_answers(), age=22, gender="Female")

    assert len(frame) == 1
    assert "osdi" in frame.columns
    assert np.isnan(frame["study_year"].iloc[0])


def test_feature_frame_matches_training_columns() -> None:
    """Inference must emit exactly the columns seen during training."""
    training = survey.prepare_survey().features.columns
    produced = inference.symptom_feature_frame(_answers(), 22, "Female", "II").columns

    missing = [name for name in training if name not in produced]
    assert not missing, f"feature builder is missing {missing}"


def test_feature_frame_includes_academic_year_when_supplied() -> None:
    frame = inference.symptom_feature_frame(_answers(), 22, "Female", "III")
    assert frame["study_year"].iloc[0] == 3


def test_build_features_handles_missing_optional_column() -> None:
    """build_features must not raise when Academic Year is absent."""
    row = {
        "Age": 30,
        "Gender": "Male",
        config.SCREEN_TIME_COLUMN: "4-6 hours",
        config.DEVICE_COLUMN: "Laptop",
    }
    row.update(dict.fromkeys(config.FREQUENCY_COLUMNS, "some times"))

    frame = survey.build_features(pd.DataFrame([row]))

    assert frame["study_year"].isna().all()
    assert frame["age"].iloc[0] == 30


def test_osdi_extremes_flow_through_feature_frame() -> None:
    high = inference.symptom_feature_frame(_answers(True), 22, "Female")
    low = inference.symptom_feature_frame(_answers(False), 22, "Female")

    assert high["osdi"].iloc[0] == pytest.approx(100.0)
    assert low["osdi"].iloc[0] == pytest.approx(0.0)


def test_severity_from_osdi_brackets() -> None:
    assert survey.severity_from_osdi(0.0) == "Normal"
    assert survey.severity_from_osdi(100.0) == "Severe"


def test_predictors_raise_when_artifact_missing(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        inference.EyeImagePredictor(tmp_path / "absent.keras")
    with pytest.raises(FileNotFoundError):
        inference.BlinkPredictor(tmp_path / "absent.keras")
    with pytest.raises(FileNotFoundError):
        inference.SymptomPredictor(tmp_path / "absent.joblib")
