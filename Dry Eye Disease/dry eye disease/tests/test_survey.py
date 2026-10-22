"""Tests for survey loading, OSDI computation and feature engineering."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ded import config, survey


@pytest.fixture(scope="module")
def bundle() -> survey.SurveyBundle:
    return survey.prepare_survey()


def test_osdi_reproduces_source_column() -> None:
    """The computed OSDI must match the workbook's own OSDI column."""
    raw = survey.load_raw_survey()
    computed = survey.osdi_index(raw)
    source = raw["OSDI"]
    comparable = computed.notna() & source.notna()

    assert comparable.sum() > 200
    assert np.abs(computed[comparable] - source[comparable]).max() < 1e-6


def test_osdi_is_bounded_zero_to_hundred() -> None:
    raw = survey.load_raw_survey()
    values = survey.osdi_index(raw).dropna()
    assert values.min() >= 0.0
    assert values.max() <= 100.0


def test_osdi_all_zero_answers_gives_zero() -> None:
    frame = pd.DataFrame(
        [dict.fromkeys(config.FREQUENCY_COLUMNS, "None of the time")]
    )
    assert survey.osdi_index(frame).iloc[0] == 0.0


def test_osdi_all_maximum_gives_hundred() -> None:
    frame = pd.DataFrame([dict.fromkeys(config.FREQUENCY_COLUMNS, "all the time")])
    assert survey.osdi_index(frame).iloc[0] == pytest.approx(100.0)


def test_standard_osdi_weights_environment_lower() -> None:
    """A pure-environment trigger scores lower than a pure-symptom trigger."""

    def frame(**hot: str) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    column: hot.get("symptom" if column in config.SYMPTOM_COLUMNS
                                    else "environment", "None of the time")
                    for column in config.FREQUENCY_COLUMNS
                }
            ]
        )

    symptom_score = survey.osdi_standard(frame(symptom="all the time")).iloc[0]
    environment_score = survey.osdi_standard(frame(environment="all the time")).iloc[0]

    assert symptom_score > environment_score
    assert symptom_score == pytest.approx(100.0 * 72 / 84)


def test_bundle_shapes_and_labels(bundle: survey.SurveyBundle) -> None:
    assert bundle.n_samples == 230
    assert len(bundle.severity) == bundle.n_samples
    assert set(bundle.severity.unique()) <= set(config.SEVERITY_LABELS.values())
    assert set(bundle.dedu_binary.unique()) == {0, 1}


def test_binary_target_matches_severity(bundle: survey.SurveyBundle) -> None:
    expected = (bundle.severity > 0).astype(int)
    assert (bundle.dedu_binary == expected).all()


def test_features_contain_no_constant_columns(bundle: survey.SurveyBundle) -> None:
    numeric = bundle.features.select_dtypes(include=[np.number])
    varying = (numeric.nunique(dropna=True) > 1).sum()
    assert varying > 0
    assert numeric.shape[1] == len(numeric.columns)


def test_missing_values_are_not_silently_dropped(bundle: survey.SurveyBundle) -> None:
    """Rows with unanswered items are retained and imputed downstream."""
    assert bundle.features["osdi"].isna().sum() == 18


def test_severity_bands_are_monotonic() -> None:
    assert survey.severity_from_osdi(0) == "Normal"
    assert survey.severity_from_osdi(20) == "Mild"
    assert survey.severity_from_osdi(25) == "Moderate"
    assert survey.severity_from_osdi(60) == "Severe"


def test_severity_from_osdi_accepts_numpy_scalar() -> None:
    assert survey.severity_from_osdi(np.float32(15.0)) == "Mild"


def test_screen_time_ordinal_matches_declaration_order() -> None:
    assert config.SCREEN_TIME_ORDER[0] == "<2 hours"
    assert config.SCREEN_TIME_ORDER[-1] == ">8 hours"


def test_feature_building_is_deterministic() -> None:
    raw = survey.load_raw_survey()
    first = survey.build_features(raw).to_numpy()
    second = survey.build_features(raw).to_numpy()
    assert np.allclose(np.nan_to_num(first), np.nan_to_num(second))
