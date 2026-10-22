"""Loading, cleaning and feature engineering for the OSDI symptom survey."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config


def load_raw_survey() -> pd.DataFrame:
    """Read the survey workbook.

    The file is named ``Data.csv`` but is really an XLSX workbook, so the
    engine is chosen from the file signature instead of the suffix.
    """
    if not config.SURVEY_FILE.exists():
        raise FileNotFoundError(f"Survey file not found: {config.SURVEY_FILE}")
    return pd.read_excel(config.SURVEY_FILE, engine="openpyxl")


def frequency_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    """Map the 12 frequency answers to the integer scale 0-4."""
    encoded = pd.DataFrame(index=frame.index)
    for column in config.FREQUENCY_COLUMNS:
        if column not in frame.columns:
            raise KeyError(f"Missing survey column: {column!r}")
        encoded[column] = frame[column].map(config.SURVEY_FREQUENCY_SCALE)
    return encoded


def osdi_index(frame: pd.DataFrame) -> pd.Series:
    """Ocular Surface Disease Index as computed in the source dataset.

    All twelve questionnaire items carry a weight of 2 and the raw sum is
    normalised by 96, giving a 0-100 score. This reproduces the ``OSDI``
    column of the source workbook exactly on complete rows.
    """
    totals = frequency_matrix(frame).sum(axis=1, min_count=len(config.FREQUENCY_COLUMNS))
    return 100.0 * (2.0 * totals) / config.OSDI_DENOMINATOR


def osdi_standard(frame: pd.DataFrame) -> pd.Series:
    """Published OSDI variant: symptom items weighted 2, environment items 1.

    Normalised by the theoretical maximum raw score of 84
    (``2 * 9 * 4`` for symptoms plus ``1 * 3 * 4`` for environment) so the
    result spans 0-100. Kept as a second feature because it down-weights
    environmental triggers that the source workbook weights equally with
    symptoms.
    """
    encoded = frequency_matrix(frame)
    raw = (
        2.0 * encoded[list(config.SYMPTOM_COLUMNS)].sum(axis=1)
        + encoded[list(config.ENVIRONMENT_COLUMNS)].sum(axis=1)
    )
    return 100.0 * raw / config.OSDI_STANDARD_DENOMINATOR


@dataclass(frozen=True)
class SurveyBundle:
    """Model-ready survey data plus the frame the features were built from."""

    features: pd.DataFrame
    severity: pd.Series
    dedu_binary: pd.Series
    osdi: pd.Series
    raw: pd.DataFrame

    @property
    def n_samples(self) -> int:
        return len(self.features)


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Turn raw survey answers into a numeric design matrix.

    Columns beyond ``Age``, ``Gender`` and the twelve frequency items are
    optional: when absent they are filled with ``NaN`` so the model
    preprocessor can impute them. This lets the dashboard score a partial
    submission without having to collect every field.
    """
    encoded = frequency_matrix(frame)
    features = pd.DataFrame(index=frame.index)

    features["age"] = pd.to_numeric(frame["Age"], errors="coerce")
    features["is_male"] = frame["Gender"].astype(str).str.strip().str.lower().eq("male").astype(float)

    features["osdi"] = osdi_index(frame)
    features["osdi_standard"] = osdi_standard(frame)
    features["symptom_burden"] = (
        2.0 * encoded[list(config.SYMPTOM_COLUMNS)].sum(axis=1) / 24.0
    )
    features["environment_burden"] = (
        encoded[list(config.ENVIRONMENT_COLUMNS)].sum(axis=1) / 12.0
    )

    features["screen_hours_ordinal"] = (
        frame[config.SCREEN_TIME_COLUMN].astype(str).str.strip().map(
            {value: i for i, value in enumerate(config.SCREEN_TIME_ORDER)}
        )
    )
    features["screen_hours_midpoint"] = features["screen_hours_ordinal"].map(
        {0: 1.0, 1: 3.0, 2: 5.0, 3: 7.0, 4: 10.0}
    )

    device_text = frame[config.DEVICE_COLUMN].astype(str).str.lower()
    for token in config.DEVICE_TOKENS:
        features[f"uses_{token.lower()}"] = device_text.str.contains(token.lower(), regex=False).astype(float)
    features["device_count"] = device_text.str.count(",") + 1

    features["study_year"] = (
        _optional_column(frame, "Academic Year")
        .astype(str)
        .str.extract(r"([IVX]+)")[0]
        .map({"I": 1, "II": 2, "III": 3, "IV": 4})
    )

    for index, column in enumerate(config.FREQUENCY_COLUMNS):
        features[f"q{index:02d}_{_slug(column)}"] = encoded[column]

    features["n_positive_answers"] = (encoded >= 3).sum(axis=1)
    features["n_answered_items"] = encoded.notna().sum(axis=1)

    return features


def _optional_column(frame: pd.DataFrame, name: str) -> pd.Series:
    """Return ``name`` if present, otherwise an all-NaN series of the same index."""
    if name in frame.columns:
        return frame[name]
    return pd.Series(np.nan, index=frame.index, name=name)


def _slug(column: str) -> str:
    keep = [character.lower() if character.isalnum() else "_" for character in column]
    slug = "".join(keep)
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_")[:40]


def prepare_survey(frame: pd.DataFrame | None = None) -> SurveyBundle:
    """Load, clean and featurise the survey into a single consistent bundle."""
    raw = load_raw_survey() if frame is None else frame.copy()

    required = set(config.FREQUENCY_COLUMNS) | {"Age", "Gender", "Results"}
    missing = required - set(raw.columns)
    if missing:
        raise KeyError(f"Survey is missing columns: {sorted(missing)}")

    keep = raw.dropna(subset=["Results"]).copy()
    keep = keep[keep["Results"].isin(config.SEVERITY_LABELS)].copy()

    features = build_features(keep)
    features = features.dropna(axis=1, how="all")

    severity = keep["Results"].map(config.SEVERITY_LABELS).astype(int)

    return SurveyBundle(
        features=features.reset_index(drop=True),
        severity=severity.reset_index(drop=True),
        dedu_binary=(severity.reset_index(drop=True) > 0).astype(int),
        osdi=features["osdi"].reset_index(drop=True),
        raw=keep.reset_index(drop=True),
    )


def severity_label(index: int) -> str:
    inverse = {value: name for name, value in config.SEVERITY_LABELS.items()}
    return inverse[int(index)]


def severity_from_osdi(score: float) -> str:
    """Map a continuous OSDI score onto the clinical severity bands.

    Cut points are the midpoints between the observed class boundaries in the
    source data. The source ``Results`` labels are not a strict function of
    OSDI, so this is a display aid rather than a replacement for the model.
    """
    value = float(np.asarray(score).reshape(-1)[0])
    if value < 13.0:
        return "Normal"
    if value < 23.0:
        return "Mild"
    if value < 32.0:
        return "Moderate"
    return "Severe"
