"""Central configuration for the Dry Eye Disease detection project."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
ARTIFACTS = PROJECT_ROOT / "artifacts"
FIGURES = PROJECT_ROOT / "docs" / "figures"

for _path in (DATA_PROCESSED, ARTIFACTS, FIGURES):
    _path.mkdir(parents=True, exist_ok=True)

SURVEY_FILE = DATA_RAW / "Data.csv"
EYE_IMAGE_ROOT = DATA_RAW / "Dataset"
BLINK_IMAGE_ROOT = DATA_RAW / "Blink"

DATA_BALANCED = PROJECT_ROOT / "data" / "balanced"
BALANCED_EYE_ROOT = DATA_BALANCED / "Dataset"
BALANCED_BLINK_ROOT = DATA_BALANCED / "Blink"

MAX_IMAGES_PER_CLASS = 250

RANDOM_STATE = 42

SURVEY_FREQUENCY_SCALE: dict[str, int] = {
    "None of the time": 0,
    "some times": 1,
    "half of the times": 2,
    "most of the times": 3,
    "all the time": 4,
}

SYMPTOM_COLUMNS: tuple[str, ...] = (
    "Eyes that are sensitive to light?",
    "Eyes that feel gritty (itchy and Scratchy) ?",
    "Painful or Sore eyes?",
    "Blurred vision?",
    "Poor Vision?",
    "Reading?",
    "Driving at night?",
    "Working with a computer or bank machine ATM?",
    "Watching TV?",
)

ENVIRONMENT_COLUMNS: tuple[str, ...] = (
    "Windy conditions?",
    "Places or areas with low humidity (very dry)?",
    "Areas that are air-conditioned?",
)

FREQUENCY_COLUMNS: tuple[str, ...] = SYMPTOM_COLUMNS + ENVIRONMENT_COLUMNS

SCREEN_TIME_COLUMN = "How many hours in a day do you spend on your smartphones, laptops, etc?"
DEVICE_COLUMN = "What type of Digital display device do you use?"

SCREEN_TIME_ORDER: tuple[str, ...] = (
    "<2 hours",
    "2-4 hours",
    "4-6 hours",
    "6-8 hours",
    ">8 hours",
)

DEVICE_TOKENS: tuple[str, ...] = ("Smartphone", "Tablet", "Laptop", "Computer", "Television")

SEVERITY_ORDER: tuple[str, ...] = ("Normal", "Mild", "Moderate", "Severe")
SEVERITY_LABELS: dict[str, int] = {name: i for i, name in enumerate(SEVERITY_ORDER)}

OSDI_MAX_ITEMS = len(FREQUENCY_COLUMNS)
OSDI_DENOMINATOR = 96.0

OSDI_STANDARD_MAX_SYMPTOM_RAW = 2 * len(SYMPTOM_COLUMNS) * max(SURVEY_FREQUENCY_SCALE.values())
OSDI_STANDARD_MAX_ENVIRONMENT_RAW = len(ENVIRONMENT_COLUMNS) * max(SURVEY_FREQUENCY_SCALE.values())
OSDI_STANDARD_DENOMINATOR = float(
    OSDI_STANDARD_MAX_SYMPTOM_RAW + OSDI_STANDARD_MAX_ENVIRONMENT_RAW
)

EYE_CLASSES: dict[str, int] = {"Not": 0, "Affected": 1}
EYE_CLASS_NAMES: tuple[str, ...] = ("Not Affected", "Affected")

BLINK_CLASSES: dict[str, int] = {
    "forward_look": 0,
    "left_look": 1,
    "right_look": 2,
    "Closed": 3,
    "Open": 4,
    "Partial": 5,
}
BLINK_CLASS_NAMES: tuple[str, ...] = tuple(BLINK_CLASSES)

BLINK_MIN_SAMPLES = 64

EYE_IMAGE_SIZE = 96
BLINK_IMAGE_SIZE = 64
