"""Streamlit dashboard for the dry eye disease detection system."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ded import config, images, inference, survey  # noqa: E402
from ded.metrics import load_json  # noqa: E402

st.set_page_config(
    page_title="Dry Eye Disease Detection",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

SEVERITY_COLORS = {
    "Normal": "#2a9d8f",
    "Mild": "#e9c46a",
    "Moderate": "#f4a261",
    "Severe": "#e76f51",
    "Healthy": "#2a9d8f",
    "Dry Eye": "#e76f51",
}

FREQUENCY_OPTIONS = list(config.SURVEY_FREQUENCY_SCALE)
FREQ_INDEX = {label: index for index, label in enumerate(FREQUENCY_OPTIONS)}


@st.cache_resource(show_spinner="Loading symptom models ...")
def load_symptom_predictor() -> inference.SymptomPredictor | None:
    try:
        return inference.SymptomPredictor()
    except FileNotFoundError:
        return None


@st.cache_resource(show_spinner="Loading eye image model ...")
def load_eye_predictor() -> inference.EyeImagePredictor | None:
    try:
        return inference.EyeImagePredictor()
    except Exception:  # noqa: BLE001 - surface any load failure as a UI message
        return None


@st.cache_resource(show_spinner="Loading blink model ...")
def load_blink_predictor() -> inference.BlinkPredictor | None:
    try:
        return inference.BlinkPredictor()
    except Exception:  # noqa: BLE001
        return None


@st.cache_data(show_spinner="Loading survey ...")
def load_bundle() -> survey.SurveyBundle:
    return survey.prepare_survey()


@st.cache_data(show_spinner="Loading metrics ...")
def load_metrics(name: str) -> dict | None:
    path = config.ARTIFACTS / f"{name}_metrics.json"
    return load_json(path) if path.exists() else None


def missing_artifact_warning(artifact: str) -> None:
    st.warning(
        f"`artifacts/{artifact}` is missing. Run the matching training script "
        "to generate it, then reload this page.",
        icon="warning",
    )


def sidebar() -> str:
    with st.sidebar:
        st.markdown("## Dry Eye Disease")
        st.caption("Multi-source detection system")

        page = st.radio(
            "Navigate",
            [
                "Overview",
                "Symptom Assessment",
                "Image Analysis",
                "Model Performance",
                "About",
            ],
            label_visibility="collapsed",
        )

        st.divider()
        st.markdown("#### Status")
        for label, artifact in (
            ("Symptom models", "symptom_models.joblib"),
            ("Eye CNN", "eye_cnn.keras"),
            ("Blink CNN", "blink_cnn.keras"),
        ):
            present = (config.ARTIFACTS / artifact).exists()
            st.markdown(
                f"{'✅' if present else '❌'} {label}",
                help=f"{'trained' if present else 'not trained'}",
            )
    return page


def page_overview() -> None:
    st.markdown(RESPONSIVE_CSS, unsafe_allow_html=True)
    st.markdown("# Dry Eye Disease Detection")
    st.markdown(
        "This system combines three independent signals into one screening "
        "pipeline: a symptom questionnaire, an eye-image classifier, and an "
        "eye-state and gaze classifier."
    )

    bundle = load_bundle()
    frame = bundle.raw
    counts = frame["Results"].value_counts().reindex(config.SEVERITY_ORDER)

    columns = st.columns(4)
    columns[0].metric("Survey respondents", f"{len(frame)}")
    columns[1].metric("OSDI questions", f"{len(config.FREQUENCY_COLUMNS)}")
    columns[2].metric("Eye images", "999")
    columns[3].metric("Blink crops", "10,750")

    left, right = st.columns([1, 1])

    with left:
        st.markdown("### Severity distribution")
        chart = pd.DataFrame(
            {"Severity": counts.index, "Respondents": counts.values}
        ).set_index("Severity")
        st.bar_chart(
            chart,
            color=SEVERITY_COLORS["Normal"],
            height=280,
        )

    with right:
        st.markdown("### How the system works")
        st.markdown(
            """
1. **Symptom track** — the 12-item OSDI questionnaire feeds a regularised
   MLP and classical baselines that grade severity as Normal, Mild, Moderate
   or Severe.
2. **Image track** — a MobileNet-style depthwise separable CNN grades an eye
   photograph as affected or unaffected.
3. **Blink track** — a VGG-19-style CNN reads gaze direction and open/closed
   state from small eye crops, capturing the reduced blink rate typical of dry eye.
"""
        )

    st.info(
        "This is a decision-support tool built on student-collected data. "
        "It is not a medical device and must not be used to diagnose.",
        icon="ℹ️",
    )


def page_symptom_assessment() -> None:
    st.markdown(RESPONSIVE_CSS, unsafe_allow_html=True)
    st.markdown("# Symptom Assessment")
    st.markdown("Complete the OSDI questionnaire to receive a severity grade.")

    predictor = load_symptom_predictor()
    if predictor is None:
        st.error(
            "Symptom models are not trained yet. Run "
            "`python3 scripts/train_symptom.py` and reload."
        )
        return

    with st.form("questionnaire"):
        st.markdown("### About you")
        columns = st.columns([1, 1, 2])
        age = columns[0].number_input("Age", min_value=5, max_value=100, value=21)
        gender = columns[1].selectbox("Gender", ["Female", "Male"])
        study_year = columns[2].selectbox(
            "Academic year", ["I", "II", "III", "IV"], index=1
        )

        screen_time = st.selectbox(
            "Hours per day on smartphones, laptops, etc.", list(config.SCREEN_TIME_ORDER)
        )

        device = st.multiselect(
            "Digital display devices used",
            list(config.DEVICE_TOKENS),
            default=["Smartphone"],
        )
        device_label = ", ".join(device) if device else "None"

        st.markdown("### Symptom frequency")
        st.caption(
            "How often have you experienced each of the following over the past week?"
        )

        frequency: dict[str, str] = {}
        groups = (
            ("Symptoms", list(config.SYMPTOM_COLUMNS)),
            ("Environment", list(config.ENVIRONMENT_COLUMNS)),
        )
        for heading, questions in groups:
            with st.expander(heading, expanded=heading == "Symptoms"):
                for position, question in enumerate(questions):
                    label = question.rstrip("? ").replace(
                        "Eyes that feel gritty (itchy and Scratchy) ",
                        "Eyes feel gritty / scratchy",
                    )
                    frequency[question] = st.selectbox(
                        label,
                        FREQUENCY_OPTIONS,
                        index=0,
                        key=f"freq_{heading}_{position}",
                    )

        submitted = st.form_submit_button("Run assessment", type="primary")

    if not submitted:
        return

    answers = {"screen_time": screen_time, "device": device_label, "frequency": frequency}
    result = predictor.predict(answers, int(age), gender, study_year)

    columns = st.columns(3)
    columns[0].metric("OSDI score", f"{result.osdi:.1f}")
    columns[1].metric("Predicted severity", result.severity_label)
    columns[2].metric("Screening result", result.dedu_label)

    st.markdown("### Severity probabilities")
    frame = pd.DataFrame(
        {"Probability": list(result.severity_probabilities.values())},
        index=list(result.severity_probabilities),
    )
    st.bar_chart(frame, color=SEVERITY_COLORS[result.severity_label], height=260)

    st.markdown("### Dry eye vs healthy")
    binary = pd.DataFrame(
        {"Probability": list(result.dedu_probabilities.values())},
        index=list(result.dedu_probabilities),
    )
    st.bar_chart(binary, color=SEVERITY_COLORS[result.dedu_label], height=220)

    st.markdown("### Interpretation")
    st.markdown(
        f"""
- Your OSDI score is **{result.osdi:.1f}**, which falls in the
  **{result.osdi_band}** band on the clinical cut points.
- The severity model predicts **{result.severity_label}**
  (confidence {result.severity_confidence:.0%}) and the binary screen predicts
  **{result.dedu_label}** (confidence {result.dedu_confidence:.0%}).
- Where the two bands disagree, the questionnaire answers are more reliable than
  the single index because the recorded training labels are not a strict
  function of OSDI.
"""
    )

    if result.dedu_label == "Dry Eye":
        st.warning(
            "Your responses suggest possible dry eye. Consider an optometry or "
            "ophthalmology assessment, particularly if symptoms persist.",
            icon="⚠️",
        )
    else:
        st.success("No strong dry eye signal detected in these responses.", icon="✅")


RESPONSIVE_CSS = """
<style>
@media (max-width: 640px) {
    .block-container { padding-left: 1rem; padding-right: 1rem; }
    h1 { font-size: 1.7rem; }
    h2 { font-size: 1.3rem; }
    h3 { font-size: 1.1rem; }
    [data-testid="stFileUploaderDropzone"] { padding: 1rem 0.75rem; }
    [data-testid="stMetricValue"] { font-size: 1.4rem; }
    [data-testid="stMetricLabel"] { font-size: 0.85rem; }
}
@media (min-width: 1500px) {
    .block-container { max-width: 1400px; }
}
</style>
"""


def image_result_panel(
    uploaded,
    size: int,
    predictor,
    colour: str,
) -> None:
    """Render the result side of an image card, or a neutral placeholder."""
    if uploaded is None:
        st.markdown(
            '<div style="padding:2.2rem 1rem;text-align:center;'
            'border:1px dashed rgba(128,128,128,0.45);border-radius:0.6rem;'
            'color:rgba(128,128,128,0.95);font-size:0.92rem;">'
            "Upload an image above to see the prediction</div>",
            unsafe_allow_html=True,
        )
        return

    array = images.load_image(uploaded, size, grayscale=True)
    if array is None:
        st.error("That file could not be read as an image.", icon="⚠️")
        return

    if predictor is None:
        st.error(
            "Model artifact is missing. Run the matching training script first.",
            icon="⚠️",
        )
        return

    st.image(uploaded, caption=f"Preprocessed to {size}×{size} grayscale", width="stretch")

    label, confidence, probabilities = predictor.predict_array(array)

    first, second = st.columns(2)
    first.metric("Prediction", label)
    second.metric("Confidence", f"{confidence:.1%}")

    frame = pd.DataFrame(
        {"Probability": list(probabilities.values())}, index=list(probabilities)
    )
    st.bar_chart(frame, color=colour, height=220)


def image_card(
    heading: str,
    blurb: str,
    uploaded,
    size: int,
    predictor,
    colour: str,
) -> None:
    """Uploader and result for one classifier, kept together in one column."""
    st.markdown(f"#### {heading}")
    st.caption(blurb)
    image_result_panel(uploaded, size, predictor, colour)


def page_image_analysis() -> None:
    """Unified uploader page holding both image classifiers."""
    st.markdown("# Image Analysis")
    st.markdown(
        "Both image models are driven from this page. Drop in an eye photograph "
        "to grade dry eye involvement, or an eye crop to read gaze direction and "
        "open/closed state. On a phone the two cards stack automatically."
    )
    st.markdown(RESPONSIVE_CSS, unsafe_allow_html=True)

    eye_predictor = load_eye_predictor()
    blink_predictor = load_blink_predictor()

    eye_column, blink_column = st.columns(2, gap="large")

    with eye_column:
        image_card(
            "Eye disease",
            "MobileNet-style CNN, 135k parameters.",
            st.file_uploader(
                "Eye photograph",
                type=["png", "jpg", "jpeg", "bmp", "webp"],
                key="eye_upload",
                label_visibility="collapsed",
            ),
            config.EYE_IMAGE_SIZE,
            eye_predictor,
            "#457b9d",
        )

    with blink_column:
        image_card(
            "Blink & gaze",
            "VGG-19-style CNN, 5.1M parameters.",
            st.file_uploader(
                "Eye crop",
                type=["png", "jpg", "jpeg", "bmp", "webp"],
                key="blink_upload",
                label_visibility="collapsed",
            ),
            config.BLINK_IMAGE_SIZE,
            blink_predictor,
            "#e76f51",
        )

    st.divider()
    st.warning(
        "The eye-image label tracks eye laterality in the source data: 533 of 549 "
        "affected photographs are the right eye and all 450 unaffected ones are the "
        "left. A high confidence here is not evidence of pathology detection.",
        icon="⚠️",
    )


def _show_report(payload: dict) -> None:
    st.code(payload["report_text"], language=None)


def _confusion(payload: dict) -> None:
    matrix = np.asarray(payload["confusion_matrix"])
    names = payload.get("class_names") or [
        str(index) for index in range(matrix.shape[0])
    ]
    frame = pd.DataFrame(matrix, index=names, columns=names)
    st.dataframe(frame, width="stretch")


def page_performance() -> None:
    st.markdown(RESPONSIVE_CSS, unsafe_allow_html=True)
    st.markdown("# Model Performance")

    symptom = load_metrics("symptom")
    eye = load_metrics("eye")
    blink = load_metrics("blink")

    symptom_tab, eye_tab, blink_tab = st.tabs(
        ["Symptom (survey)", "Eye images", "Blink / gaze"]
    )

    with symptom_tab:
        if symptom is None:
            missing_artifact_warning("symptom_metrics.json")
            return

        st.markdown("#### Severity leaderboard")
        st.caption("Repeated stratified cross-validation, macro F1")
        leaderboard = symptom["severity"]["leaderboard"]
        table = pd.DataFrame(
            {
                "Model": list(leaderboard),
                "Macro F1": [entry["macro_f1_mean"] for entry in leaderboard.values()],
                "Std": [entry.get("macro_f1_std", 0.0) for entry in leaderboard.values()],
                "Accuracy": [entry["accuracy_mean"] for entry in leaderboard.values()],
                "Balanced Acc": [entry["balanced_accuracy_mean"] for entry in leaderboard.values()],
            }
        ).sort_values("Macro F1", ascending=False)
        st.dataframe(
            table.style.format(
                {"Macro F1": "{:.4f}", "Std": "{:.4f}",
                 "Accuracy": "{:.4f}", "Balanced Acc": "{:.4f}"}
            ),
            width="stretch",
        )
        st.success(f"Selected model: `{symptom['severity']['best_model']}`")

        importance = symptom.get("feature_importance") or {}
        if importance:
            st.markdown("#### Feature importance")
            top = pd.Series(importance).sort_values(ascending=True).tail(15)
            st.bar_chart(pd.DataFrame({"Importance": top}), height=320, color="#457b9d")

    with eye_tab:
        if eye is None:
            missing_artifact_warning("eye_metrics.json")
            return
        columns = st.columns(3)
        columns[0].metric("Test accuracy", f"{eye['test']['accuracy']:.4f}")
        columns[1].metric("Macro F1", f"{eye['test']['macro_f1']:.4f}")
        columns[2].metric("Parameters", f"{eye['architecture']['parameters']:,}")
        st.markdown("#### Classification report (test split)")
        _show_report(eye["test"])
        st.markdown("#### Confusion matrix")
        _confusion(eye["test"])

    with blink_tab:
        if blink is None:
            missing_artifact_warning("blink_metrics.json")
            return
        columns = st.columns(3)
        columns[0].metric("Test accuracy", f"{blink['test']['accuracy']:.4f}")
        columns[1].metric("Macro F1", f"{blink['test']['macro_f1']:.4f}")
        columns[2].metric("Parameters", f"{blink['architecture']['parameters']:,}")
        st.markdown("#### Classification report (test split)")
        _show_report(blink["test"])
        st.markdown("#### Confusion matrix")
        _confusion(blink["test"])


def page_about() -> None:
    st.markdown("# About")
    st.markdown(
        """
## Project layout

```
src/ded/            reusable library
  config.py         paths, label maps, class definitions
  survey.py         survey loading, OSDI computation, feature engineering
  images.py         image loading, resizing, stratified splits
  cnn_pipeline.py   shared CNN training loop
  inference.py      predictors used by the app
  metrics.py        evaluation harness and serialisation
  models/           symptom.py, eye.py, blink.py
scripts/            eda.py, train_symptom.py, train_eye.py, train_blink.py
app/app.py          Streamlit dashboard
tests/              pytest suite
```

## Reproducing the results

```bash
pip install -r requirements.txt
python3 scripts/eda.py
python3 scripts/train_symptom.py
python3 scripts/train_eye.py
python3 scripts/train_blink.py
streamlit run app/app.py
```

## Known limitations

- The survey cohort is 91% female with a mean age of 20.4, so results are
  calibrated for a young screen-using population.
- 18 of 230 respondents have incomplete answers and receive median imputation.
- The severity labels in the source data are not a strict function of the OSDI
  index, which caps achievable accuracy on the 4-class task.
- The blink corpus is heavily imbalanced; *Partial* contains only 10 images and
  is excluded from training.
"""
    )

    readme = PROJECT_ROOT / "README.md"
    if readme.exists():
        st.markdown("---")
        st.caption(f"`README.md` — {len(readme.read_text(encoding='utf-8').splitlines())} lines")


def main() -> None:
    page = sidebar()

    if page == "Overview":
        page_overview()
    elif page == "Symptom Assessment":
        page_symptom_assessment()
    elif page == "Image Analysis":
        page_image_analysis()
    elif page == "Model Performance":
        page_performance()
    else:
        page_about()


if __name__ == "__main__":
    main()
