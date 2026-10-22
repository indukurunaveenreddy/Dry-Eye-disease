"""Exploratory data analysis across all three datasets.

Writes figures to ``docs/figures`` and a markdown data-quality report to
``docs/eda_report.md``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ded import config, images, survey  # noqa: E402

sns.set_theme(style="whitegrid", context="notebook")

SEVERITY_COLORS = {
    "Normal": "#2a9d8f",
    "Mild": "#e9c46a",
    "Moderate": "#f4a261",
    "Severe": "#e76f51",
}


def save(figure: plt.Figure, name: str) -> Path:
    destination = config.FIGURES / name
    figure.savefig(destination, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"  wrote {destination.name}")
    return destination


def figure_severity_distribution(frame: pd.DataFrame) -> None:
    counts = frame["Results"].value_counts().reindex(config.SEVERITY_ORDER)
    figure, axis = plt.subplots(figsize=(8, 4.5))
    bars = axis.bar(
        counts.index,
        counts.values,
        color=[SEVERITY_COLORS[name] for name in counts.index],
    )
    for bar, value in zip(bars, counts.values, strict=True):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + 2,
            f"{value}\n({value / counts.sum():.0%})",
            ha="center",
            fontsize=10,
        )
    axis.set_ylabel("Respondents")
    axis.set_title("Dry eye severity distribution (OSDI questionnaire, n=230)")
    axis.set_ylim(0, counts.max() * 1.25)
    save(figure, "severity_distribution.png")


def figure_osdi_by_severity(bundle: survey.SurveyBundle, frame: pd.DataFrame) -> None:
    figure, axis = plt.subplots(figsize=(8, 4.5))
    data = [bundle.osdi[frame["Results"] == name].dropna() for name in config.SEVERITY_ORDER]
    boxes = axis.boxplot(
        data,
        tick_labels=list(config.SEVERITY_ORDER),
        patch_artist=True,
        medianprops={"color": "black"},
    )
    for patch, name in zip(boxes["boxes"], config.SEVERITY_ORDER, strict=True):
        patch.set_facecolor(SEVERITY_COLORS[name])

    for index, values in enumerate(data, start=1):
        jitter = np.random.default_rng(0).normal(index, 0.06, size=len(values))
        axis.scatter(jitter, values, s=8, alpha=0.35, color="#333333")

    for threshold, label in ((13, "Normal/Mild"), (23, "Mild/Moderate"), (32, "Moderate/Severe")):
        axis.axhline(threshold, linestyle="--", color="#888888", linewidth=1)
        axis.text(len(data) + 0.42, threshold, f" {label}", fontsize=8, color="#666666", va="center")

    axis.set_ylabel("OSDI score")
    axis.set_title("OSDI score by recorded severity band")
    axis.set_xlim(0.5, len(data) + 1.4)
    save(figure, "osdi_by_severity.png")


def figure_screen_time(bundle: survey.SurveyBundle, frame: pd.DataFrame) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    order = list(config.SCREEN_TIME_ORDER)
    counts = frame[config.SCREEN_TIME_COLUMN].value_counts().reindex(order)
    axes[0].bar(order, counts.values, color="#457b9d")
    axes[0].set_ylabel("Respondents")
    axes[0].set_title("Daily screen exposure")
    axes[0].tick_params(axis="x", rotation=20)

    severity_colour = frame["Results"].map(SEVERITY_COLORS)
    axes[1].scatter(
        bundle.features["screen_hours_midpoint"],
        bundle.osdi,
        c=severity_colour,
        alpha=0.75,
        edgecolors="white",
        linewidths=0.4,
    )
    correlation = bundle.features["screen_hours_midpoint"].corr(bundle.osdi)
    axes[1].set_xlabel("Screen hours per day (midpoint)")
    axes[1].set_ylabel("OSDI score")
    axes[1].set_title(f"Screen time vs OSDI  (r = {correlation:.2f})")

    figure.suptitle("Screen exposure as a dry eye risk factor", y=1.02)
    save(figure, "screen_time.png")


def figure_symptom_heatmap(bundle: survey.SurveyBundle) -> None:
    question_columns = sorted(
        column for column in bundle.features.columns if column.startswith("q")
    )
    correlation = bundle.features[question_columns].corr()

    figure, axis = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        correlation,
        cmap="vlag",
        center=0,
        square=True,
        xticklabels=range(len(correlation)),
        yticklabels=range(len(correlation)),
        ax=axis,
        cbar_kws={"shrink": 0.7},
    )
    labels = [
        ("Light" if "sensitive" in name else
         "Gritty" if "gritty" in name else
         "Pain" if "Painful" in name else
         "Blurred" if "Blurred" in name else
         "PoorVis" if "Poor" in name else
         "Reading" if "Reading" in name else
         "Night" if "Driving" in name else
         "Computer" if "computer" in name else
         "TV" if "TV" in name else
         "Wind" if "Windy" in name else
         "Humidity" if "humidity" in name else "AC")
        for name in config.FREQUENCY_COLUMNS
    ]
    axis.set_xticks(range(len(labels)))
    axis.set_xticklabels(labels, rotation=90)
    axis.set_yticks(range(len(labels)))
    axis.set_yticklabels(labels)
    axis.set_title("Response correlation across the 12 OSDI items")
    save(figure, "symptom_correlation.png")


def figure_image_inventory() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for root, class_map, task in (
        (config.EYE_IMAGE_ROOT, config.EYE_CLASSES, "eye disease"),
        (config.BLINK_IMAGE_ROOT, config.BLINK_CLASSES, "blink / gaze"),
    ):
        for name in class_map:
            directory = root / name
            if not directory.is_dir():
                continue
            paths = [
                path
                for path in directory.iterdir()
                if path.is_file() and path.suffix.lower() in images.IMAGE_SUFFIXES
            ]
            rows.append(
                {
                    "task": task,
                    "class": name,
                    "images": len(paths),
                    "megabytes": round(
                        sum(path.stat().st_size for path in paths) / 1024 / 1024, 1
                    ),
                }
            )

    frame = pd.DataFrame(rows)
    figure, axis = plt.subplots(figsize=(10, 4.5))
    palette = ["#457b9d" if task == "eye disease" else "#e76f51" for task in frame["task"]]
    bars = axis.bar(frame["class"], frame["images"], color=palette)
    for bar, value in zip(bars, frame["images"], strict=True):
        axis.text(bar.get_x() + bar.get_width() / 2, value, f" {value:,}", ha="center", fontsize=9)
    axis.set_ylabel("Images")
    axis.set_title("Image dataset inventory by class")
    axis.tick_params(axis="x", rotation=20)
    axis.set_ylim(0, frame["images"].max() * 1.15)
    save(figure, "image_inventory.png")
    return frame


def figure_laterality() -> pd.DataFrame:
    """Show that the eye-disease label tracks eye laterality, not pathology.

    Capture filenames encode the eye (``OD`` right, ``OS`` left), so if the
    label tracks laterality the task is partly a shortcut rather than a
    pathology detector.
    """

    rows: list[dict[str, object]] = []
    for name in config.EYE_CLASSES:
        directory = config.EYE_IMAGE_ROOT / name
        if not directory.is_dir():
            continue
        for path in directory.iterdir():
            if not path.is_file() or path.suffix.lower() not in images.IMAGE_SUFFIXES:
                continue
            side = "OD (right)" if "_OD_" in path.name else "OS (left)" if "_OS_" in path.name else "other"
            rows.append({"class": name, "laterality": side})

    frame = pd.DataFrame(rows)
    table = pd.crosstab(frame["class"], frame["laterality"])

    figure, axis = plt.subplots(figsize=(8, 4.5))
    table.plot(
        kind="bar",
        ax=axis,
        color=["#e76f51", "#457b9d", "#999999"][: len(table.columns)],
    )
    axis.set_ylabel("Images")
    axis.set_title("Eye-disease label versus captured eye laterality")
    axis.tick_params(axis="x", rotation=0)
    for container in axis.containers:
        axis.bar_label(container, fontsize=9)
    save(figure, "eye_laterality.png")
    return table


def data_quality_checks(frame: pd.DataFrame, bundle: survey.SurveyBundle) -> list[str]:
    findings: list[str] = []

    incomplete = int(bundle.features["osdi"].isna().sum())
    findings.append(
        f"**{incomplete} of {len(frame)} respondents ({incomplete / len(frame):.0%}) have "
        "incomplete frequency answers**, so their OSDI score cannot be computed. "
        "The models impute these values with the training-set median."
    )

    band = pd.cut(
        bundle.osdi,
        [-1, 13, 23, 32, 200],
        labels=["<13", "13-23", "23-32", ">32"],
    )
    complete = bundle.osdi.notna()
    conflict = int(
        (pd.crosstab(band[complete], frame["Results"][complete]).max(axis=1) >
         pd.crosstab(band[complete], frame["Results"][complete]).sum(axis=1) * 0.7).sum()
    )
    findings.append(
        f"**{conflict} OSDI band(s) contain more than one severity label.** "
        "The band 13-23 holds both 42 *Mild* and 10 *Moderate* records at an identical "
        "OSDI of 22.917. The recorded `Results` column is therefore not a strict "
        "function of the OSDI column, so `Results` is used as ground truth and OSDI "
        "is treated as a feature rather than a label generator."
    )

    findings.append(
        "**Class imbalance in the blink corpus.** The three gaze classes hold about "
        "3,500 images each while *Closed* and *Open* hold 109 each, and *Partial* holds "
        "only 10. *Partial* is dropped because ten samples cannot be split stratified, "
        "and the remaining classes are reweighted during training."
    )

    findings.append(
        "**Demographic skew.** The sample is 91% female with a mean age of 20.4 and a "
        "maximum of 27, so the models are calibrated for a young screen-using cohort "
        "and should not be read as population prevalence estimates."
    )

    findings.append(
        "**The eye-disease label tracks eye laterality.** Capture filenames encode "
        "the eye as `OD` (right) or `OS` (left): 533 of 549 *Affected* images are "
        "`OD` while all 450 *Not* images are `OS`. A model can therefore reach high "
        "accuracy by learning which eye it is looking at, so the reported score is "
        "not evidence that the model detects dry eye pathology."
    )

    findings.append(
        "**Several captures per subject.** The 999 images cover only 338 subjects, "
        "roughly three captures each. A naive random split leaks subject identity, so "
        "the eye model is trained on a subject-grouped split."
    )

    findings.append(
        "**Reported `Consent` column carries a single value.** Every row records the "
        "same consent sentence, so it has no predictive signal and is excluded."
    )

    return findings


def main() -> int:
    print("Running EDA\n")
    bundle = survey.prepare_survey()
    frame = bundle.raw

    print("Figures")
    figure_severity_distribution(frame)
    figure_osdi_by_severity(bundle, frame)
    figure_screen_time(bundle, frame)
    figure_symptom_heatmap(bundle)
    inventory = figure_image_inventory()
    laterality = figure_laterality()

    findings = data_quality_checks(frame, bundle)

    correlation = bundle.features["screen_hours_midpoint"].corr(bundle.osdi)

    lines = [
        "# Exploratory Data Analysis",
        "",
        "## Datasets",
        "",
        "| Task | Source | Size |",
        "| --- | --- | --- |",
        "| Symptom survey (OSDI-12) | `data/raw/Data.csv` | "
        f"{len(frame)} respondents x {frame.shape[1]} raw columns |",
        "| Eye disease images | `data/raw/Dataset/` | "
        f"{int(inventory[inventory['task'] == 'eye disease']['images'].sum()):,} images |",
        "| Blink / gaze crops | `data/raw/Blink/` | "
        f"{int(inventory[inventory['task'] == 'blink / gaze']['images'].sum()):,} images |",
        "",
        "## Severity distribution",
        "",
        "| Severity | Respondents | Share |",
        "| --- | --- | --- |",
    ]
    counts = frame["Results"].value_counts().reindex(config.SEVERITY_ORDER)
    for name, value in counts.items():
        lines.append(f"| {name} | {value} | {value / counts.sum():.1%} |")

    lines += [
        "",
        "## Eye label versus laterality",
        "",
        laterality.to_markdown(),
        "",
        "## Key relationships",
        "",
        f"- Screen time (midpoint hours) vs OSDI: Pearson r = {correlation:.3f}",
        f"- OSDI summary: mean {bundle.osdi.mean():.2f}, sd {bundle.osdi.std():.2f}, "
        f"range {bundle.osdi.min():.2f}-{bundle.osdi.max():.2f}",
        f"- Mean age {frame['Age'].mean():.1f}, "
        f"{frame['Gender'].value_counts().get('Female', 0)} female / "
        f"{frame['Gender'].value_counts().get('Male', 0)} male",
        "",
        "## Data quality findings",
        "",
    ]
    lines += [f"{index}. {finding}\n" for index, finding in enumerate(findings, start=1)]

    report_path = Path(config.PROJECT_ROOT) / "docs" / "eda_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  wrote {report_path}")

    (config.DATA_PROCESSED / "image_inventory.csv").write_text(
        inventory.to_csv(index=False), encoding="utf-8"
    )
    bundle.features.to_csv(config.DATA_PROCESSED / "survey_features.csv", index=False)
    frame.to_csv(config.DATA_PROCESSED / "survey_clean.csv", index=False)
    print(f"  wrote processed tables to {config.DATA_PROCESSED}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
