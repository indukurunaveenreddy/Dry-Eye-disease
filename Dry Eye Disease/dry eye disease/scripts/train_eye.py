"""Train the MobileNet-style eye-disease classifier on the Affected/Not split.

Also fits a logistic-regression baseline on the same pixels so the CNN can be
judged against a model with far less capacity.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ded import cnn_pipeline, config, images, metrics  # noqa: E402
from ded.models.eye import build_eye_cnn, compute_class_weights  # noqa: E402

SUBJECT_PATTERN = re.compile(r"IR_(\d+)")


def pixel_pipeline() -> Pipeline:
    """Flatten then z-score, since pixel features are already per-image scaled."""
    return Pipeline([("flatten", images.Flatten()), ("scale", StandardScaler())])


def subject_of(path: Path) -> str:
    """Recover the subject identifier encoded in the capture filename."""
    match = SUBJECT_PATTERN.search(path.name)
    return match.group(1) if match else path.stem


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--augment", type=float, default=0.5)
    parser.add_argument("--width-multiplier", type=float, default=0.5)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--subject-split", action="store_true")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Train on the complete corpus instead of the capped subset.",
    )
    args = parser.parse_args()

    root = config.EYE_IMAGE_ROOT if args.full else config.BALANCED_EYE_ROOT
    if not root.exists():
        raise SystemExit(
            f"{root} not found. Run scripts/reduce_dataset.py first, or pass --full."
        )

    print(f"Loading eye images from {root} ...", flush=True)
    folders = images.collect_folders(root, config.EYE_CLASSES)
    features_list, targets_list, paths_list = [], [], []
    counts: dict[str, int] = {}
    for folder in folders:
        arrays, labels, kept = images.load_folder_batch(
            folder, config.EYE_IMAGE_SIZE, grayscale=True, standardize=True
        )
        features_list.append(arrays)
        targets_list.append(labels)
        paths_list.extend(kept)
        counts[folder.name] = len(arrays)

    features = np.concatenate(features_list)
    targets = np.concatenate(targets_list)
    groups = np.asarray([subject_of(path) for path in paths_list])
    print(f"  {features.shape[0]} images {features.shape[1:]}  counts={counts}")
    print(f"  {len(set(groups))} distinct subjects")

    if args.subject_split:
        x_train, y_train, x_test, y_test = images.stratified_group_split(
            features, targets, groups, test_size=args.test_size
        )
        print("  split by subject (no subject leakage)")
    else:
        x_train, y_train, x_test, y_test = images.stratified_split(
            features, targets, test_size=args.test_size
        )
    print(f"  train={len(x_train)} test={len(x_test)}")

    rng = np.random.default_rng(config.RANDOM_STATE)
    if args.augment > 0:
        x_train, y_train = cnn_pipeline.augmented_copy(
            x_train, y_train, rng, factor=args.augment
        )
        print(f"  after augmentation train={len(x_train)}")

    weights = compute_class_weights(y_train)
    print(f"  class weights={ {key: round(value, 3) for key, value in weights.items()} }")

    print("\nFitting logistic-regression baseline on the same pixels ...", flush=True)
    baseline = Pipeline(
        [
            ("prep", pixel_pipeline()),
            ("clf", LogisticRegression(max_iter=3000, C=0.05, random_state=config.RANDOM_STATE)),
        ]
    )
    baseline.fit(x_train.reshape(len(x_train), -1), y_train)
    baseline_probabilities = baseline.predict_proba(x_test.reshape(len(x_test), -1))
    baseline_metrics = metrics.classification_metrics(
        y_test,
        baseline_probabilities.argmax(axis=1),
        labels=[0, 1],
        class_names=list(config.EYE_CLASS_NAMES),
    )
    print(f"  baseline accuracy={baseline_metrics['accuracy']:.4f} "
          f"macro_f1={baseline_metrics['macro_f1']:.4f}")

    model = build_eye_cnn(
        image_size=config.EYE_IMAGE_SIZE,
        n_classes=len(config.EYE_CLASSES),
        dropout=args.dropout,
        width_multiplier=args.width_multiplier,
    )
    print(f"\n  parameters={model.count_params():,}")
    print(f"  training for up to {args.epochs} epochs ...", flush=True)

    outcome = cnn_pipeline.train_cnn(
        model,
        x_train,
        y_train,
        x_test,
        y_test,
        list(config.EYE_CLASS_NAMES),
        class_weight=weights,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=10,
    )

    model.save(config.ARTIFACTS / "eye_cnn.keras")
    import joblib

    joblib.dump(baseline, config.ARTIFACTS / "eye_baseline.joblib")

    cnn_score = outcome.test_metrics["macro_f1"]
    baseline_score = baseline_metrics["macro_f1"]
    selected = "cnn" if cnn_score >= baseline_score else "baseline"
    print(f"\nSelected model for the eye track: {selected} "
          f"(cnn macro_f1={cnn_score:.4f}, baseline macro_f1={baseline_score:.4f})")

    payload = cnn_pipeline.outcome_payload(
        outcome,
        task="Binary eye disease classification (Affected vs Not)",
        selected_model=selected,
        selection_rule="higher held-out macro F1",
        architecture={
            "name": "MobileNet-style depthwise separable CNN",
            "input": f"{config.EYE_IMAGE_SIZE}x{config.EYE_IMAGE_SIZE} grayscale, per-image standardised",
            "parameters": int(outcome.model.count_params()),
            "width_multiplier": args.width_multiplier,
            "dropout": args.dropout,
            "class_weights": weights,
        },
        raw_counts=counts,
        n_subjects=len(set(groups)),
        subject_split=bool(args.subject_split),
        train_size=int(len(x_train)),
        test_size=int(len(x_test)),
        baseline={
            "model": "logistic regression on standardised pixels",
            **{key: value for key, value in baseline_metrics.items() if key != "report"},
        },
    )
    metrics.save_json(payload, config.ARTIFACTS / "eye_metrics.json")

    np.save(config.ARTIFACTS / "eye_test_true.npy", np.asarray(y_test))
    np.save(
        config.ARTIFACTS / "eye_test_prob.npy",
        np.asarray(outcome.model.predict(x_test, verbose=0)),
    )
    np.save(config.ARTIFACTS / "eye_baseline_prob.npy", baseline_probabilities)

    print("\n=== CNN classification report (test split) ===")
    print(outcome.test_metrics["report_text"])
    print(
        f"CNN  test accuracy={outcome.test_metrics['accuracy']:.4f} "
        f"macro_f1={outcome.test_metrics['macro_f1']:.4f} "
        f"({outcome.seconds:.0f}s, {len(outcome.history['loss'])} epochs)"
    )
    print(
        f"Base test accuracy={baseline_metrics['accuracy']:.4f} "
        f"macro_f1={baseline_metrics['macro_f1']:.4f}"
    )
    if selected == "baseline":
        print(
            "\nNote: at the capped subset size the convolutional model does not beat "
            "a linear model on the same pixels. The baseline is deployed for this "
            "track and the CNN is kept for comparison."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
