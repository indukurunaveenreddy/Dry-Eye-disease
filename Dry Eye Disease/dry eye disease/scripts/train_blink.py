"""Train the VGG-19-style network on eye-state and gaze-direction crops."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ded import cnn_pipeline, config, images, metrics  # noqa: E402
from ded.models.blink import (  # noqa: E402
    architecture_summary,
    build_blink_cnn,
    compute_blink_class_weights,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--augment", type=float, default=0.4)
    parser.add_argument("--limit-per-class", type=int, default=0)
    parser.add_argument("--width-multiplier", type=float, default=0.5)
    parser.add_argument("--weight-power", type=float, default=0.5)
    parser.add_argument("--weight-cap", type=float, default=4.0)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument(
        "--full",
        action="store_true",
        help="Train on the complete corpus instead of the capped subset.",
    )
    args = parser.parse_args()

    class_map = dict(config.BLINK_CLASSES)
    if args.limit_per_class > 0:
        class_map = dict(class_map)

    root = config.BLINK_IMAGE_ROOT if args.full else config.BALANCED_BLINK_ROOT
    if not root.exists():
        raise SystemExit(
            f"{root} not found. Run scripts/reduce_dataset.py first, or pass --full."
        )

    print(f"Loading blink crops from {root} ...", flush=True)
    folders = images.collect_folders(
        root, class_map, min_samples=config.BLINK_MIN_SAMPLES
    )
    kept = []
    counts: dict[str, int] = {}
    for folder in folders:
        paths = folder.paths
        if args.limit_per_class > 0 and len(paths) > args.limit_per_class:
            rng = np.random.default_rng(config.RANDOM_STATE + folder.label)
            chosen = rng.choice(len(paths), size=args.limit_per_class, replace=False)
            paths = tuple(paths[index] for index in sorted(chosen))
        kept.append((folder.name, folder.label, paths))
        counts[folder.name] = len(paths)

    if not kept:
        raise SystemExit("No blink class folders found with enough samples.")

    features_list, targets_list = [], []
    for name, label, paths in kept:
        images_list, labels, _ = images.load_folder_batch(
            images.ImageFolder(name=name, label=label, paths=paths),
            config.BLINK_IMAGE_SIZE,
            grayscale=True,
        )
        features_list.append(images_list)
        targets_list.append(labels)
        print(f"  {name}: {images_list.shape[0]}")

    features = np.concatenate(features_list)
    targets = np.concatenate(targets_list)

    class_names = [name for name, _, _ in kept]
    remap = {old: new for new, (_, old, _) in enumerate(kept)}
    targets = np.asarray([remap[int(value)] for value in targets], dtype=np.int32)
    n_classes = len(kept)

    print(f"  total {features.shape[0]} crops, {n_classes} classes")

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

    weights = compute_blink_class_weights(
        y_train, n_classes, power=args.weight_power, cap=args.weight_cap
    )
    print(f"  class weights={ {key: round(value, 2) for key, value in weights.items()} }")

    model = build_blink_cnn(
        image_size=config.BLINK_IMAGE_SIZE,
        n_classes=n_classes,
        width_multiplier=args.width_multiplier,
    )
    print(f"  parameters={model.count_params():,}")

    outcome = cnn_pipeline.train_cnn(
        model,
        x_train,
        y_train,
        x_test,
        y_test,
        class_names,
        class_weight=weights,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
    )

    model.save(config.ARTIFACTS / "blink_cnn.keras")

    payload = cnn_pipeline.outcome_payload(
        outcome,
        task="Eye-state and gaze-direction classification",
        architecture={**architecture_summary(), "parameters": int(outcome.model.count_params())},
        raw_counts=counts,
        class_weights=weights,
        train_size=int(len(x_train)),
        test_size=int(len(x_test)),
        excluded_classes=[
            name for name in config.BLINK_CLASSES if name not in class_map_kept(kept)
        ],
    )
    metrics.save_json(payload, config.ARTIFACTS / "blink_metrics.json")

    np.save(config.ARTIFACTS / "blink_test_true.npy", np.asarray(y_test))
    np.save(
        config.ARTIFACTS / "blink_test_prob.npy",
        np.asarray(outcome.model.predict(x_test, verbose=0)),
    )

    print("\n" + outcome.test_metrics["report_text"])
    print(
        f"test accuracy={outcome.test_metrics['accuracy']:.4f} "
        f"macro_f1={outcome.test_metrics['macro_f1']:.4f} "
        f"({outcome.seconds:.0f}s, {len(outcome.history['loss'])} epochs)"
    )
    return 0


def class_map_kept(kept: list[tuple[str, int, tuple[Path, ...]]]) -> set[str]:
    return {name for name, _, _ in kept}


if __name__ == "__main__":
    raise SystemExit(main())
