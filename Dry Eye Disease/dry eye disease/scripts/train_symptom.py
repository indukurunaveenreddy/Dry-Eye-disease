"""Train symptom-based dry eye disease severity models.

Runs repeated stratified cross-validation over the classical baselines and a
Keras MLP for both the four-class severity task and the binary dry-eye task,
then refits the winner on the full dataset.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ded import config, metrics, survey  # noqa: E402
from ded.models.symptom import build_classical_models, keras_mlp_sklearn_wrapper  # noqa: E402

SEVERITY_NAMES = list(config.SEVERITY_ORDER)


def permutation_importance_table(
    model,
    features: np.ndarray,
    targets: np.ndarray,
    feature_names: list[str],
    repeats: int = 10,
    seed: int = config.RANDOM_STATE,
) -> dict[str, float]:
    """Permutation importance, which works for any estimator.

    Preferred over ``feature_importances_`` because the selected model is
    often a gradient-boosting or neural pipeline that does not expose one.
    """
    from sklearn.inspection import permutation_importance

    result = permutation_importance(
        model,
        features,
        targets,
        n_repeats=repeats,
        random_state=seed,
        scoring="f1_macro",
    )
    return {
        name: float(value)
        for name, value in sorted(
            zip(feature_names, result.importances_mean, strict=True),
            key=lambda item: item[1],
            reverse=True,
        )
    }


def evaluate_models(
    features: np.ndarray,
    targets: np.ndarray,
    n_splits: int,
    n_repeats: int,
    class_names: list[str],
    include_keras: bool,
    keras_repeats: int = 2,
) -> tuple[dict[str, dict], list[str]]:
    """Cross-validate every candidate model and return scored results."""
    n_features = features.shape[1]
    n_classes = int(targets.max()) + 1
    labels = list(range(n_classes))

    candidates = build_classical_models()
    keras_name: str | None = None
    if include_keras:
        keras_name = "mlp_keras"
        candidates[keras_name] = keras_mlp_sklearn_wrapper(
            n_features=n_features, n_classes=n_classes, max_epochs=200
        )

    results: dict[str, dict] = {}
    for name, estimator in candidates.items():
        repeats = keras_repeats if name == keras_name else n_repeats
        print(f"  [{name}] cross-validating ({repeats} repeats) ...", flush=True)
        try:
            outcome = metrics.cross_validate(
                estimator,
                features,
                targets,
                n_splits=n_splits,
                n_repeats=repeats,
                scoring=["accuracy", "macro_f1", "balanced_accuracy"],
            )
        except Exception as error:  # pragma: no cover - defensive
            print(f"  [{name}] failed: {error}", flush=True)
            continue

        outcome["oof_metrics"].update(
            metrics.classification_metrics(
                targets,
                outcome["oof_predictions"],
                labels=labels,
                class_names=class_names,
            )
        )
        results[name] = outcome

    ranking = sorted(
        results,
        key=lambda key: results[key]["cv"]["summary"]["macro_f1"],
        reverse=True,
    )
    return results, ranking


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument("--skip-keras", action="store_true")
    args = parser.parse_args()

    bundle = survey.prepare_survey()
    features = bundle.features.to_numpy(dtype=np.float64)
    print(f"Survey: {bundle.n_samples} respondents, {features.shape[1]} features")

    severity = bundle.severity.to_numpy()
    binary = bundle.dedu_binary.to_numpy()

    print("\nFour-class severity task")
    severity_results, severity_ranking = evaluate_models(
        features,
        severity,
        args.splits,
        args.repeats,
        SEVERITY_NAMES,
        include_keras=not args.skip_keras,
    )

    print("\nBinary dry-eye task")
    binary_results, binary_ranking = evaluate_models(
        features,
        binary,
        args.splits,
        args.repeats,
        ["Healthy", "Dry Eye"],
        include_keras=not args.skip_keras,
    )

    best_severity = severity_ranking[0]
    best_binary = binary_ranking[0]

    severity_model = (
        keras_mlp_sklearn_wrapper(features.shape[1], len(SEVERITY_NAMES))
        if best_severity == "mlp_keras"
        else build_classical_models()[best_severity]
    )
    binary_model = (
        keras_mlp_sklearn_wrapper(features.shape[1], 2)
        if best_binary == "mlp_keras"
        else build_classical_models()[best_binary]
    )

    print(f"\nRefitting '{best_severity}' on all data")
    severity_model.fit(features, severity)
    print(f"Refitting '{best_binary}' on all data")
    binary_model.fit(features, binary)

    config.ARTIFACTS.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "severity_model": severity_model,
            "binary_model": binary_model,
            "feature_names": list(bundle.features.columns),
            "severity_classes": SEVERITY_NAMES,
        },
        config.ARTIFACTS / "symptom_models.joblib",
    )

    importance = permutation_importance_table(
        severity_model, features, severity, feature_names=list(bundle.features.columns)
    )

    payload = {
        "n_samples": bundle.n_samples,
        "n_features": features.shape[1],
        "severity": {
            "task": "4-class OSDI severity",
            "class_names": SEVERITY_NAMES,
            "best_model": best_severity,
            "leaderboard": {
                name: {
                    "macro_f1_mean": outcome["cv"]["summary"]["macro_f1"],
                    "macro_f1_std": outcome["cv"]["summary"]["std"]["macro_f1"],
                    "accuracy_mean": outcome["cv"]["summary"]["accuracy"],
                    "balanced_accuracy_mean": outcome["cv"]["summary"]["balanced_accuracy"],
                    "oof": outcome["oof_metrics"],
                }
                for name, outcome in severity_results.items()
            },
        },
        "binary": {
            "task": "binary dry eye vs healthy",
            "class_names": ["Healthy", "Dry Eye"],
            "best_model": best_binary,
            "leaderboard": {
                name: {
                    "macro_f1_mean": outcome["cv"]["summary"]["macro_f1"],
                    "accuracy_mean": outcome["cv"]["summary"]["accuracy"],
                    "balanced_accuracy_mean": outcome["cv"]["summary"]["balanced_accuracy"],
                }
                for name, outcome in binary_results.items()
            },
        },
        "feature_importance": importance,
        "osdi_summary": {
            "mean": float(bundle.osdi.mean()),
            "std": float(bundle.osdi.std()),
            "min": float(bundle.osdi.min()),
            "max": float(bundle.osdi.max()),
        },
    }

    metrics.save_json(payload, config.ARTIFACTS / "symptom_metrics.json")

    print("\n=== Severity leaderboard (macro F1) ===")
    for name in severity_ranking:
        entry = payload["severity"]["leaderboard"][name]
        print(
            f"  {name:24s} macro_f1={entry['macro_f1_mean']:.4f}"
            f" +/- {entry['macro_f1_std']:.4f}  acc={entry['accuracy_mean']:.4f}"
        )
    print("\n=== Binary leaderboard (macro F1) ===")
    for name in binary_ranking:
        entry = payload["binary"]["leaderboard"][name]
        print(
            f"  {name:24s} macro_f1={entry['macro_f1_mean']:.4f}"
            f"  acc={entry['accuracy_mean']:.4f}"
        )
    print(f"\nSaved artifacts to {config.ARTIFACTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
