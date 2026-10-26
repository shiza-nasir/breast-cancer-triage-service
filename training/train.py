
import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import sklearn
import yaml

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    make_scorer,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "raw" / "breast_cancer.csv"
DEFAULT_OUTPUT = ROOT / "models"
PARAMS_FILE = ROOT / "params.yaml"


def load_params():
    with open(PARAMS_FILE, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def malignant_recall(y_true, y_pred):
    return recall_score(
        (y_true == 0).astype(int),
        (y_pred == 0).astype(int),
        zero_division=0,
    )


def malignant_auc(estimator, x, y):
    probabilities = estimator.predict_proba(x)
    malignant_index = list(estimator.classes_).index(0)
    scores = probabilities[:, malignant_index]
    return roc_auc_score((y == 0).astype(int), scores)


def evaluate(model, x, y):
    predictions = model.predict(x)
    probabilities = model.predict_proba(x)
    malignant_index = list(model.classes_).index(0)

    return {
        "malignant_recall": float(malignant_recall(y, predictions)),
        "accuracy": float(accuracy_score(y, predictions)),
        "roc_auc": float(
            roc_auc_score(
                (y == 0).astype(int),
                probabilities[:, malignant_index],
            )
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model-version", default="2.0.0")
    args = parser.parse_args()

    params = load_params()
    seed = int(params["seed"])
    settings = params["train"]

    data = pd.read_csv(args.data_path)

    if "target" not in data.columns:
        raise ValueError("Dataset must contain the target column.")

    feature_names = [c for c in data.columns if c != "target"]

    if len(feature_names) != 30:
        raise ValueError(f"Expected 30 features, got {len(feature_names)}.")

    if data.isnull().values.any():
        raise ValueError("Dataset contains missing values.")

    x = data[feature_names]
    y = data["target"].astype(int)

    if set(y.unique()) != {0, 1}:
        raise ValueError("Expected target labels 0 and 1.")

    # 80% training data, 20% held-out test data.
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=float(settings["test_size"]),
        random_state=seed,
        stratify=y,
    )

    cv = StratifiedKFold(
        n_splits=int(settings["cv_folds"]),
        shuffle=True,
        random_state=seed,
    )

    candidates = {
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                C=float(settings["logistic_regression"]["C"]),
                max_iter=int(settings["logistic_regression"]["max_iter"]),
                random_state=seed,
            )),
        ]),
        "RandomForest": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", RandomForestClassifier(
                n_estimators=int(settings["random_forest"]["n_estimators"]),
                max_depth=settings["random_forest"]["max_depth"],
                min_samples_leaf=int(
                    settings["random_forest"]["min_samples_leaf"]
                ),
                random_state=seed,
                n_jobs=-1,
            )),
        ]),
    }

    scoring = {
        "malignant_recall": make_scorer(
            malignant_recall,
        ),
        "accuracy": "accuracy",
        "roc_auc": malignant_auc,
    }

    results = {}

    for name, pipeline in candidates.items():
        cv_results = cross_validate(
            pipeline,
            x_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=1,
            error_score="raise",
        )

        # Fit this candidate on the complete training split.
        pipeline.fit(x_train, y_train)

        results[name] = {
            "cv": {
                metric: float(cv_results[f"test_{metric}"].mean())
                for metric in scoring
            },
            "test": evaluate(pipeline, x_test, y_test),
        }

    # Choose by mean CV malignant recall, not test-set results.
    selected_name = max(
        results,
        key=lambda name: results[name]["cv"]["malignant_recall"],
    )
    selected_model = candidates[selected_name]

    output_dir = args.output_dir
    reports_dir = ROOT / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "model.joblib"
    card_path = output_dir / "model_card.json"
    metrics_path = reports_dir / "metrics.json"

    joblib.dump(selected_model, model_path)

    selected_test = results[selected_name]["test"]
    minimum_recall = float(settings["malignant_recall_min"])

    model_card = {
        "model_version": args.model_version,
        "model_family": selected_name,
        "feature_names": feature_names,
        "metrics": results,
        "selected_model_test_metrics": selected_test,
        "seed": seed,
        "scikit_learn_version": sklearn.__version__,
        "target_mapping": {"0": "malignant", "1": "benign"},
    }

    metrics_output = {
        "selected_model": selected_name,
        "models": results,
        "minimum_test_malignant_recall": minimum_recall,
        "selected_test_malignant_recall": selected_test["malignant_recall"],
    }

    with open(card_path, "w", encoding="utf-8") as file:
        json.dump(model_card, file, indent=2)

    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(metrics_output, file, indent=2)

    print(f"Dataset shape: {data.shape}")
    print(f"Training rows: {len(x_train)}")
    print(f"Test rows: {len(x_test)}")
    print(f"Seed: {seed}")

    for name, result in results.items():
        label = " (SELECTED)" if name == selected_name else ""
        print(f"\n{name}{label}")
        for metric, value in result["cv"].items():
            print(f"  CV mean {metric}: {value:.4f}")
        for metric, value in result["test"].items():
            print(f"  Test {metric}: {value:.4f}")

    print(f"\nSaved model: {model_path}")
    print(f"Saved model card: {card_path}")
    print(f"Saved metrics: {metrics_path}")

    if selected_test["malignant_recall"] < minimum_recall:
        print(
            f"\nFAIL: malignant recall "
            f"{selected_test['malignant_recall']:.4f} "
            f"is below {minimum_recall:.4f}.",
            file=sys.stderr,
        )
        return 1

    print("\nPASS: required test malignant recall achieved.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())