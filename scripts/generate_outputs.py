from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay

from src.data import build_analytical_dataset, load_raw_sources
from src.evaluation import (
    classification_metrics,
    permutation_importance_table,
    regression_metrics,
)
from src.features import build_preprocessor, prepare_feature_frame
from src.modeling import (
    compare_models_cv,
    derive_high_cost_labels,
    get_classification_candidates,
    get_regression_candidates,
    grouped_train_test_split,
)
from src.validation import (
    clean_analytical_dataset,
    validate_analytical_dataset,
    validate_source_schema,
)

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
FIGURES_DIR = ROOT / "outputs" / "figures"
METRICS_DIR = ROOT / "outputs" / "metrics"


def save_figure(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    sources = load_raw_sources(RAW_DIR)
    validate_source_schema(sources)

    analytical, diagnostics = build_analytical_dataset(
        sources["hospitalisations"], sources["examinations"]
    )
    quality = validate_analytical_dataset(analytical)
    cleaned = clean_analytical_dataset(analytical)
    cleaned.to_csv(PROCESSED_DIR / "analytical_dataset.csv", index=False)

    X, y, groups = prepare_feature_frame(cleaned)
    X_train, X_test, y_train, y_test, groups_train, _ = grouped_train_test_split(
        X, y, groups
    )

    preprocessor = build_preprocessor(X_train)

    regression_models = get_regression_candidates(preprocessor)
    regression_comparison = compare_models_cv(
        regression_models, X_train, y_train, groups_train, "regression"
    )
    regression_name = regression_comparison.iloc[0]["model"]
    regression_model = regression_models[regression_name].fit(X_train, y_train)
    regression_pred = regression_model.predict(X_test)
    regression_result = regression_metrics(y_test, regression_pred)

    y_train_high, y_test_high, threshold = derive_high_cost_labels(y_train, y_test)
    classification_models = get_classification_candidates(preprocessor)
    classification_comparison = compare_models_cv(
        classification_models,
        X_train,
        y_train_high,
        groups_train,
        "classification",
    )
    classification_name = classification_comparison.iloc[0]["model"]
    classification_model = classification_models[classification_name].fit(
        X_train, y_train_high
    )
    classification_score = classification_model.predict_proba(X_test)[:, 1]
    classification_pred = (classification_score >= 0.5).astype(int)
    classification_result = classification_metrics(
        y_test_high, classification_score, classification_pred
    )

    regression_importance = permutation_importance_table(
        regression_model,
        X_test,
        y_test,
        scoring="neg_mean_absolute_error",
        n_repeats=10,
    )
    classification_importance = permutation_importance_table(
        classification_model,
        X_test,
        y_test_high,
        scoring="average_precision",
        n_repeats=10,
    )

    regression_comparison.to_csv(
        METRICS_DIR / "regression_model_comparison.csv", index=False
    )
    classification_comparison.to_csv(
        METRICS_DIR / "classification_model_comparison.csv", index=False
    )
    regression_importance.to_csv(
        METRICS_DIR / "regression_permutation_importance.csv", index=False
    )
    classification_importance.to_csv(
        METRICS_DIR / "classification_permutation_importance.csv", index=False
    )

    final_metrics = {
        "diagnostics": diagnostics,
        "quality": quality,
        "regression": {
            "selected_model": regression_name,
            **regression_result,
        },
        "classification": {
            "selected_model": classification_name,
            "high_cost_threshold": threshold,
            **classification_result,
        },
    }
    (METRICS_DIR / "final_test_metrics.json").write_text(
        json.dumps(final_metrics, indent=2), encoding="utf-8"
    )

    cleaned["charges"].plot.hist(bins=40, figsize=(8, 4), title="Hospitalisation charge distribution")
    plt.xlabel("Charges")
    save_figure(FIGURES_DIR / "charge_distribution.png")

    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, regression_pred, alpha=0.55)
    lower = min(float(y_test.min()), float(regression_pred.min()))
    upper = max(float(y_test.max()), float(regression_pred.max()))
    plt.plot([lower, upper], [lower, upper], linestyle="--")
    plt.xlabel("Actual charges")
    plt.ylabel("Predicted charges")
    plt.title("Actual vs predicted hospitalisation charges")
    save_figure(FIGURES_DIR / "actual_vs_predicted_regression.png")

    ConfusionMatrixDisplay.from_predictions(
        y_test_high,
        classification_pred,
        display_labels=["Not high-cost", "High-cost"],
    )
    plt.title("High-cost classification confusion matrix")
    save_figure(FIGURES_DIR / "classification_confusion_matrix.png")

    PrecisionRecallDisplay.from_predictions(y_test_high, classification_score)
    plt.title("High-cost classification precision-recall curve")
    save_figure(FIGURES_DIR / "classification_pr_curve.png")

    regression_top = regression_importance.head(10).sort_values("importance_mean")
    regression_top.plot.barh(
        x="feature",
        y="importance_mean",
        legend=False,
        figsize=(8, 5),
        title="Regression permutation importance",
    )
    plt.xlabel("Decrease in score after permutation")
    plt.ylabel("")
    save_figure(FIGURES_DIR / "regression_feature_importance.png")

    classification_top = classification_importance.head(10).sort_values("importance_mean")
    classification_top.plot.barh(
        x="feature",
        y="importance_mean",
        legend=False,
        figsize=(8, 5),
        title="Classification permutation importance",
    )
    plt.xlabel("Decrease in score after permutation")
    plt.ylabel("")
    save_figure(FIGURES_DIR / "classification_feature_importance.png")

    print(json.dumps(final_metrics, indent=2))
    print(f"Saved figures to: {FIGURES_DIR}")
    print(f"Saved metrics to: {METRICS_DIR}")


if __name__ == "__main__":
    main()
