from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
FIGURES_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "figures"
METRICS_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "metrics"


def save_current_figure(output_path: Path) -> None:
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()


def main() -> None:
    FIGURES_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    raw_sources = load_raw_sources(RAW_DATA_DIR)
    validate_source_schema(raw_sources)

    analytical_dataset, data_diagnostics = build_analytical_dataset(
        raw_sources["hospitalisations"],
        raw_sources["examinations"],
    )
    data_quality_summary = validate_analytical_dataset(analytical_dataset)
    cleaned_dataset = clean_analytical_dataset(analytical_dataset)
    cleaned_dataset.to_csv(
        PROCESSED_DATA_DIR / "analytical_dataset.csv",
        index=False,
    )

    features, target_charges, customer_groups = prepare_feature_frame(cleaned_dataset)
    (
        training_features,
        test_features,
        training_charges,
        test_charges,
        training_customer_groups,
        _,
    ) = grouped_train_test_split(features, target_charges, customer_groups)

    feature_preprocessor = build_preprocessor(training_features)

    regression_candidates = get_regression_candidates(feature_preprocessor)
    regression_comparison = compare_models_cv(
        regression_candidates,
        training_features,
        training_charges,
        training_customer_groups,
        "regression",
    )
    selected_regression_name = regression_comparison.iloc[0]["model"]
    selected_regression_model = regression_candidates[selected_regression_name].fit(
        training_features,
        training_charges,
    )
    regression_predictions = selected_regression_model.predict(test_features)
    regression_test_metrics = regression_metrics(
        test_charges,
        regression_predictions,
    )

    (
        training_high_cost_labels,
        test_high_cost_labels,
        high_cost_threshold,
    ) = derive_high_cost_labels(training_charges, test_charges)

    classification_candidates = get_classification_candidates(feature_preprocessor)
    classification_comparison = compare_models_cv(
        classification_candidates,
        training_features,
        training_high_cost_labels,
        training_customer_groups,
        "classification",
    )
    selected_classification_name = classification_comparison.iloc[0]["model"]
    selected_classification_model = classification_candidates[
        selected_classification_name
    ].fit(
        training_features,
        training_high_cost_labels,
    )
    classification_probability_scores = selected_classification_model.predict_proba(
        test_features
    )[:, 1]
    classification_predictions = (
        classification_probability_scores >= 0.5
    ).astype(int)
    classification_test_metrics = classification_metrics(
        test_high_cost_labels,
        classification_probability_scores,
        classification_predictions,
    )

    regression_permutation_importance = permutation_importance_table(
        selected_regression_model,
        test_features,
        test_charges,
        scoring="neg_mean_absolute_error",
        n_repeats=10,
    )
    classification_permutation_importance = permutation_importance_table(
        selected_classification_model,
        test_features,
        test_high_cost_labels,
        scoring="average_precision",
        n_repeats=10,
    )

    regression_comparison.to_csv(
        METRICS_OUTPUT_DIR / "regression_model_comparison.csv",
        index=False,
    )
    classification_comparison.to_csv(
        METRICS_OUTPUT_DIR / "classification_model_comparison.csv",
        index=False,
    )
    regression_permutation_importance.to_csv(
        METRICS_OUTPUT_DIR / "regression_permutation_importance.csv",
        index=False,
    )
    classification_permutation_importance.to_csv(
        METRICS_OUTPUT_DIR / "classification_permutation_importance.csv",
        index=False,
    )

    final_metrics = {
        "diagnostics": data_diagnostics,
        "quality": data_quality_summary,
        "regression": {
            "selected_model": selected_regression_name,
            **regression_test_metrics,
        },
        "classification": {
            "selected_model": selected_classification_name,
            "high_cost_threshold": high_cost_threshold,
            **classification_test_metrics,
        },
    }
    (METRICS_OUTPUT_DIR / "final_test_metrics.json").write_text(
        json.dumps(final_metrics, indent=2),
        encoding="utf-8",
    )

    cleaned_dataset["charges"].plot.hist(
        bins=40,
        figsize=(8, 4),
        title="Hospitalisation charge distribution",
    )
    plt.xlabel("Charges")
    save_current_figure(FIGURES_OUTPUT_DIR / "charge_distribution.png")

    plt.figure(figsize=(6, 6))
    plt.scatter(test_charges, regression_predictions, alpha=0.55)
    minimum_charge = min(
        float(test_charges.min()),
        float(regression_predictions.min()),
    )
    maximum_charge = max(
        float(test_charges.max()),
        float(regression_predictions.max()),
    )
    plt.plot(
        [minimum_charge, maximum_charge],
        [minimum_charge, maximum_charge],
        linestyle="--",
    )
    plt.xlabel("Actual charges")
    plt.ylabel("Predicted charges")
    plt.title("Actual vs predicted hospitalisation charges")
    save_current_figure(
        FIGURES_OUTPUT_DIR / "actual_vs_predicted_regression.png"
    )

    ConfusionMatrixDisplay.from_predictions(
        test_high_cost_labels,
        classification_predictions,
        display_labels=["Not high-cost", "High-cost"],
    )
    plt.title("High-cost classification confusion matrix")
    save_current_figure(
        FIGURES_OUTPUT_DIR / "classification_confusion_matrix.png"
    )

    PrecisionRecallDisplay.from_predictions(
        test_high_cost_labels,
        classification_probability_scores,
    )
    plt.title("High-cost classification precision-recall curve")
    save_current_figure(FIGURES_OUTPUT_DIR / "classification_pr_curve.png")

    top_regression_features = regression_permutation_importance.head(10).sort_values(
        "importance_mean"
    )
    top_regression_features.plot.barh(
        x="feature",
        y="importance_mean",
        legend=False,
        figsize=(8, 5),
        title="Regression permutation importance",
    )
    plt.xlabel("Decrease in score after permutation")
    plt.ylabel("")
    save_current_figure(
        FIGURES_OUTPUT_DIR / "regression_feature_importance.png"
    )

    top_classification_features = classification_permutation_importance.head(10).sort_values(
        "importance_mean"
    )
    top_classification_features.plot.barh(
        x="feature",
        y="importance_mean",
        legend=False,
        figsize=(8, 5),
        title="Classification permutation importance",
    )
    plt.xlabel("Decrease in score after permutation")
    plt.ylabel("")
    save_current_figure(
        FIGURES_OUTPUT_DIR / "classification_feature_importance.png"
    )

    print(json.dumps(final_metrics, indent=2))
    print(f"Saved figures to: {FIGURES_OUTPUT_DIR}")
    print(f"Saved metrics to: {METRICS_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
