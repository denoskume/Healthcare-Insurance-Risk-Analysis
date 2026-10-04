import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


def regression_metrics(actual_values, predicted_values) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(actual_values, predicted_values)),
        "rmse": float(np.sqrt(mean_squared_error(actual_values, predicted_values))),
        "r2": float(r2_score(actual_values, predicted_values)),
    }


def classification_metrics(
    actual_labels,
    probability_scores,
    predicted_labels,
) -> dict[str, float]:
    return {
        "precision": float(
            precision_score(actual_labels, predicted_labels, zero_division=0)
        ),
        "recall": float(recall_score(actual_labels, predicted_labels, zero_division=0)),
        "f1": float(f1_score(actual_labels, predicted_labels, zero_division=0)),
        "roc_auc": float(roc_auc_score(actual_labels, probability_scores)),
        "pr_auc": float(average_precision_score(actual_labels, probability_scores)),
    }


def regression_error_table(
    metadata_features,
    actual_values,
    predicted_values,
) -> pd.DataFrame:
    error_table = pd.DataFrame(metadata_features).reset_index(drop=True).copy()
    error_table["actual"] = np.asarray(actual_values)
    error_table["predicted"] = np.asarray(predicted_values)
    error_table["error"] = error_table["predicted"] - error_table["actual"]
    error_table["absolute_error"] = np.abs(error_table["error"])
    return error_table.sort_values(
        "absolute_error",
        ascending=False,
    ).reset_index(drop=True)


def classification_error_table(
    metadata_features,
    actual_labels,
    predicted_labels,
    probability_scores,
) -> pd.DataFrame:
    error_table = pd.DataFrame(metadata_features).reset_index(drop=True).copy()
    error_table["actual"] = np.asarray(actual_labels)
    error_table["predicted"] = np.asarray(predicted_labels)
    error_table["score"] = np.asarray(probability_scores)
    error_table["error_type"] = np.select(
        [
            (error_table["actual"] == 1) & (error_table["predicted"] == 0),
            (error_table["actual"] == 0) & (error_table["predicted"] == 1),
        ],
        ["false_negative", "false_positive"],
        default="correct",
    )
    return error_table


def permutation_importance_table(
    model,
    features,
    target,
    scoring: str,
    n_repeats: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    importance_result = permutation_importance(
        model,
        features,
        target,
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1,
    )
    return pd.DataFrame({
        "feature": features.columns,
        "importance_mean": importance_result.importances_mean,
        "importance_std": importance_result.importances_std,
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True)
