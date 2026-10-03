import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score, f1_score, mean_absolute_error,
    mean_squared_error, precision_score, r2_score, recall_score, roc_auc_score,
)


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def classification_metrics(y_true, y_score, y_pred) -> dict[str, float]:
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
    }


def regression_error_table(X_meta, y_true, y_pred) -> pd.DataFrame:
    table = pd.DataFrame(X_meta).reset_index(drop=True).copy()
    table["actual"] = np.asarray(y_true)
    table["predicted"] = np.asarray(y_pred)
    table["error"] = table["predicted"] - table["actual"]
    table["absolute_error"] = np.abs(table["error"])
    return table.sort_values("absolute_error", ascending=False).reset_index(drop=True)


def classification_error_table(X_meta, y_true, y_pred, y_score) -> pd.DataFrame:
    table = pd.DataFrame(X_meta).reset_index(drop=True).copy()
    table["actual"] = np.asarray(y_true)
    table["predicted"] = np.asarray(y_pred)
    table["score"] = np.asarray(y_score)
    table["error_type"] = np.select(
        [
            (table["actual"] == 1) & (table["predicted"] == 0),
            (table["actual"] == 0) & (table["predicted"] == 1),
        ],
        ["false_negative", "false_positive"],
        default="correct",
    )
    return table


def permutation_importance_table(model, X, y, scoring: str, n_repeats: int = 10, random_state: int = 42) -> pd.DataFrame:
    result = permutation_importance(
        model, X, y, scoring=scoring, n_repeats=n_repeats,
        random_state=random_state, n_jobs=-1,
    )
    return pd.DataFrame({
        "feature": X.columns,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True)
