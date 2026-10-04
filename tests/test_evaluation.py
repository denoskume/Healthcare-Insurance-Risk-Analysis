import numpy as np
import pandas as pd
from src.evaluation import (
    classification_error_table,
    classification_metrics,
    regression_error_table,
    regression_metrics,
)


def test_regression_metrics_are_deterministic():
    regression_results = regression_metrics([1, 2, 3], [1, 2, 4])

    assert round(regression_results["mae"], 6) == round(1 / 3, 6)
    assert round(regression_results["rmse"], 6) == round(np.sqrt(1 / 3), 6)


def test_classification_metrics_include_pr_auc_and_recall():
    classification_results = classification_metrics(
        [0, 0, 1, 1],
        [0.1, 0.4, 0.7, 0.9],
        [0, 0, 1, 1],
    )

    assert classification_results["recall"] == 1.0
    assert classification_results["pr_auc"] == 1.0


def test_error_tables_keep_actionable_failures():
    regression_errors = regression_error_table(
        pd.DataFrame({"id": [1, 2]}),
        [10, 10],
        [11, 20],
    )
    classification_errors = classification_error_table(
        pd.DataFrame({"id": [1, 2]}),
        [1, 0],
        [0, 1],
        [0.4, 0.6],
    )

    assert regression_errors.iloc[0]["id"] == 2
    assert classification_errors["error_type"].tolist() == [
        "false_negative",
        "false_positive",
    ]
