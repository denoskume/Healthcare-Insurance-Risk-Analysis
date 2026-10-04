import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, mean_absolute_error
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline

RANDOM_STATE = 42


def grouped_train_test_split(
    features,
    target,
    customer_groups,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
):
    group_splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state,
    )
    train_indices, test_indices = next(
        group_splitter.split(features, target, customer_groups)
    )
    return (
        features.iloc[train_indices].copy(),
        features.iloc[test_indices].copy(),
        target.iloc[train_indices].copy(),
        target.iloc[test_indices].copy(),
        customer_groups.iloc[train_indices].copy(),
        customer_groups.iloc[test_indices].copy(),
    )


def derive_high_cost_labels(
    training_charges: pd.Series,
    comparison_charges: pd.Series,
    quantile: float = 0.90,
):
    high_cost_threshold = float(training_charges.quantile(quantile))
    training_high_cost_labels = (training_charges >= high_cost_threshold).astype(int)
    comparison_high_cost_labels = (comparison_charges >= high_cost_threshold).astype(int)
    return training_high_cost_labels, comparison_high_cost_labels, high_cost_threshold


def get_regression_candidates(preprocessor):
    return {
        "Dummy Mean": Pipeline([
            ("preprocessor", clone(preprocessor)),
            ("model", DummyRegressor(strategy="mean")),
        ]),
        "Ridge": Pipeline([
            ("preprocessor", clone(preprocessor)),
            ("model", Ridge(alpha=1.0)),
        ]),
        "Random Forest": Pipeline([
            ("preprocessor", clone(preprocessor)),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=120,
                    min_samples_leaf=2,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]),
        "HistGradientBoosting": Pipeline([
            ("preprocessor", clone(preprocessor)),
            (
                "model",
                HistGradientBoostingRegressor(
                    max_iter=150,
                    learning_rate=0.08,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]),
        "HistGradientBoosting Log Target": TransformedTargetRegressor(
            regressor=Pipeline([
                ("preprocessor", clone(preprocessor)),
                (
                    "model",
                    HistGradientBoostingRegressor(
                        max_iter=150,
                        learning_rate=0.08,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]),
            func=np.log1p,
            inverse_func=np.expm1,
        ),
    }


def get_classification_candidates(preprocessor):
    return {
        "Dummy Prior": Pipeline([
            ("preprocessor", clone(preprocessor)),
            ("model", DummyClassifier(strategy="prior")),
        ]),
        "Logistic Regression": Pipeline([
            ("preprocessor", clone(preprocessor)),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]),
        "Random Forest": Pipeline([
            ("preprocessor", clone(preprocessor)),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=120,
                    min_samples_leaf=2,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]),
        "HistGradientBoosting": Pipeline([
            ("preprocessor", clone(preprocessor)),
            (
                "model",
                HistGradientBoostingClassifier(
                    max_iter=150,
                    learning_rate=0.08,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]),
    }


def compare_models_cv(
    candidate_models,
    training_features,
    training_target,
    customer_groups,
    task: str,
) -> pd.DataFrame:
    cross_validator = GroupKFold(n_splits=5)
    model_results = []

    for model_name, candidate_model in candidate_models.items():
        fold_scores = []

        for train_indices, validation_indices in cross_validator.split(
            training_features,
            training_target,
            customer_groups,
        ):
            fitted_model = clone(candidate_model).fit(
                training_features.iloc[train_indices],
                training_target.iloc[train_indices],
            )

            if task == "regression":
                validation_predictions = fitted_model.predict(
                    training_features.iloc[validation_indices]
                )
                fold_scores.append(
                    mean_absolute_error(
                        training_target.iloc[validation_indices],
                        validation_predictions,
                    )
                )
            elif task == "classification":
                validation_probability_scores = fitted_model.predict_proba(
                    training_features.iloc[validation_indices]
                )[:, 1]
                fold_scores.append(
                    average_precision_score(
                        training_target.iloc[validation_indices],
                        validation_probability_scores,
                    )
                )
            else:
                raise ValueError("task must be 'regression' or 'classification'")

        model_results.append({
            "model": model_name,
            "cv_mean": float(np.mean(fold_scores)),
            "cv_std": float(np.std(fold_scores)),
        })

    comparison_table = pd.DataFrame(model_results)
    return comparison_table.sort_values(
        "cv_mean",
        ascending=(task == "regression"),
    ).reset_index(drop=True)
