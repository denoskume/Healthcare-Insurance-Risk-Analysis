import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, mean_absolute_error
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline

RANDOM_STATE = 42


def grouped_train_test_split(X, y, groups, test_size: float = 0.2, random_state: int = RANDOM_STATE):
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    return (
        X.iloc[train_idx].copy(), X.iloc[test_idx].copy(),
        y.iloc[train_idx].copy(), y.iloc[test_idx].copy(),
        groups.iloc[train_idx].copy(), groups.iloc[test_idx].copy(),
    )


def derive_high_cost_labels(y_train: pd.Series, y_other: pd.Series, quantile: float = 0.90):
    threshold = float(y_train.quantile(quantile))
    return (y_train >= threshold).astype(int), (y_other >= threshold).astype(int), threshold


def get_regression_candidates(preprocessor):
    return {
        "Dummy Mean": Pipeline([("preprocessor", clone(preprocessor)), ("model", DummyRegressor(strategy="mean"))]),
        "Ridge": Pipeline([("preprocessor", clone(preprocessor)), ("model", Ridge(alpha=1.0))]),
        "Random Forest": Pipeline([("preprocessor", clone(preprocessor)), ("model", RandomForestRegressor(n_estimators=120, min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1))]),
        "HistGradientBoosting": Pipeline([("preprocessor", clone(preprocessor)), ("model", HistGradientBoostingRegressor(max_iter=150, learning_rate=0.08, random_state=RANDOM_STATE))]),
        "HistGradientBoosting Log Target": TransformedTargetRegressor(
            regressor=Pipeline([("preprocessor", clone(preprocessor)), ("model", HistGradientBoostingRegressor(max_iter=150, learning_rate=0.08, random_state=RANDOM_STATE))]),
            func=np.log1p,
            inverse_func=np.expm1,
        ),
    }


def get_classification_candidates(preprocessor):
    return {
        "Dummy Prior": Pipeline([("preprocessor", clone(preprocessor)), ("model", DummyClassifier(strategy="prior"))]),
        "Logistic Regression": Pipeline([("preprocessor", clone(preprocessor)), ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE))]),
        "Random Forest": Pipeline([("preprocessor", clone(preprocessor)), ("model", RandomForestClassifier(n_estimators=120, min_samples_leaf=2, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1))]),
        "HistGradientBoosting": Pipeline([("preprocessor", clone(preprocessor)), ("model", HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08, random_state=RANDOM_STATE))]),
    }


def compare_models_cv(models, X_train, y_train, groups, task: str) -> pd.DataFrame:
    cv = GroupKFold(n_splits=5)
    rows = []
    for name, model in models.items():
        fold_scores = []
        for tr, va in cv.split(X_train, y_train, groups):
            fitted = clone(model).fit(X_train.iloc[tr], y_train.iloc[tr])
            if task == "regression":
                pred = fitted.predict(X_train.iloc[va])
                fold_scores.append(mean_absolute_error(y_train.iloc[va], pred))
            elif task == "classification":
                score = fitted.predict_proba(X_train.iloc[va])[:, 1]
                fold_scores.append(average_precision_score(y_train.iloc[va], score))
            else:
                raise ValueError("task must be 'regression' or 'classification'")
        rows.append({"model": name, "cv_mean": float(np.mean(fold_scores)), "cv_std": float(np.std(fold_scores))})
    result = pd.DataFrame(rows)
    return result.sort_values("cv_mean", ascending=(task == "regression")).reset_index(drop=True)
