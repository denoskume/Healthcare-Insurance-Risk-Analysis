import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

EXCLUDED_FEATURES = {"Customer ID", "charges", "name"}


def prepare_feature_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    model_df = df.copy()
    y = pd.to_numeric(model_df["charges"], errors="raise").astype(float)
    groups = model_df["Customer ID"].astype(str)
    X = model_df.drop(columns=[c for c in EXCLUDED_FEATURES if c in model_df.columns])
    numeric_candidates = ["year", "date", "children", "BMI", "HBA1C", "NumberOfMajorSurgeries"]
    for col in numeric_candidates:
        if col in X:
            X[col] = pd.to_numeric(X[col], errors="coerce")
    for col in X.columns:
        if col not in numeric_candidates:
            X[col] = X[col].astype(object).where(X[col].notna(), None)
    return X, y, groups


def get_feature_groups(X: pd.DataFrame) -> dict[str, list[str]]:
    numeric = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    return {"numeric": numeric, "categorical": categorical}


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    groups = get_feature_groups(X)
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent", missing_values=None)),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipe, groups["numeric"]),
        ("categorical", categorical_pipe, groups["categorical"]),
    ], remainder="drop")


def add_descriptive_categories(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "BMI" in out:
        out["BMI category"] = pd.cut(
            out["BMI"], [-np.inf, 18.5, 25, 30, np.inf],
            labels=["underweight", "normal", "overweight", "obesity"],
            right=False,
        )
    if "HBA1C" in out:
        out["HBA1C category"] = pd.cut(
            out["HBA1C"], [-np.inf, 5.7, 6.5, np.inf],
            labels=["lower", "intermediate", "higher"],
            right=False,
        )
    return out
