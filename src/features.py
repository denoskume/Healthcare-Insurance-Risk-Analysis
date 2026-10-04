import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

EXCLUDED_FEATURES = {"Customer ID", "charges", "name"}
MISSING_CUSTOMER_GROUP = "__missing_customer_id__"


def prepare_feature_frame(
    dataset: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    modeling_dataset = dataset.copy()
    target_charges = pd.to_numeric(
        modeling_dataset["charges"],
        errors="raise",
    ).astype(float)
    customer_groups = (
        modeling_dataset["Customer ID"]
        .astype("string")
        .fillna(MISSING_CUSTOMER_GROUP)
        .astype(str)
    )
    features = modeling_dataset.drop(
        columns=[
            column_name
            for column_name in EXCLUDED_FEATURES
            if column_name in modeling_dataset.columns
        ]
    )

    numeric_feature_candidates = [
        "year",
        "date",
        "children",
        "BMI",
        "HBA1C",
        "NumberOfMajorSurgeries",
    ]

    for column_name in numeric_feature_candidates:
        if column_name in features:
            features[column_name] = pd.to_numeric(
                features[column_name],
                errors="coerce",
            )

    for column_name in features.columns:
        if column_name not in numeric_feature_candidates:
            features[column_name] = (
                features[column_name]
                .astype(object)
                .where(features[column_name].notna(), None)
            )

    return features, target_charges, customer_groups


def get_feature_groups(features: pd.DataFrame) -> dict[str, list[str]]:
    numeric_features = features.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [
        column_name
        for column_name in features.columns
        if column_name not in numeric_features
    ]
    return {
        "numeric": numeric_features,
        "categorical": categorical_features,
    }


def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    feature_groups = get_feature_groups(features)

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent", missing_values=None)),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer([
        ("numeric", numeric_pipeline, feature_groups["numeric"]),
        ("categorical", categorical_pipeline, feature_groups["categorical"]),
    ], remainder="drop")


def add_descriptive_categories(dataset: pd.DataFrame) -> pd.DataFrame:
    enriched_dataset = dataset.copy()

    if "BMI" in enriched_dataset:
        enriched_dataset["BMI category"] = pd.cut(
            enriched_dataset["BMI"],
            [-np.inf, 18.5, 25, 30, np.inf],
            labels=["underweight", "normal", "overweight", "obesity"],
            right=False,
        )

    if "HBA1C" in enriched_dataset:
        enriched_dataset["HBA1C category"] = pd.cut(
            enriched_dataset["HBA1C"],
            [-np.inf, 5.7, 6.5, np.inf],
            labels=["lower", "intermediate", "higher"],
            right=False,
        )

    return enriched_dataset
