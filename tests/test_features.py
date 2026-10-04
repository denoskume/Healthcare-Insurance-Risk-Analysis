import pandas as pd
from src.data import build_analytical_dataset
from src.features import build_preprocessor, prepare_feature_frame
from src.validation import clean_analytical_dataset


def test_feature_contract_excludes_identifiers_and_target(sample_sources):
    analytical_dataset, _ = build_analytical_dataset(
        sample_sources["hospitalisations"], sample_sources["examinations"]
    )
    features, target_charges, customer_groups = prepare_feature_frame(
        clean_analytical_dataset(analytical_dataset)
    )

    assert "Customer ID" not in features.columns
    assert "charges" not in features.columns
    assert "name" not in features.columns
    assert len(features) == len(target_charges) == len(customer_groups)


def test_missing_customer_ids_are_kept_in_one_valid_group(sample_sources):
    analytical_dataset, _ = build_analytical_dataset(
        sample_sources["hospitalisations"], sample_sources["examinations"]
    )
    _, _, customer_groups = prepare_feature_frame(
        clean_analytical_dataset(analytical_dataset)
    )

    assert customer_groups.isna().sum() == 0
    assert (customer_groups == "__missing_customer_id__").sum() == 2


def test_preprocessor_handles_unseen_category(sample_sources):
    analytical_dataset, _ = build_analytical_dataset(
        sample_sources["hospitalisations"], sample_sources["examinations"]
    )
    features, _, _ = prepare_feature_frame(clean_analytical_dataset(analytical_dataset))
    preprocessor = build_preprocessor(features)

    training_features = features.iloc[:3].copy()
    test_features = features.iloc[[3]].copy()
    test_features.loc[test_features.index[0], "State ID"] = "UNSEEN"

    preprocessor.fit(training_features)

    assert preprocessor.transform(test_features).shape[0] == 1
    assert pd.api.types.is_numeric_dtype(features["BMI"])
