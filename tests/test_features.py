import pandas as pd
from src.data import build_analytical_dataset
from src.features import build_preprocessor, prepare_feature_frame
from src.validation import clean_analytical_dataset


def test_feature_contract_excludes_identifiers_and_target(sample_sources):
    analytical, _ = build_analytical_dataset(
        sample_sources["hospitalisations"], sample_sources["examinations"]
    )
    X, y, groups = prepare_feature_frame(clean_analytical_dataset(analytical))
    assert "Customer ID" not in X.columns
    assert "charges" not in X.columns
    assert "name" not in X.columns
    assert len(X) == len(y) == len(groups)


def test_preprocessor_handles_unseen_category(sample_sources):
    analytical, _ = build_analytical_dataset(
        sample_sources["hospitalisations"], sample_sources["examinations"]
    )
    X, _, _ = prepare_feature_frame(clean_analytical_dataset(analytical))
    preprocessor = build_preprocessor(X)
    train = X.iloc[:3].copy()
    test = X.iloc[[3]].copy()
    test.loc[test.index[0], "State ID"] = "UNSEEN"
    preprocessor.fit(train)
    assert preprocessor.transform(test).shape[0] == 1
    assert pd.api.types.is_numeric_dtype(X["BMI"])
