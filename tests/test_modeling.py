import pandas as pd
from src.modeling import derive_high_cost_labels, grouped_train_test_split


def test_grouped_split_has_no_patient_overlap():
    features = pd.DataFrame({"example_feature": range(12)})
    target_values = pd.Series(range(12), dtype=float)
    customer_groups = pd.Series([f"p{i // 2}" for i in range(12)])

    (
        training_features,
        test_features,
        training_target,
        test_target,
        training_customer_groups,
        test_customer_groups,
    ) = grouped_train_test_split(features, target_values, customer_groups)

    assert set(training_customer_groups).isdisjoint(set(test_customer_groups))
    assert len(training_features) + len(test_features) == len(features)
    assert len(training_target) + len(test_target) == len(target_values)


def test_high_cost_threshold_uses_training_values_only():
    training_charges = pd.Series([10, 20, 30, 40, 50], dtype=float)
    test_charges = pd.Series([1, 2], dtype=float)

    _, _, high_cost_threshold = derive_high_cost_labels(
        training_charges,
        test_charges,
    )
    _, _, changed_test_threshold = derive_high_cost_labels(
        training_charges,
        test_charges * 1000,
    )

    assert high_cost_threshold == training_charges.quantile(0.90)
    assert changed_test_threshold == high_cost_threshold
