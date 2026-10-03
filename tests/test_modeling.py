import pandas as pd
from src.modeling import derive_high_cost_labels, grouped_train_test_split


def test_grouped_split_has_no_patient_overlap():
    X = pd.DataFrame({"x": range(12)})
    y = pd.Series(range(12), dtype=float)
    groups = pd.Series([f"p{i//2}" for i in range(12)])
    Xtr, Xte, ytr, yte, gtr, gte = grouped_train_test_split(X, y, groups)
    assert set(gtr).isdisjoint(set(gte))
    assert len(Xtr) + len(Xte) == len(X)


def test_high_cost_threshold_uses_training_values_only():
    y_train = pd.Series([10, 20, 30, 40, 50], dtype=float)
    y_test = pd.Series([1, 2], dtype=float)
    _, _, threshold = derive_high_cost_labels(y_train, y_test)
    _, _, changed_threshold = derive_high_cost_labels(y_train, y_test * 1000)
    assert threshold == y_train.quantile(0.90)
    assert changed_threshold == threshold
