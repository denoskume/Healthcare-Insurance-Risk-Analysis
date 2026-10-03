import pandas as pd
import pytest
from src.data import build_analytical_dataset
from src.validation import clean_analytical_dataset, validate_analytical_dataset, validate_source_schema


def test_join_reports_duplicates_and_unmatched_ids(sample_sources):
    analytical, diagnostics = build_analytical_dataset(
        sample_sources["hospitalisations"], sample_sources["examinations"]
    )
    assert diagnostics["duplicate_ids"] == ["?"]
    assert diagnostics["duplicate_extra_rows"] == 1
    assert diagnostics["unmatched_ids"] == ["?", "id2444"]
    assert len(analytical) == 5


def test_schema_validation_rejects_missing_required_column(sample_sources):
    sample_sources["examinations"] = sample_sources["examinations"].drop(columns="BMI")
    with pytest.raises(ValueError):
        validate_source_schema(sample_sources)


def test_cleaning_normalizes_unknowns_and_text_fields(sample_sources):
    analytical, _ = build_analytical_dataset(
        sample_sources["hospitalisations"], sample_sources["examinations"]
    )
    report = validate_analytical_dataset(analytical)
    cleaned = clean_analytical_dataset(analytical)
    assert report["unknown_year_count"] == 1
    assert set(cleaned["Heart Issues"].dropna().unique()) <= {"Yes", "No"}
    assert pd.api.types.is_numeric_dtype(cleaned["NumberOfMajorSurgeries"])
    assert cleaned["charges"].notna().all()
