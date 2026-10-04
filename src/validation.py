import re
import numpy as np
import pandas as pd

HOSPITAL_REQUIRED = {
    "Customer ID",
    "year",
    "month",
    "date",
    "children",
    "charges",
    "Hospital tier",
    "City tier",
    "State ID",
}
EXAM_REQUIRED = {
    "Customer ID",
    "BMI",
    "HBA1C",
    "Heart Issues",
    "Any Transplants",
    "Cancer history",
    "NumberOfMajorSurgeries",
    "smoker",
}
BINARY_COLUMNS = ["Heart Issues", "Any Transplants", "Cancer history", "smoker"]


def validate_source_schema(sources: dict[str, pd.DataFrame]) -> None:
    missing_hospital_columns = HOSPITAL_REQUIRED - set(
        sources["hospitalisations"].columns
    )
    missing_examination_columns = EXAM_REQUIRED - set(
        sources["examinations"].columns
    )

    if missing_hospital_columns or missing_examination_columns:
        raise ValueError(
            "Missing required columns: "
            f"hospitalisations={sorted(missing_hospital_columns)}, "
            f"examinations={sorted(missing_examination_columns)}"
        )


def validate_analytical_dataset(dataset: pd.DataFrame) -> dict[str, object]:
    return {
        "rows": int(len(dataset)),
        "duplicate_extra_rows": int(dataset["Customer ID"].duplicated().sum()),
        "unknown_year_count": int(
            (dataset["year"].astype(str).str.strip() == "?").sum()
        ),
        "unknown_smoker_count": int(
            (dataset["smoker"].astype(str).str.strip() == "?").sum()
        ),
        "missing_medical_rows": int(dataset["BMI"].isna().sum()),
        "non_positive_charge_rows": int(
            (pd.to_numeric(dataset["charges"], errors="coerce") <= 0).sum()
        ),
    }


def normalize_binary_value(value):
    if pd.isna(value):
        return np.nan

    normalized_text = str(value).strip().lower()

    if normalized_text in {"?", "", "nan", "none"}:
        return np.nan
    if normalized_text in {"yes", "y", "1", "true"}:
        return "Yes"
    if normalized_text in {"no", "n", "0", "false"}:
        return "No"

    return np.nan


def parse_major_surgery_count(value):
    if pd.isna(value):
        return np.nan

    normalized_text = str(value).strip().lower()

    if normalized_text in {"?", "", "nan", "none"}:
        return np.nan
    if "no major surgery" in normalized_text:
        return 0.0

    surgery_count_match = re.search(r"\d+", normalized_text)
    return float(surgery_count_match.group()) if surgery_count_match else np.nan


def clean_analytical_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
    cleaned_dataset = dataset.copy().replace("?", np.nan)

    numeric_columns = ["year", "date", "children", "charges", "BMI", "HBA1C"]
    for column_name in numeric_columns:
        cleaned_dataset[column_name] = pd.to_numeric(
            cleaned_dataset[column_name],
            errors="coerce",
        )

    for column_name in BINARY_COLUMNS:
        cleaned_dataset[column_name] = cleaned_dataset[column_name].map(
            normalize_binary_value
        )

    cleaned_dataset["NumberOfMajorSurgeries"] = cleaned_dataset[
        "NumberOfMajorSurgeries"
    ].map(parse_major_surgery_count)
    cleaned_dataset["month"] = (
        cleaned_dataset["month"]
        .astype("string")
        .str.strip()
        .replace({"<NA>": pd.NA})
    )
    cleaned_dataset["Hospital tier"] = (
        cleaned_dataset["Hospital tier"].astype("string").str.strip().str.lower()
    )
    cleaned_dataset["City tier"] = (
        cleaned_dataset["City tier"].astype("string").str.strip().str.lower()
    )
    cleaned_dataset["State ID"] = (
        cleaned_dataset["State ID"].astype("string").str.strip()
    )

    valid_charge_rows = (
        cleaned_dataset["charges"].notna() & (cleaned_dataset["charges"] > 0)
    )
    return cleaned_dataset.loc[valid_charge_rows].reset_index(drop=True)
