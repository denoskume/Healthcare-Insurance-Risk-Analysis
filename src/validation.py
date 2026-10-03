import re
import numpy as np
import pandas as pd

HOSPITAL_REQUIRED = {
    "Customer ID", "year", "month", "date", "children", "charges",
    "Hospital tier", "City tier", "State ID",
}
EXAM_REQUIRED = {
    "Customer ID", "BMI", "HBA1C", "Heart Issues", "Any Transplants",
    "Cancer history", "NumberOfMajorSurgeries", "smoker",
}
BINARY_COLUMNS = ["Heart Issues", "Any Transplants", "Cancer history", "smoker"]


def validate_source_schema(sources: dict[str, pd.DataFrame]) -> None:
    missing_h = HOSPITAL_REQUIRED - set(sources["hospitalisations"].columns)
    missing_e = EXAM_REQUIRED - set(sources["examinations"].columns)
    if missing_h or missing_e:
        raise ValueError(
            f"Missing required columns: hospitalisations={sorted(missing_h)}, examinations={sorted(missing_e)}"
        )


def validate_analytical_dataset(df: pd.DataFrame) -> dict[str, object]:
    return {
        "rows": int(len(df)),
        "duplicate_extra_rows": int(df["Customer ID"].duplicated().sum()),
        "unknown_year_count": int((df["year"].astype(str).str.strip() == "?").sum()),
        "unknown_smoker_count": int((df["smoker"].astype(str).str.strip() == "?").sum()),
        "missing_medical_rows": int(df["BMI"].isna().sum()),
        "non_positive_charge_rows": int((pd.to_numeric(df["charges"], errors="coerce") <= 0).sum()),
    }


def _normalize_binary(value):
    if pd.isna(value):
        return np.nan
    text = str(value).strip().lower()
    if text in {"?", "", "nan", "none"}:
        return np.nan
    if text in {"yes", "y", "1", "true"}:
        return "Yes"
    if text in {"no", "n", "0", "false"}:
        return "No"
    return np.nan


def _surgery_count(value):
    if pd.isna(value):
        return np.nan
    text = str(value).strip().lower()
    if text in {"?", "", "nan", "none"}:
        return np.nan
    if "no major surgery" in text:
        return 0.0
    match = re.search(r"\d+", text)
    return float(match.group()) if match else np.nan


def clean_analytical_dataset(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy().replace("?", np.nan)
    for col in ["year", "date", "children", "charges", "BMI", "HBA1C"]:
        cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")
    for col in BINARY_COLUMNS:
        cleaned[col] = cleaned[col].map(_normalize_binary)
    cleaned["NumberOfMajorSurgeries"] = cleaned["NumberOfMajorSurgeries"].map(_surgery_count)
    cleaned["month"] = cleaned["month"].astype("string").str.strip().replace({"<NA>": pd.NA})
    cleaned["Hospital tier"] = cleaned["Hospital tier"].astype("string").str.strip().str.lower()
    cleaned["City tier"] = cleaned["City tier"].astype("string").str.strip().str.lower()
    cleaned["State ID"] = cleaned["State ID"].astype("string").str.strip()
    return cleaned.loc[cleaned["charges"].notna() & (cleaned["charges"] > 0)].reset_index(drop=True)
