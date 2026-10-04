from pathlib import Path
import pandas as pd

HOSPITAL_FILE = "Hospitalisation details.csv"
EXAM_FILE = "Medical Examinations.csv"
NAME_FILE = "Names.xlsx"


def load_raw_sources(data_directory: Path) -> dict[str, pd.DataFrame]:
    data_directory = Path(data_directory)
    sources = {
        "hospitalisations": pd.read_csv(data_directory / HOSPITAL_FILE),
        "examinations": pd.read_csv(data_directory / EXAM_FILE),
    }

    names_file_path = data_directory / NAME_FILE
    if names_file_path.exists():
        sources["names"] = pd.read_excel(names_file_path)

    return sources


def build_analytical_dataset(
    hospitalisations: pd.DataFrame,
    examinations: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    duplicate_customer_mask = hospitalisations["Customer ID"].duplicated(keep=False)
    duplicate_customer_ids = sorted(
        hospitalisations.loc[duplicate_customer_mask, "Customer ID"]
        .astype(str)
        .unique()
        .tolist()
    )
    unmatched_customer_ids = sorted(
        set(hospitalisations["Customer ID"].astype(str))
        - set(examinations["Customer ID"].astype(str))
    )

    analytical_dataset = hospitalisations.merge(
        examinations,
        on="Customer ID",
        how="left",
        validate="many_to_one",
    )

    diagnostics = {
        "hospitalisation_rows": int(len(hospitalisations)),
        "examination_rows": int(len(examinations)),
        "join_rows": int(len(analytical_dataset)),
        "duplicate_ids": duplicate_customer_ids,
        "duplicate_extra_rows": int(
            hospitalisations["Customer ID"].duplicated().sum()
        ),
        "unmatched_ids": unmatched_customer_ids,
        "unmatched_row_count": int(analytical_dataset["BMI"].isna().sum()),
    }

    return analytical_dataset, diagnostics
