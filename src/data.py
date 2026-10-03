from pathlib import Path
import pandas as pd

HOSPITAL_FILE = "Hospitalisation details.csv"
EXAM_FILE = "Medical Examinations.csv"
NAME_FILE = "Names.xlsx"


def load_raw_sources(data_dir: Path) -> dict[str, pd.DataFrame]:
    data_dir = Path(data_dir)
    sources = {
        "hospitalisations": pd.read_csv(data_dir / HOSPITAL_FILE),
        "examinations": pd.read_csv(data_dir / EXAM_FILE),
    }
    names_path = data_dir / NAME_FILE
    if names_path.exists():
        sources["names"] = pd.read_excel(names_path)
    return sources


def build_analytical_dataset(
    hospitalisations: pd.DataFrame,
    examinations: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    duplicate_mask = hospitalisations["Customer ID"].duplicated(keep=False)
    duplicate_ids = sorted(hospitalisations.loc[duplicate_mask, "Customer ID"].astype(str).unique().tolist())
    unmatched_ids = sorted(set(hospitalisations["Customer ID"].astype(str)) - set(examinations["Customer ID"].astype(str)))

    analytical = hospitalisations.merge(examinations, on="Customer ID", how="left", validate="many_to_one")
    diagnostics = {
        "hospitalisation_rows": int(len(hospitalisations)),
        "examination_rows": int(len(examinations)),
        "join_rows": int(len(analytical)),
        "duplicate_ids": duplicate_ids,
        "duplicate_extra_rows": int(hospitalisations["Customer ID"].duplicated().sum()),
        "unmatched_ids": unmatched_ids,
        "unmatched_row_count": int(analytical["BMI"].isna().sum()),
    }
    return analytical, diagnostics
