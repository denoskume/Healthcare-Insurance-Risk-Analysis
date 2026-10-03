import pandas as pd
import pytest


@pytest.fixture
def sample_sources():
    hospitalisations = pd.DataFrame({
        "Customer ID": ["Id1", "Id2", "?", "?", "id2444"],
        "year": [1992, 1993, "?", 1995, 1996],
        "month": ["Jan", "Feb", "Mar", "Apr", "May"],
        "date": [1, 2, 3, 4, 5],
        "children": [0, 1, 2, 0, 1],
        "charges": [1000.0, 2500.0, 5000.0, 8000.0, 12000.0],
        "Hospital tier": ["tier - 1", "tier - 2", "tier - 3", "tier - 2", "tier - 1"],
        "City tier": ["tier - 1", "tier - 2", "tier - 3", "tier - 2", "tier - 1"],
        "State ID": ["R1", "R1", "R2", "R2", "R3"],
    })
    examinations = pd.DataFrame({
        "Customer ID": ["Id1", "Id2"],
        "BMI": [25.0, 31.0],
        "HBA1C": [5.4, 6.1],
        "Heart Issues": ["No", "yes"],
        "Any Transplants": ["No", "No"],
        "Cancer history": ["No", "Yes"],
        "NumberOfMajorSurgeries": ["No major surgery", "2"],
        "smoker": ["No", "yes"],
    })
    return {"hospitalisations": hospitalisations, "examinations": examinations}
