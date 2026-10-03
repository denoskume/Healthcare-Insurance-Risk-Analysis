# Healthcare Insurance Risk Analysis

Machine learning analysis of hospitalisation costs and high-cost patient risk using medical and contextual data.

## Overview

This project builds an end-to-end applied machine learning workflow for two related problems:

1. **Cost regression** — predict hospitalisation charges.
2. **High-cost risk classification** — identify hospitalisation events in the top 10% of charges, with the threshold learned from the training set only.

The focus is not only model performance. The repository demonstrates data integration, explicit validation, leakage-safe preprocessing, grouped splitting, model comparison, explainability, error analysis, and reproducibility.

> This is an analytical machine learning project, not a clinical diagnosis system.

## Data

The analysis uses two modeling sources:

| Source | Rows | Main information |
|---|---:|---|
| Hospitalisation details | 2,343 | charges, date/context, children, hospital tier, city tier, state |
| Medical examinations | 2,335 | BMI, HBA1C, medical history, surgeries, smoking status |

A third supplied file contained names. It is deliberately excluded from this public repository and from all model features to keep identity data separate from analytical data.

The raw analytical files are also not redistributed until their publication/licensing terms are documented. `data/raw/README.md` lists the expected filenames for local reproduction.

### Data-quality issues handled explicitly

- 5 duplicate rows beyond the first occurrence of placeholder customer ID `?`
- 3 hospitalisation identifiers without a medical-record match: `?`, `id2444`, `id3444`
- 2 unknown year values represented by `?`
- 2 unknown smoking-status values represented by `?`
- inconsistent binary labels such as `Yes`, `yes`, and `No`
- surgery counts stored as text

Unknown values are kept as missing values rather than silently invented.

## ML workflow

```text
Raw sources
    ↓
Schema and join validation
    ↓
Explicit cleaning
    ↓
Grouped train/test split by Customer ID
    ↓
Train-only preprocessing
    ├── Cost regression
    └── High-cost classification
    ↓
5-fold grouped cross-validation
    ↓
Final untouched test evaluation
    ↓
Permutation importance and error analysis
```

The same customer cannot appear in both train and test partitions. Imputation, encoding, scaling, model selection, and the high-cost threshold are learned without using the final test set.

## Models and results

### Cost regression

Compared models:

- Mean baseline
- Ridge Regression
- Random Forest Regressor
- Histogram Gradient Boosting Regressor
- Histogram Gradient Boosting with log-transformed target

**Selected model: Random Forest Regressor**

| Test metric | Result |
|---|---:|
| MAE | **1,663.17** |
| RMSE | **3,037.04** |
| R² | **0.926** |

### High-cost classification

High cost is defined using the **90th percentile of training charges only**. For the final split, the learned threshold is **34,962.10**.

Compared models:

- Prior-probability baseline
- Logistic Regression
- Random Forest Classifier
- Histogram Gradient Boosting Classifier

**Selected model: Histogram Gradient Boosting Classifier**

| Test metric | Result |
|---|---:|
| Precision | **0.927** |
| Recall | **0.884** |
| F1 | **0.905** |
| ROC-AUC | **0.995** |
| PR-AUC | **0.974** |

PR-AUC and recall are emphasized because the positive high-cost class is intentionally rare.

## Explainability and error analysis

The final analysis covers:

- actual vs predicted charges
- largest regression errors
- false-positive and false-negative inspection
- permutation importance for both selected models

The strongest predictive signals in the fitted models include smoking status, BMI, year, and hospital tier. These are **predictive associations within this dataset**, not medical causal claims.

## Repository structure

```text
Healthcare-Insurance-Risk-Analysis/
├── data/
│   └── raw/README.md
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_data_quality_and_cleaning.ipynb
│   ├── 03_exploratory_analysis.ipynb
│   ├── 04_feature_engineering.ipynb
│   ├── 05_cost_regression.ipynb
│   ├── 06_high_cost_classification.ipynb
│   └── 07_model_explainability_and_error_analysis.ipynb
├── outputs/
│   └── metrics/
├── src/
│   ├── data.py
│   ├── validation.py
│   ├── features.py
│   ├── modeling.py
│   └── evaluation.py
├── tests/
├── requirements.txt
└── README.md
```

## Run locally

Place the two analytical CSV files listed in `data/raw/README.md` inside `data/raw/`, then run:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
pytest -q
jupyter notebook
```

Run notebooks `01` through `07` in order. Reusable logic lives in `src/`, keeping the notebooks focused on analysis and interpretation.

## Limitations

- The dataset is relatively small and does not establish clinical validity.
- Some hospitalisation rows have no matching medical examination record.
- Model importance measures association with predictions, not causation.
- Results should not be generalized to real insurance populations without external validation.

## Technologies

Python · pandas · NumPy · scikit-learn · Matplotlib · Jupyter · pytest
