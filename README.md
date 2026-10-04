# Healthcare Insurance Risk Analysis

<p align="center">
  <strong>Applied machine learning for hospitalisation cost prediction and high-cost risk identification.</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-blue" alt="Python">
  <img src="https://img.shields.io/badge/scikit--learn-ML-orange" alt="scikit-learn">
  <img src="https://img.shields.io/github/actions/workflow/status/denoskume/Healthcare-Insurance-Risk-Analysis/tests.yml?branch=main&label=tests" alt="tests">
</p>

---

## Key Results

| Task | Selected model | Main test result |
|---|---|---:|
| Hospitalisation cost regression | **Random Forest Regressor** | **R² = 0.926** |
| High-cost risk classification | **Histogram Gradient Boosting Classifier** | **PR-AUC = 0.974** |

Additional held-out test metrics:

- **Regression:** MAE = **1,663.17**, RMSE = **3,037.04**
- **Classification:** Precision = **0.927**, Recall = **0.884**, F1 = **0.905**, ROC-AUC = **0.995**
- **High-cost threshold:** **34,962.10**, learned from the training partition only

> This is an analytical machine learning project, not a clinical diagnosis system.

## What This Project Demonstrates

- integration of hospitalisation and medical examination data
- explicit data-quality validation and transparent cleaning
- leakage-safe preprocessing
- grouped train/test splitting by customer identifier
- regression and imbalanced classification workflows
- 5-fold grouped cross-validation
- baseline and model comparison
- held-out error analysis and permutation importance
- reusable Python modules with automated tests
- clear separation between identity data and analytical features

## Problem

The project addresses two related machine learning tasks:

1. **Cost regression** — predict hospitalisation charges.
2. **High-cost classification** — identify hospitalisation events in the highest-cost 10%.

The final test set is isolated before preprocessing, threshold definition, and model selection. The same customer cannot appear in both train and test partitions.

## Data

| Source | Rows | Main information |
|---|---:|---|
| Hospitalisation details | 2,343 | charges, date, children, hospital tier, city tier, state |
| Medical examinations | 2,335 | BMI, HBA1C, medical history, surgeries, smoking status |

A third supplied source contained names. It is intentionally excluded from the model and public repository so that identity data remains separate from analytical features.

The analytical raw files are not redistributed until their publication and licensing terms are documented. `data/raw/README.md` lists the filenames required for local reproduction.

### Data-quality issues handled

- repeated placeholder customer identifier `?`
- hospitalisation identifiers without a matching medical examination
- unknown year and smoking-status values represented by `?`
- inconsistent binary labels such as `Yes`, `yes`, and `No`
- surgery counts stored as text

Unknown values remain missing rather than being silently invented.

## ML Workflow

```text
Raw analytical sources
        ↓
Schema + join validation
        ↓
Cleaning + type normalization
        ↓
Grouped split by Customer ID
        ↓
Train-only preprocessing
        ↓
┌──────────────────────┬─────────────────────────┐
│ Cost regression      │ High-cost classification│
└──────────────────────┴─────────────────────────┘
        ↓
5-fold grouped cross-validation
        ↓
Final untouched test evaluation
        ↓
Error analysis + permutation importance
```

Imputation, encoding, scaling, model selection, and high-cost threshold estimation are all performed without using the final test set.

## Model Comparison

### Cost Regression

Models compared:

- Mean baseline
- Ridge Regression
- Random Forest Regressor
- Histogram Gradient Boosting Regressor
- Histogram Gradient Boosting with log-transformed target

**Selected model — Random Forest Regressor**

| Metric | Test result |
|---|---:|
| MAE | **1,663.17** |
| RMSE | **3,037.04** |
| R² | **0.926** |

### High-Cost Classification

Models compared:

- Prior-probability baseline
- Logistic Regression
- Random Forest Classifier
- Histogram Gradient Boosting Classifier

**Selected model — Histogram Gradient Boosting Classifier**

| Metric | Test result |
|---|---:|
| Precision | **0.927** |
| Recall | **0.884** |
| F1 | **0.905** |
| ROC-AUC | **0.995** |
| PR-AUC | **0.974** |

PR-AUC and recall are emphasized because the positive high-cost class is intentionally rare.

## Notebook Guide

| Notebook | Purpose |
|---|---|
| `01_data_understanding.ipynb` | inspect schemas, keys, source dimensions, and join behaviour |
| `02_data_quality_and_cleaning.ipynb` | validate sources, normalize values, and build the analytical dataset |
| `03_exploratory_analysis.ipynb` | examine charge distribution and descriptive relationships |
| `04_feature_engineering.ipynb` | define model features and leakage-safe preprocessing |
| `05_cost_regression.ipynb` | compare regressors and evaluate the selected model |
| `06_high_cost_classification.ipynb` | define the train-only risk threshold and compare classifiers |
| `07_model_explainability_and_error_analysis.ipynb` | inspect prediction errors and permutation importance |

Reusable implementation lives in `src/`; notebooks remain focused on analysis and interpretation.

## Repository Structure

```text
Healthcare-Insurance-Risk-Analysis/
├── data/raw/README.md
├── notebooks/
├── outputs/
│   ├── figures/
│   └── metrics/
├── src/
├── tests/
├── requirements.txt
└── README.md
```

## Run Locally

Place the two analytical CSV files listed in `data/raw/README.md` inside `data/raw/`, then run:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
pytest -q
jupyter notebook
```

Run notebooks `01` through `07` in order.

## Limitations

- The dataset is relatively small and does not establish clinical validity.
- Some hospitalisation rows do not have a matching medical examination record.
- Feature importance measures predictive contribution, not causality.
- External validation would be required before generalising to real insurance populations.

## Technologies

**Python · pandas · NumPy · scikit-learn · Matplotlib · Jupyter · pytest · GitHub Actions**
