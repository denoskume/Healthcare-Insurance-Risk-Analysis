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

| Task | Selected model | Main result |
|---|---|---:|
| Hospitalisation cost regression | Random Forest Regressor | **R² = 0.926** |
| High-cost risk classification | Histogram Gradient Boosting Classifier | **PR-AUC = 0.974** |

Additional test results:

- Regression: **MAE = 1,663.17**, **RMSE = 3,037.04**
- Classification: **Precision = 0.927**, **Recall = 0.884**, **F1 = 0.905**, **ROC-AUC = 0.995**
- High-cost threshold: **34,962.10**, learned from the training set only

## What This Project Demonstrates

- integration of hospitalisation and medical examination data
- explicit data-quality validation and cleaning
- leakage-safe preprocessing
- grouped train/test splitting by patient identifier
- regression and imbalanced classification workflows
- 5-fold grouped cross-validation
- baseline and model comparison
- error analysis and permutation importance
- reusable Python modules with automated tests
- clear separation between identity data and analytical features

## Problem

The project addresses two related machine learning tasks:

1. **Cost regression** — predict hospitalisation charges.
2. **High-cost classification** — identify hospitalisation events in the top 10% of charges.

The classification threshold is estimated from the training partition only. The final test set remains untouched during preprocessing, threshold definition, and model selection.

> This repository is an analytical machine learning project, not a clinical diagnosis system.

## Data

| Source | Rows | Main information |
|---|---:|---|
| Hospitalisation details | 2,343 | charges, date, children, hospital tier, city tier, state |
| Medical examinations | 2,335 | BMI, HBA1C, medical history, surgeries, smoking status |

A third supplied source contained names. It is intentionally excluded from both the public repository and model features so that identity data stays separate from analytical data.

The raw analytical files are not redistributed until their publication and licensing terms are documented. `data/raw/README.md` lists the expected filenames for local reproduction.

### Data-quality issues handled

- placeholder customer identifier `?` appearing repeatedly
- hospitalisation identifiers without a matching medical examination
- unknown year values represented by `?`
- unknown smoking-status values represented by `?`
- inconsistent binary labels such as `Yes`, `yes`, and `No`
- surgery counts stored as text

Unknown values remain missing rather than being silently invented.

## Workflow

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

The same customer cannot appear in both train and test partitions. Imputation, encoding, scaling, model selection, and high-cost threshold estimation are performed without using the final test set.

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

High cost is defined using the **90th percentile of training charges only**.

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

## Explainability and Error Analysis

The final analysis examines:

- actual vs predicted hospitalisation charges
- largest regression errors
- false positives and false negatives
- permutation importance for both selected models

The strongest predictive signals include smoking status, BMI, year, and hospital tier. These are **predictive associations within this dataset**, not medical causal claims.

## Notebook Guide

| Notebook | Purpose |
|---|---|
| `01_data_understanding.ipynb` | inspect schemas, distributions, keys, and join behaviour |
| `02_data_quality_and_cleaning.ipynb` | normalize values and build the analytical dataset |
| `03_exploratory_analysis.ipynb` | examine cost distribution and feature relationships |
| `04_feature_engineering.ipynb` | define model features and leakage-safe preprocessing |
| `05_cost_regression.ipynb` | compare regressors and evaluate the selected model |
| `06_high_cost_classification.ipynb` | define train-only risk threshold and compare classifiers |
| `07_model_explainability_and_error_analysis.ipynb` | inspect errors and permutation importance |

Reusable implementation lives in `src/`; the notebooks focus on analysis and interpretation.

## Repository Structure

```text
Healthcare-Insurance-Risk-Analysis/
├── data/raw/README.md
├── notebooks/
├── outputs/metrics/
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
- Feature importance reflects predictive contribution, not causality.
- External validation would be required before generalising to real insurance populations.

## Technologies

**Python · pandas · NumPy · scikit-learn · Matplotlib · Jupyter · pytest**
