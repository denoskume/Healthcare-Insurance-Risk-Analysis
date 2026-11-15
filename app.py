"""RiskLens — interactive showcase and private-session model laboratory."""
from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src.data import build_analytical_dataset
from src.evaluation import classification_metrics, regression_metrics
from src.features import build_preprocessor, prepare_feature_frame
from src.modeling import (
    derive_high_cost_labels,
    get_classification_candidates,
    get_regression_candidates,
    grouped_train_test_split,
)
from src.validation import (
    clean_analytical_dataset,
    validate_source_schema,
)

ROOT = Path(__file__).resolve().parent
METRICS = ROOT / "outputs" / "metrics" / "final_test_metrics.json"
FIGURES = ROOT / "outputs" / "figures"

st.set_page_config(
    page_title="RiskLens | Insurance Predictive Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp {background:#0b1220;color:#e9edf6;}
    section[data-testid="stSidebar"] {background:#101b2e;}
    .block-container {max-width:1240px;padding-top:2rem;padding-bottom:3rem;}
    .hero {background:linear-gradient(105deg,#17314b,#102340 62%,#143750);
        padding:30px 34px;border:1px solid #284666;border-radius:18px;margin-bottom:22px;}
    .eyebrow {font-size:12px;font-weight:700;letter-spacing:2px;color:#85d7d2;}
    .hero h1 {font-size:36px;color:white;margin:6px 0 8px;}
    .hero p {font-size:15px;color:#cbd7ea;max-width:760px;margin:0;}
    .info-card {padding:17px 20px;border-radius:12px;border:1px solid #29425f;
        background:#12223a;margin-bottom:12px;}
    .info-card h3 {color:#f7fafc;margin:0 0 6px;font-size:16px;}
    .info-card p {color:#c3d0e1;margin:0;font-size:14px;}
    [data-testid="stMetric"] {background:#14243a;border:1px solid #29415b;
        border-radius:13px;padding:15px;}
    [data-testid="stMetricLabel"] {color:#b6c9df;}
    h2,h3 {color:#f2f6ff;}
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_data
def historical_results() -> dict:
    return json.loads(METRICS.read_text(encoding="utf-8"))


def fit_session_models(hospital_file: bytes, exam_file: bytes) -> dict:
    hospital = pd.read_csv(io.BytesIO(hospital_file))
    exam = pd.read_csv(io.BytesIO(exam_file))
    sources = {"hospitalisations": hospital, "examinations": exam}
    validate_source_schema(sources)
    combined, diagnostics = build_analytical_dataset(hospital, exam)
    cleaned = clean_analytical_dataset(combined)
    if len(cleaned) < 100:
        raise ValueError("At least 100 valid records are required for a meaningful demonstration.")
    features, charges, groups = prepare_feature_frame(cleaned)
    if groups.nunique() < 8:
        raise ValueError("Not enough distinct customer groups for grouped evaluation.")
    x_train, x_test, y_train, y_test, train_groups, _ = grouped_train_test_split(
        features, charges, groups
    )
    prep = build_preprocessor(x_train)
    reg = get_regression_candidates(prep)["Random Forest"]
    reg.fit(x_train, y_train)
    reg_scores = regression_metrics(y_test, reg.predict(x_test))

    y_train_risk, y_test_risk, threshold = derive_high_cost_labels(y_train, y_test)
    if y_train_risk.nunique() < 2 or y_test_risk.nunique() < 2:
        raise ValueError("Both risk classes must be present in train and test subsets.")
    clf = get_classification_candidates(prep)["HistGradientBoosting"]
    clf.fit(x_train, y_train_risk)
    probability = clf.predict_proba(x_test)[:, 1]
    cls_scores = classification_metrics(y_test_risk, probability, (probability >= 0.5).astype(int))
    return {
        "reg": reg,
        "clf": clf,
        "features": features,
        "threshold": threshold,
        "rows": len(cleaned),
        "groups": groups.nunique(),
        "diagnostics": diagnostics,
        "reg_scores": reg_scores,
        "cls_scores": cls_scores,
    }


with st.sidebar:
    st.markdown("### ◈ RISK<span style='color:#70d3ce'>LENS</span>", unsafe_allow_html=True)
    st.caption("Healthcare Insurance Risk Analysis")
    st.markdown("---")
    st.markdown("**Built with**")
    st.write("Python · pandas · scikit-learn · Streamlit")
    st.markdown("---")
    st.caption("Research portfolio application. Not intended for clinical decision-making, insurance pricing, or real-person risk decisions.")
    st.link_button("Source on GitHub", "https://github.com/denoskume/Healthcare-Insurance-Risk-Analysis")

st.markdown(
    """<div class="hero">
    <div class="eyebrow">DATA SCIENCE · MACHINE LEARNING · DEPLOYMENT</div>
    <h1>Healthcare Risk Intelligence</h1>
    <p>From messy source records to explainable predictive analytics.
    Explore verified benchmark results, then train and test models using your own authorized datasets.</p>
    </div>""",
    unsafe_allow_html=True,
)

historical = historical_results()
regression = historical["regression"]
classification = historical["classification"]

overview, laboratory, methodology = st.tabs(["📊 Model performance", "🧪 Train & predict", "🔍 Methodology"])

with overview:
    st.caption("Previously generated test-set results from the repository, not live inference.")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Cost prediction · R²", f'{regression["r2"]:.3f}')
    m2.metric("Cost prediction · MAE", f'{regression["mae"]:,.0f}')
    m3.metric("High-cost detection · PR-AUC", f'{classification["pr_auc"]:.3f}')
    m4.metric("High-cost detection · F1", f'{classification["f1"]:.3f}')

    lcol, rcol = st.columns(2, gap="large")
    with lcol:
        st.subheader("Cost regression")
        st.caption("Random Forest Regressor")
        fig = FIGURES / "actual_vs_predicted_regression.png"
        if fig.exists():
            st.image(str(fig), use_container_width=True)
    with rcol:
        st.subheader("High-cost classification")
        st.caption("Histogram Gradient Boosting Classifier")
        fig = FIGURES / "classification_pr_curve.png"
        if fig.exists():
            st.image(str(fig), use_container_width=True)

    st.subheader("Interpretability")
    lcol, rcol = st.columns(2, gap="large")
    with lcol:
        p = FIGURES / "regression_feature_importance.png"
        if p.exists():
            st.image(str(p), use_container_width=True)
    with rcol:
        p = FIGURES / "classification_feature_importance.png"
        if p.exists():
            st.image(str(p), use_container_width=True)

    st.info("These metrics were obtained offline using the original analytical inputs. The dataset itself is not distributed with this site.")

with laboratory:
    st.subheader("Private-session model laboratory")
    st.write("Upload two CSV files in the same schema as the original research project. Models train in the current session, and uploaded rows are not written to this repository.")
    st.warning("Use only datasets you are authorized to upload. Do not upload names or other personally identifying medical records to a public demo.")
    u1, u2 = st.columns(2)
    with u1:
        hospital_upload = st.file_uploader("Hospitalisation details.csv", type=["csv"], key="hospital")
    with u2:
        exam_upload = st.file_uploader("Medical Examinations.csv", type=["csv"], key="exam")

    enabled = hospital_upload is not None and exam_upload is not None
    if st.button("Train models on uploaded data", type="primary", disabled=not enabled):
        if hospital_upload.size > 5_000_000 or exam_upload.size > 5_000_000:
            st.error("Each CSV must be 5 MB or smaller.")
        else:
            with st.spinner("Cleaning, splitting by customer, and training both models..."):
                try:
                    st.session_state["trained_models"] = fit_session_models(
                        hospital_upload.getvalue(), exam_upload.getvalue()
                    )
                except (ValueError, KeyError, TypeError, pd.errors.ParserError) as err:
                    st.session_state.pop("trained_models", None)
                    st.error(f"Unable to train: {err}")
                except Exception:
                    st.session_state.pop("trained_models", None)
                    st.error("Training failed. Check source columns and data quality.")

    trained = st.session_state.get("trained_models")
    if trained:
        st.success("Models trained and evaluated on this session's uploaded data.")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Records analyzed", f'{trained["rows"]:,}')
        k2.metric("Distinct customers", f'{trained["groups"]:,}')
        k3.metric("Session test R²", f'{trained["reg_scores"]["r2"]:.3f}')
        k4.metric("Session test PR-AUC", f'{trained["cls_scores"]["pr_auc"]:.3f}')

        st.subheader("Predict an example record")
        st.caption("Input fields follow the training schema; the prediction is exploratory only.")
        frame = trained["features"]
        row = {}
        with st.form("predict_record"):
            left, right = st.columns(2)
            for idx, feature in enumerate(frame.columns):
                with (left if idx % 2 == 0 else right):
                    col = frame[feature]
                    if pd.api.types.is_numeric_dtype(col):
                        median = pd.to_numeric(col, errors="coerce").median()
                        default = float(median) if pd.notna(median) and np.isfinite(median) else 0.0
                        row[feature] = st.number_input(feature, value=default, format="%.2f")
                    else:
                        categories = sorted(str(x) for x in col.dropna().unique())
                        if not categories:
                            categories = ["Unknown"]
                        row[feature] = st.selectbox(feature, options=categories)
            predict = st.form_submit_button("Run predictions", type="primary")
        if predict:
            sample = pd.DataFrame([row], columns=frame.columns)
            prediction = float(trained["reg"].predict(sample)[0])
            probability = float(trained["clf"].predict_proba(sample)[0, 1])
            p1, p2 = st.columns(2)
            p1.metric("Predicted hospitalisation charges", f"{prediction:,.2f}")
            p2.metric("Estimated high-cost probability", f"{probability:.1%}")
            st.caption(
                f"High-cost cases are defined relative to the training set (90th-percentile threshold: {trained['threshold']:,.2f} dataset units). "
                "The dataset does not document a currency; no currency is assumed."
            )
    else:
        st.markdown(
            """<div class="info-card"><h3>Ready to train</h3>
            <p>Upload both original-schema CSV files to enable model training and interactive prediction.
            Historical metrics are available in the first tab without uploading anything.</p></div>""",
            unsafe_allow_html=True,
        )

with methodology:
    st.subheader("Reproducible modelling workflow")
    st.markdown(
        "1. Validate and join hospitalisation and medical-examination records by customer ID.\n"
        "2. Normalize missing values and inconsistent labels; keep unnamed identity data out of modeling.\n"
        "3. Split train/test by customer group to prevent customer overlap.\n"
        "4. Fit preprocessing only on training data; develop cost regression and high-cost classification.\n"
        "5. Evaluate using R², MAE, PR-AUC, F1 and held-out test data."
    )
    st.markdown("**Model choices:** Random Forest Regressor and Histogram Gradient Boosting Classifier.")
    st.markdown("**Limits:** Small observational dataset; model output is not causal and requires external validation.")
    st.markdown("**Data governance:** Original CSV files are intentionally excluded from public GitHub due to unresolved licensing. Uploads are only held in the current application session.")
    st.link_button("View model code & research notebooks", "https://github.com/denoskume/Healthcare-Insurance-Risk-Analysis")
