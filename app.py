import math
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split


# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Depression Risk Prediction Demo",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# -----------------------------
# Styling
# -----------------------------
st.markdown(
    """
    <style>
        .stApp {
            background: #f7f9fc;
        }

        .block-container {
            max-width: 1180px;
            padding-top: 1.2rem;
            padding-bottom: 3rem;
        }

        h1, h2, h3 {
            color: #14395b;
        }

        .um-header {
            background: linear-gradient(135deg, #0b2f4f 0%, #164f7d 100%);
            color: white;
            border-radius: 16px;
            padding: 24px 28px;
            margin-bottom: 18px;
            box-shadow: 0 8px 24px rgba(20, 57, 91, 0.12);
        }

        .um-kicker {
            font-size: 0.86rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            opacity: 0.85;
            margin-bottom: 6px;
        }

        .um-title {
            font-size: 2rem;
            font-weight: 750;
            line-height: 1.2;
            margin: 0;
        }

        .um-subtitle {
            font-size: 1rem;
            opacity: 0.92;
            margin-top: 9px;
            margin-bottom: 0;
        }

        .section-card {
            background: white;
            border: 1px solid #e3eaf2;
            border-radius: 14px;
            padding: 20px 22px;
            margin: 12px 0 18px 0;
            box-shadow: 0 5px 16px rgba(23, 57, 86, 0.06);
        }

        .step-row {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            align-items: center;
            margin-top: 12px;
        }

        .step {
            background: #edf4fa;
            border: 1px solid #d7e5f1;
            border-radius: 999px;
            padding: 8px 13px;
            color: #163f63;
            font-weight: 600;
            font-size: 0.91rem;
        }

        .arrow {
            color: #6a8095;
            font-weight: 700;
        }

        .predictor-chip {
            display: inline-block;
            background: #eef6fb;
            color: #124a74;
            border: 1px solid #d3e6f2;
            border-radius: 999px;
            padding: 7px 12px;
            margin: 4px 6px 4px 0;
            font-size: 0.91rem;
            font-weight: 600;
        }

        .small-note {
            color: #5d6d7e;
            font-size: 0.91rem;
            line-height: 1.5;
        }

        .result-card {
            border-radius: 16px;
            padding: 22px 24px;
            margin-top: 14px;
            border: 1px solid #dce6ef;
            background: white;
            box-shadow: 0 7px 20px rgba(23, 57, 86, 0.08);
        }

        .risk-low {
            border-left: 7px solid #2e7d65;
        }

        .risk-high {
            border-left: 7px solid #b44a4a;
        }

        .risk-label {
            font-size: 1.45rem;
            font-weight: 760;
            margin-bottom: 4px;
        }

        .risk-prob {
            font-size: 2.4rem;
            font-weight: 800;
            color: #14395b;
            line-height: 1.05;
        }

        .footer-note {
            background: #fffaf0;
            border: 1px solid #f0dfb8;
            border-radius: 12px;
            padding: 14px 16px;
            color: #68562e;
            font-size: 0.9rem;
            margin-top: 18px;
        }

        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #e3eaf2;
            padding: 12px;
            border-radius: 12px;
        }

        .stButton > button {
            width: 100%;
            border-radius: 10px;
            min-height: 46px;
            font-weight: 700;
            background: #164f7d;
            color: white;
            border: none;
        }

        .stButton > button:hover {
            background: #0f3e65;
            color: white;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# Model and data
# -----------------------------
FEATURES = [
    "age",
    "sleep_duration",
    "chronic_disease_count",
    "self_rated_health",
    "adl_limitation",
    "cognitive_score",
    "gender",
    "marital_status",
]

DISPLAY_NAMES = {
    "age": "Age",
    "sleep_duration": "Sleep Duration",
    "chronic_disease_count": "Chronic Disease Count",
    "self_rated_health": "Self-rated Health",
    "adl_limitation": "ADL Limitation",
    "cognitive_score": "Cognitive Score",
    "gender": "Gender",
    "marital_status": "Marital Status",
}

SELECTED_BY_RFE = [
    "age",
    "sleep_duration",
    "chronic_disease_count",
    "self_rated_health",
    "adl_limitation",
    "cognitive_score",
]


@st.cache_resource
def build_demo_model() -> Tuple[GradientBoostingClassifier, pd.DataFrame]:
    """
    Build a stable temporary demonstration model using synthetic CHARLS-like data.

    The final research implementation should replace this with the real CHARLS 2020
    preprocessing, RFE, XGBoost, Grid Search, evaluation, and SHAP pipeline.
    """
    rng = np.random.default_rng(42)
    n = 2600

    data = pd.DataFrame(
        {
            "age": rng.integers(60, 91, n),
            "sleep_duration": np.clip(rng.normal(6.6, 1.3, n), 3.0, 10.0),
            "chronic_disease_count": np.clip(rng.poisson(1.8, n), 0, 7),
            "self_rated_health": rng.integers(1, 6, n),
            "adl_limitation": np.clip(rng.poisson(0.8, n), 0, 5),
            "cognitive_score": np.clip(rng.normal(18.5, 4.2, n), 5.0, 30.0),
            "gender": rng.integers(0, 2, n),
            "marital_status": rng.integers(0, 2, n),
        }
    )

    linear_score = (
        -2.45
        + 0.035 * (data["age"] - 70)
        - 0.42 * (data["sleep_duration"] - 7)
        + 0.38 * data["chronic_disease_count"]
        + 0.50 * (data["self_rated_health"] - 3)
        + 0.34 * data["adl_limitation"]
        - 0.065 * (data["cognitive_score"] - 18)
        + 0.035 * data["gender"]
        - 0.045 * data["marital_status"]
        + 0.20 * (data["sleep_duration"] < 5).astype(float)
        + 0.15 * (
            (data["chronic_disease_count"] >= 3)
            & (data["adl_limitation"] >= 2)
        ).astype(float)
    )

    probability = 1.0 / (1.0 + np.exp(-linear_score))
    target = rng.binomial(1, probability)

    x_train, _, y_train, _ = train_test_split(
        data[FEATURES],
        target,
        test_size=0.20,
        random_state=42,
        stratify=target,
    )

    model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.045,
        max_depth=2,
        min_samples_leaf=18,
        random_state=42,
    )
    model.fit(x_train, y_train)
    return model, x_train


def get_local_shap(
    model: GradientBoostingClassifier,
    background: pd.DataFrame,
    sample: pd.DataFrame,
) -> pd.DataFrame:
    """Return robust local SHAP contributions for one binary-classification case."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)

    if isinstance(shap_values, list):
        values = np.asarray(shap_values[-1])[0]
    else:
        values = np.asarray(shap_values)
        if values.ndim == 3:
            values = values[0, :, -1]
        elif values.ndim == 2:
            values = values[0]
        else:
            values = values.reshape(-1)

    result = pd.DataFrame(
        {
            "Feature": [DISPLAY_NAMES[f] for f in FEATURES],
            "Value": [sample.iloc[0][f] for f in FEATURES],
            "SHAP value": values,
        }
    )
    result["Absolute contribution"] = result["SHAP value"].abs()
    return result.sort_values("Absolute contribution", ascending=False)


def format_value(feature: str, value: float) -> str:
    if feature == "gender":
        return "Female" if int(value) == 1 else "Male"
    if feature == "marital_status":
        return "Married / partnered" if int(value) == 1 else "Not married / not partnered"
    if feature in {"age", "chronic_disease_count", "self_rated_health", "adl_limitation"}:
        return str(int(value))
    return f"{float(value):.1f}"


def make_explanation_chart(explanation: pd.DataFrame):
    chart_data = explanation.head(8).sort_values("SHAP value")
    labels = chart_data["Feature"].tolist()
    values = chart_data["SHAP value"].tolist()
    colors = ["#b44a4a" if v > 0 else "#2e7d65" for v in values]

    fig, ax = plt.subplots(figsize=(8.4, 4.5))
    ax.barh(labels, values, color=colors)
    ax.axvline(0, color="#6c7a89", linewidth=1)
    ax.set_xlabel("SHAP contribution to the prediction")
    ax.set_ylabel("")
    ax.grid(axis="x", alpha=0.18)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    return fig


model, background_data = build_demo_model()


# -----------------------------
# Header
# -----------------------------
st.markdown(
    """
    <div class="um-header">
        <div class="um-kicker">Universiti Malaya · Master in Data Science</div>
        <div class="um-title">Predicting Depression Risk Among Older Adults</div>
        <p class="um-subtitle">
            Single-page demonstration of the proposed compact and explainable prediction framework
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="section-card">
        <h3 style="margin-top:0;">Proposed Framework</h3>
        <p class="small-note">
            The final research framework uses CHARLS 2020 data for adults aged 60 years and above.
            CES-D-related items will define the depression-risk outcome.
        </p>
        <div class="step-row">
            <span class="step">CHARLS 2020</span><span class="arrow">→</span>
            <span class="step">Pre-processing</span><span class="arrow">→</span>
            <span class="step">RFE</span><span class="arrow">→</span>
            <span class="step">Optimized classifier</span><span class="arrow">→</span>
            <span class="step">Evaluation</span><span class="arrow">→</span>
            <span class="step">SHAP explanation</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# Selected predictors
# -----------------------------
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.subheader("Selected Predictors by RFE")
st.caption(
    "The current demo highlights six core predictors to illustrate a compact screening-oriented feature set."
)
chips = "".join(
    f'<span class="predictor-chip">{DISPLAY_NAMES[feature]}</span>'
    for feature in SELECTED_BY_RFE
)
st.markdown(chips, unsafe_allow_html=True)
st.markdown(
    """
    <p class="small-note" style="margin-top:12px;">
        Gender and marital status are also included as contextual inputs in this temporary demonstration model,
        but they are intentionally assigned weaker predictive influence. The final selected feature set will be
        determined by RFE using the real CHARLS 2020 data.
    </p>
    """,
    unsafe_allow_html=True,
)
st.markdown("</div>", unsafe_allow_html=True)


# -----------------------------
# Manual input controls
# -----------------------------
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.subheader("Enter Participant Information")
st.caption("Adjust the values, then select “Predict Depression Risk”.")

left, right = st.columns(2, gap="large")

with left:
    age = st.slider("Age", min_value=60, max_value=90, value=68, step=1)
    sleep_duration = st.slider(
        "Sleep Duration (hours/night)",
        min_value=3.0,
        max_value=10.0,
        value=6.5,
        step=0.1,
    )
    chronic_disease_count = st.slider(
        "Chronic Disease Count",
        min_value=0,
        max_value=7,
        value=1,
        step=1,
    )
    self_rated_health = st.select_slider(
        "Self-rated Health",
        options=[1, 2, 3, 4, 5],
        value=3,
        help="1 = very good, 5 = poor",
    )

with right:
    adl_limitation = st.slider(
        "ADL Limitation Count",
        min_value=0,
        max_value=5,
        value=0,
        step=1,
        help="Number of reported limitations in activities of daily living.",
    )
    cognitive_score = st.slider(
        "Cognitive Score",
        min_value=5.0,
        max_value=30.0,
        value=18.0,
        step=0.5,
    )
    gender_label = st.selectbox("Gender", ["Male", "Female"], index=1)
    marital_label = st.selectbox(
        "Marital Status",
        ["Not married / not partnered", "Married / partnered"],
        index=1,
    )

gender = 1 if gender_label == "Female" else 0
marital_status = 1 if marital_label == "Married / partnered" else 0

participant = pd.DataFrame(
    [
        {
            "age": age,
            "sleep_duration": sleep_duration,
            "chronic_disease_count": chronic_disease_count,
            "self_rated_health": self_rated_health,
            "adl_limitation": adl_limitation,
            "cognitive_score": cognitive_score,
            "gender": gender,
            "marital_status": marital_status,
        }
    ]
)

predict_clicked = st.button("Predict Depression Risk", type="primary")
st.markdown("</div>", unsafe_allow_html=True)


# -----------------------------
# Result and explanation
# -----------------------------
if predict_clicked:
    predicted_probability = float(model.predict_proba(participant)[0, 1])
    threshold = 0.50
    higher_risk = predicted_probability >= threshold
    risk_label = "Higher Depression Risk" if higher_risk else "Lower Depression Risk"
    card_class = "risk-high" if higher_risk else "risk-low"

    st.markdown(
        f"""
        <div class="result-card {card_class}">
            <div class="risk-label">{risk_label}</div>
            <div class="risk-prob">{predicted_probability:.1%}</div>
            <div class="small-note">
                Predicted probability of meeting the CES-D-related depression-risk threshold
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.progress(min(max(predicted_probability, 0.0), 1.0))

    metric_1, metric_2, metric_3 = st.columns(3)
    metric_1.metric("Classification threshold", "50%")
    metric_2.metric("Predicted class", "Higher risk" if higher_risk else "Lower risk")
    metric_3.metric("Model used in this demo", "Gradient Boosting")

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Explanation of This Prediction")
    st.caption(
        "SHAP shows how each input pushes this individual prediction toward higher or lower risk."
    )

    local_explanation = get_local_shap(model, background_data, participant)

    positive = local_explanation[local_explanation["SHAP value"] > 0].copy()
    negative = local_explanation[local_explanation["SHAP value"] < 0].copy()

    chart_col, text_col = st.columns([1.35, 1], gap="large")

    with chart_col:
        fig = make_explanation_chart(local_explanation)
        st.pyplot(fig, clear_figure=True)
        st.caption(
            "Positive SHAP values push the prediction toward higher risk; negative values push it toward lower risk."
        )

    with text_col:
        st.markdown("#### Main factors increasing predicted risk")
        if positive.empty:
            st.write("No input produced a positive contribution for this case.")
        else:
            for _, row in positive.head(3).iterrows():
                feature_key = next(
                    key for key, name in DISPLAY_NAMES.items() if name == row["Feature"]
                )
                st.write(
                    f"• **{row['Feature']} = {format_value(feature_key, row['Value'])}** "
                    f"(SHAP {row['SHAP value']:+.3f})"
                )

        st.markdown("#### Main factors reducing predicted risk")
        if negative.empty:
            st.write("No input produced a negative contribution for this case.")
        else:
            for _, row in negative.head(3).iterrows():
                feature_key = next(
                    key for key, name in DISPLAY_NAMES.items() if name == row["Feature"]
                )
                st.write(
                    f"• **{row['Feature']} = {format_value(feature_key, row['Value'])}** "
                    f"(SHAP {row['SHAP value']:+.3f})"
                )

    with st.expander("View all input values and SHAP contributions"):
        full_table = local_explanation.copy()
        full_table["Direction"] = np.where(
            full_table["SHAP value"] > 0,
            "Toward higher risk",
            "Toward lower risk",
        )
        full_table["Input value"] = [
            format_value(
                next(key for key, name in DISPLAY_NAMES.items() if name == feature_name),
                value,
            )
            for feature_name, value in zip(full_table["Feature"], full_table["Value"])
        ]
        st.dataframe(
            full_table[
                ["Feature", "Input value", "SHAP value", "Direction"]
            ].style.format({"SHAP value": "{:+.3f}"}),
            use_container_width=True,
            hide_index=True,
        )

    st.markdown(
        """
        <p class="small-note">
            Interpretation note: SHAP explains this model’s prediction for the selected case.
            It does not prove that any predictor causes depression.
        </p>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)


# -----------------------------
# Research limitation note
# -----------------------------
st.markdown(
    """
    <div class="footer-note">
        <strong>Research demo only.</strong>
        This website uses synthetic CHARLS-like data and a temporary Gradient Boosting classifier
        to demonstrate the planned workflow. It is not a clinical diagnosis tool. The final study
        will use CHARLS 2020, CES-D-related items, RFE, Grid Search, XGBoost, classification metrics,
        and SHAP-based global and local explanations.
    </div>
    """,
    unsafe_allow_html=True,
)
