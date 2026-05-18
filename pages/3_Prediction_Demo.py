import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier

st.title("Prediction Demo")

st.markdown("""
This page demonstrates the training and prediction process using sample CHARLS-like data.

The final study will use CHARLS 2020 and CES-D-related depression outcome.
""")

@st.cache_data
def load_data():
    return pd.read_csv("data/sample_charls_demo.csv")

df = load_data()

X = df.drop(columns=["depression_risk"])
y = df["depression_risk"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

base_model = GradientBoostingClassifier(
    n_estimators=80,
    max_depth=3,
    learning_rate=0.08,
    random_state=42
)

selector = RFE(
    estimator=base_model,
    n_features_to_select=6,
    step=1
)

selector.fit(X_train, y_train)

selected_features = X.columns[selector.support_].tolist()

X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]

model = GradientBoostingClassifier(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.08,
    random_state=42
)

model.fit(X_train_selected, y_train)

y_pred = model.predict(X_test_selected)
y_prob = model.predict_proba(X_test_selected)[:, 1]

st.subheader("Selected Predictors by RFE")
st.write(selected_features)

st.subheader("Model Evaluation")

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.3f}")
col2.metric("Precision", f"{precision_score(y_test, y_pred):.3f}")
col3.metric("Recall", f"{recall_score(y_test, y_pred):.3f}")
col4.metric("F1-score", f"{f1_score(y_test, y_pred):.3f}")
col5.metric("AUC", f"{roc_auc_score(y_test, y_prob):.3f}")

st.write("Confusion Matrix")
st.write(confusion_matrix(y_test, y_pred))

st.subheader("Individual Prediction")

with st.form("prediction_form"):
    age = st.slider("Age", 60, 90, 70)
    gender = st.selectbox("Gender", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
    education_level = st.slider("Education Level", 0, 3, 1)
    marital_status = st.selectbox("Marital Status", [0, 1], format_func=lambda x: "Not married" if x == 0 else "Married")
    sleep_duration = st.slider("Sleep Duration", 0.0, 12.0, 6.5)
    chronic_disease_count = st.slider("Chronic Disease Count", 0, 6, 2)
    self_rated_health = st.slider("Self-rated Health", 1, 5, 3)
    adl_limitation = st.selectbox("ADL Limitation", [0, 1])
    iadl_limitation = st.selectbox("IADL Limitation", [0, 1])
    cognitive_score = st.slider("Cognitive Score", 0.0, 30.0, 15.0)
    life_satisfaction = st.slider("Life Satisfaction", 1, 5, 3)

    submitted = st.form_submit_button("Predict Depression Risk")

if submitted:
    input_data = pd.DataFrame([{
        "age": age,
        "gender": gender,
        "education_level": education_level,
        "marital_status": marital_status,
        "sleep_duration": sleep_duration,
        "chronic_disease_count": chronic_disease_count,
        "self_rated_health": self_rated_health,
        "adl_limitation": adl_limitation,
        "iadl_limitation": iadl_limitation,
        "cognitive_score": cognitive_score,
        "life_satisfaction": life_satisfaction
    }])

    input_selected = input_data[selected_features]

    prediction = model.predict(input_selected)[0]
    probability = model.predict_proba(input_selected)[0, 1]

    if prediction == 1:
        st.error(f"Predicted Result: Higher depression risk. Probability: {probability:.3f}")
    else:
        st.success(f"Predicted Result: Lower depression risk. Probability: {probability:.3f}")

    st.caption("This result is for screening support only, not clinical diagnosis.")