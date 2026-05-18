import streamlit as st
import pandas as pd
import shap

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.ensemble import GradientBoostingClassifier

st.title("Simple SHAP Explanation")

st.markdown("""
This page gives a simple explanation of the prediction result.

SHAP is used to show which selected predictors have stronger influence on the model output.
In the final study, SHAP will be used to explain the XGBoost-based depression risk prediction model.
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

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test_selected, check_additivity=False)

st.subheader("1. Selected Predictors by RFE")

for feature in selected_features:
    st.write(f"- {feature}")

st.markdown("""
These are the predictors selected by RFE. They are used as the input variables for the prediction model.
""")

st.subheader("2. Main Important Predictors")

mean_abs_shap = abs(shap_values).mean(axis=0)

importance_df = pd.DataFrame({
    "Predictor": selected_features,
    "SHAP Importance": mean_abs_shap
}).sort_values("SHAP Importance", ascending=False)

importance_df["SHAP Importance"] = importance_df["SHAP Importance"].round(3)

top_importance = importance_df.head(6)

st.table(top_importance)

top_predictor = top_importance.iloc[0]["Predictor"]

st.markdown(f"""
Based on this demo model, **{top_predictor}** has the strongest overall influence on the prediction result.

A larger SHAP importance value means that the predictor has a stronger effect on the model output.
""")

st.subheader("3. Simple Explanation for One Case")

case_index = st.slider("Select one case", 0, len(X_test_selected) - 1, 0)

case = X_test_selected.iloc[[case_index]]
case_pred = model.predict(case)[0]
case_prob = model.predict_proba(case)[0, 1]

st.write("Selected case values:")
st.table(case.T.rename(columns={case.index[0]: "Value"}))

if case_pred == 1:
    st.error(f"Predicted Result: Higher depression risk. Probability: {case_prob:.3f}")
else:
    st.success(f"Predicted Result: Lower depression risk. Probability: {case_prob:.3f}")

case_shap = pd.DataFrame({
    "Predictor": selected_features,
    "SHAP Contribution": shap_values[case_index]
})

risk_increasing = case_shap[case_shap["SHAP Contribution"] > 0].sort_values(
    "SHAP Contribution", ascending=False
).head(3)

risk_decreasing = case_shap[case_shap["SHAP Contribution"] < 0].sort_values(
    "SHAP Contribution", ascending=True
).head(3)

risk_increasing["SHAP Contribution"] = risk_increasing["SHAP Contribution"].round(3)
risk_decreasing["SHAP Contribution"] = risk_decreasing["SHAP Contribution"].round(3)

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Factors pushing toward higher risk")
    if len(risk_increasing) > 0:
        st.table(risk_increasing)
    else:
        st.write("No main increasing factor in this case.")

with col2:
    st.markdown("#### Factors pushing toward lower risk")
    if len(risk_decreasing) > 0:
        st.table(risk_decreasing)
    else:
        st.write("No main decreasing factor in this case.")

st.info(
    "Positive SHAP values push the prediction toward higher depression risk. "
    "Negative SHAP values push the prediction toward lower depression risk."
)