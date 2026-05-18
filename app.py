import streamlit as st

st.set_page_config(
    page_title="Depression Risk Prediction Demo",
    layout="wide"
)

st.title("Depression Risk Prediction Demo")
st.subheader("RFE-based Feature Selection + XGBoost Classification + SHAP Interpretation")

st.markdown("""
This demo is based on the proposed framework in Chapter 1 to Chapter 3.

The framework includes:

1. CHARLS 2020 survey-based tabular data  
2. Data pre-processing  
3. RFE-based feature selection  
4. XGBoost classification  
5. Model evaluation  
6. SHAP-based interpretation  

**Note:** This demo is for research and screening support only. It is not a clinical diagnosis tool.
""")

st.info(
    "Current demo uses sample CHARLS-like data. "
    "The final implementation will use CHARLS 2020 and CES-D-related depressive symptom items."
)