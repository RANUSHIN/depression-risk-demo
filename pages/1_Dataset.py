import streamlit as st
import pandas as pd

st.title("Dataset")

st.markdown("""
### Dataset Used in the Study

The proposed study uses the **China Health and Retirement Longitudinal Study (CHARLS) 2020** dataset.

CHARLS 2020 is a **survey-based tabular dataset**. It provides structured information on middle-aged and older adults in China.

### Research Population

This study focuses on older adults aged **60 years and above**.

### Outcome

The final study will define the depression-related outcome based on **CES-D-related depressive symptom items** available in CHARLS 2020.

### Input Variables

The input variables may include:

- Socio-demographic variables
- Behavioural variables
- Health-related variables
- Functional variables
- Cognitive variables
- Psychological variables

### Research Boundary

This study does not use image data, speech data, social media text data, clinical interview records, or biological signal data.
""")

st.subheader("Sample Demo Data")

df = pd.read_csv("data/sample_charls_demo.csv")
st.dataframe(df.head(20))

st.write("Shape of sample data:", df.shape)