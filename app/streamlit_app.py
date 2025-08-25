import streamlit as st
import pandas as pd
from pathlib import Path
import joblib

DEFAULT_MODEL = Path(__file__).resolve().parents[1] / "models"
DEFAULT_REPORTS = Path(__file__).resolve().parents[1] / "reports"

st.set_page_config(page_title="Credit Risk Prediction", layout="wide")
st.title("💳 Credit Risk Prediction Dashboard")

st.markdown("""
Upload applicant data (CSV) to score **default probability**.
If no model is found yet, run `python -m src.train` to train and create artifacts.
""")

# Sidebar
model_files = list(DEFAULT_MODEL.glob("*_pipeline.joblib"))
model_path = None
if model_files:
    model_names = [p.name for p in model_files]
    sel = st.sidebar.selectbox("Select Trained Model", model_names, index=0)
    model_path = DEFAULT_MODEL / sel
else:
    st.warning("No trained model found in 'models/'. Please train first.")

uploaded = st.file_uploader("Upload CSV", type=["csv"])

if uploaded and model_path:
    df = pd.read_csv(uploaded)
    pipe = joblib.load(model_path)
    proba = pipe.predict_proba(df)[:,1]
    preds = (proba >= 0.5).astype(int)
    out = df.copy()
    out["default_prob"] = proba
    out["predicted_risk"] = preds
    st.subheader("Scored Applicants")
    st.dataframe(out.head(100))
    st.download_button("Download Scored CSV", out.to_csv(index=False).encode("utf-8"), file_name="scored_applicants.csv")
    st.caption("Prediction threshold = 0.5")

st.sidebar.subheader("Reports")
roc = DEFAULT_REPORTS / "roc_curve.png"
cm = DEFAULT_REPORTS / "confusion_matrix.png"
if roc.exists():
    st.sidebar.image(str(roc), caption="ROC Curve")
if cm.exists():
    st.sidebar.image(str(cm), caption="Confusion Matrix")
