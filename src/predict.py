from pathlib import Path
import pandas as pd
import joblib

def load_model(model_path: str):
    return joblib.load(model_path)

def predict_df(df: pd.DataFrame, model_path: str):
    pipe = load_model(model_path)
    proba = pipe.predict_proba(df)[:,1]
    preds = (proba >= 0.5).astype(int)
    out = df.copy()
    out["default_prob"] = proba
    out["predicted_risk"] = preds
    return out
