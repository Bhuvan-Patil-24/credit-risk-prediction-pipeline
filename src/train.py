import argparse, warnings
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from .pipeline import build_preprocessing_pipeline
from .evaluate import evaluate_binary, save_roc_curve, save_confusion_matrix, dump_json
from .config import TARGET_COL

warnings.filterwarnings("ignore", category=UserWarning)

def train_models(df: pd.DataFrame, reports_dir: Path, models_dir: Path):
    assert TARGET_COL in df.columns, f"Missing target column {TARGET_COL}"
    y = df[TARGET_COL].astype(int).values
    X = df.drop(columns=[TARGET_COL])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pre = build_preprocessing_pipeline()

    candidates = {}

    # Logistic Regression
    lr = Pipeline(steps=[("pre", pre), ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))])
    lr.fit(X_train, y_train)
    y_prob = lr.predict_proba(X_test)[:,1]
    metrics_lr = evaluate_binary(y_test, y_prob)
    candidates["logreg"] = {"estimator": lr, "metrics": metrics_lr}

    # Random Forest
    rf = Pipeline(steps=[("pre", pre), ("clf", RandomForestClassifier(
        n_estimators=300, max_depth=None, min_samples_leaf=2, class_weight="balanced_subsample", random_state=42
    ))])
    rf.fit(X_train, y_train)
    y_prob = rf.predict_proba(X_test)[:,1]
    metrics_rf = evaluate_binary(y_test, y_prob)
    candidates["random_forest"] = {"estimator": rf, "metrics": metrics_rf}

    # XGBoost (optional)
    try:
        from xgboost import XGBClassifier
        scale_pos_weight = (y_train==0).sum() / max(1,(y_train==1).sum())
        xgb = Pipeline(steps=[("pre", pre), ("clf", XGBClassifier(
            n_estimators=500, learning_rate=0.05, max_depth=4, subsample=0.9, colsample_bytree=0.8,
            eval_metric="logloss", reg_lambda=1.0, random_state=42, n_jobs=4, scale_pos_weight=scale_pos_weight
        ))])
        xgb.fit(X_train, y_train)
        y_prob = xgb.predict_proba(X_test)[:,1]
        metrics_xgb = evaluate_binary(y_test, y_prob)
        candidates["xgboost"] = {"estimator": xgb, "metrics": metrics_xgb}
    except Exception as e:
        print("XGBoost not available -> skipping.", e)

    # pick best by ROC-AUC
    best_name, best = max(candidates.items(), key=lambda kv: kv[1]["metrics"]["roc_auc"])
    print("Best model:", best_name, best["metrics"])

    # Save artifacts
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(best["estimator"], models_dir / f"{best_name}_pipeline.joblib")
    dump_json(best["metrics"], reports_dir / "best_metrics.json")

    # save global comparisons
    all_metrics = {k: v["metrics"] for k,v in candidates.items()}
    dump_json(all_metrics, reports_dir / "all_metrics.json")

    # Save ROC and Confusion Matrix for best
    # Need transformed test set to use estimator directly
    try:
        from .evaluate import save_roc_curve, save_confusion_matrix
        save_roc_curve(best["estimator"], X_test, y_test, reports_dir / "roc_curve.png")
        save_confusion_matrix(best["metrics"], reports_dir / "confusion_matrix.png")
    except Exception as e:
        print("Plotting failed:", e)

    return best_name

def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/credit_risk_dataset.csv")
    parser.add_argument("--reports_dir", type=str, default="reports")
    parser.add_argument("--models_dir", type=str, default="models")
    ns = parser.parse_args(args=args)

    data_path = Path(ns.data_path)
    df = pd.read_csv(data_path)

    best = train_models(df, Path(ns.reports_dir), Path(ns.models_dir))
    print("Training complete. Best:", best)

if __name__ == "__main__":
    main()
