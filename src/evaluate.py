import json
import numpy as np
from pathlib import Path
from typing import Dict, Any
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, RocCurveDisplay
)
import matplotlib.pyplot as plt

def evaluate_binary(y_true, y_prob, threshold=0.5) -> Dict[str, Any]:
    y_pred = (y_prob >= threshold).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "threshold": threshold
    }
    return metrics

def save_roc_curve(estimator, X_test, y_test, out_path: Path):
    y_prob = estimator.predict_proba(X_test)[:,1]
    RocCurveDisplay.from_predictions(y_test, y_prob)
    plt.title("ROC Curve")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def save_confusion_matrix(metrics: Dict[str, Any], out_path: Path):
    import seaborn as sns
    import matplotlib.pyplot as plt
    cm = np.array(metrics["confusion_matrix"])
    fig = plt.figure()
    sns.heatmap(cm, annot=True, fmt="d", cbar=False)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

def dump_json(obj, path: Path):
    path.write_text(json.dumps(obj, indent=2))
