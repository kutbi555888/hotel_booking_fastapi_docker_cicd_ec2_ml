from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

def calculate_binary_metrics(y_true, y_pred, y_proba) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
    }

def save_classification_report(y_true, y_pred, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    report_text = classification_report(y_true, y_pred, digits=4, zero_division=0)
    path.write_text(report_text, encoding="utf-8")

def metrics_to_frame(metrics_dict: dict[str, dict[str, float]]) -> pd.DataFrame:
    return pd.DataFrame(metrics_dict).T.reset_index().rename(columns={"index": "model_name"})
