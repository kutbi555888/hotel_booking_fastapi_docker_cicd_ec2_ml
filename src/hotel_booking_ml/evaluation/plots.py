from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    confusion_matrix,
)

def save_target_distribution_plot(target_series: pd.Series, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    target_series.value_counts().sort_index().plot(kind="bar", ax=ax)
    ax.set_title("Target taqsimoti")
    ax.set_xlabel("is_canceled")
    ax.set_ylabel("Soni")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

def save_missing_values_plot(missing_df: pd.DataFrame, path: Path) -> None:
    top_missing = missing_df.head(10)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(top_missing["column"], top_missing["missing_pct"])
    ax.set_title("Eng ko‘p missing bo‘lgan ustunlar")
    ax.set_xlabel("Missing %")
    ax.set_ylabel("Ustun")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

def save_boxplot_by_target(df: pd.DataFrame, column: str, target: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    df.boxplot(column=column, by=target, ax=ax)
    ax.set_title(f"{column} va {target}")
    ax.set_xlabel(target)
    ax.set_ylabel(column)
    fig.suptitle("")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

def save_monthly_cancellation_rate_plot(month_df: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(month_df["arrival_date_month"], month_df["cancellation_rate"])
    ax.set_title("Oylar bo‘yicha cancel rate")
    ax.set_xlabel("Oy")
    ax.set_ylabel("Cancel rate")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

def save_confusion_matrix_plot(y_true, y_pred, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    matrix = confusion_matrix(y_true, y_pred)
    display = ConfusionMatrixDisplay(confusion_matrix=matrix)
    display.plot(ax=ax, colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

def save_roc_curve_comparison(curves: list[tuple[str, object, object]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for model_name, y_true, y_proba in curves:
        RocCurveDisplay.from_predictions(y_true, y_proba, ax=ax, name=model_name)
    ax.set_title("ROC Curve Comparison")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

def save_pr_curve_comparison(curves: list[tuple[str, object, object]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for model_name, y_true, y_proba in curves:
        PrecisionRecallDisplay.from_predictions(y_true, y_proba, ax=ax, name=model_name)
    ax.set_title("Precision-Recall Curve Comparison")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
