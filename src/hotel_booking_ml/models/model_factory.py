from __future__ import annotations

from typing import Any

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

try:
    from xgboost import XGBClassifier
except ImportError as error:
    raise ImportError(
        "xgboost kutubxonasi topilmadi. requirements.txt ichidagi dependency larni o‘rnating."
    ) from error

DEFAULT_XGB_PARAMS: dict[str, Any] = {
    "n_estimators": 220,
    "max_depth": 6,
    "learning_rate": 0.07,
    "subsample": 0.90,
    "colsample_bytree": 0.80,
    "min_child_weight": 2,
    "gamma": 0.10,
    "reg_alpha": 0.01,
    "reg_lambda": 1.20,
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "random_state": 42,
    "n_jobs": 4,
    "tree_method": "hist",
}

def get_dummy_model() -> DummyClassifier:
    return DummyClassifier(strategy="most_frequent")

def get_baseline_model() -> LogisticRegression:
    return LogisticRegression(
        max_iter=500,
        class_weight="balanced",
        solver="saga",
    )

def get_improvement_model(custom_params: dict[str, Any] | None = None) -> XGBClassifier:
    params = DEFAULT_XGB_PARAMS.copy()
    if custom_params:
        params.update(custom_params)
    return XGBClassifier(**params)
