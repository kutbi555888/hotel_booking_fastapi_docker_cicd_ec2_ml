from __future__ import annotations


from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import joblib
import pandas as pd

from src.hotel_booking_ml.config import (
    BASELINE_MODEL_PATH,
    DUMMY_MODEL_PATH,
    MODELS_REPORT_DIR,
    TARGET_COLUMN,
    TRAIN_PATH,
    VAL_PATH,
)
from src.hotel_booking_ml.evaluation.metrics import calculate_binary_metrics, metrics_to_frame
from src.hotel_booking_ml.models.model_factory import get_baseline_model, get_dummy_model
from src.hotel_booking_ml.preprocessing.pipeline import make_modeling_pipeline
from src.hotel_booking_ml.utils.io import ensure_project_directories, save_json
from src.hotel_booking_ml.utils.logger import setup_logger

logger = setup_logger()

def train_and_evaluate(model_name: str, model, X_train, y_train, X_val, y_val):
    pipeline = make_modeling_pipeline(model)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_val)

    if hasattr(pipeline, "predict_proba"):
        y_proba = pipeline.predict_proba(X_val)[:, 1]
    else:
        y_proba = y_pred

    metrics = calculate_binary_metrics(y_val, y_pred, y_proba)
    logger.info("%s metrics: %s", model_name, metrics)
    return pipeline, metrics

def main() -> None:
    ensure_project_directories()

    train_df = pd.read_csv(TRAIN_PATH)
    val_df = pd.read_csv(VAL_PATH)

    X_train = train_df.drop(columns=[TARGET_COLUMN])
    y_train = train_df[TARGET_COLUMN]

    X_val = val_df.drop(columns=[TARGET_COLUMN])
    y_val = val_df[TARGET_COLUMN]

    dummy_pipeline, dummy_metrics = train_and_evaluate(
        "dummy_baseline", get_dummy_model(), X_train, y_train, X_val, y_val
    )
    baseline_pipeline, baseline_metrics = train_and_evaluate(
        "baseline_logreg", get_baseline_model(), X_train, y_train, X_val, y_val
    )

    DUMMY_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(dummy_pipeline, DUMMY_MODEL_PATH)
    joblib.dump(baseline_pipeline, BASELINE_MODEL_PATH)

    metrics_dict = {
        "dummy_baseline": dummy_metrics,
        "baseline_logreg": baseline_metrics,
    }
    save_json(metrics_dict, MODELS_REPORT_DIR / "baseline_metrics.json")

    metrics_df = metrics_to_frame(metrics_dict).sort_values("roc_auc", ascending=False)
    metrics_df.to_csv(MODELS_REPORT_DIR / "baseline_metrics.csv", index=False)

    logger.info("Baseline train tugadi. Natijalar %s ga saqlandi.", MODELS_REPORT_DIR)

if __name__ == "__main__":
    main()
