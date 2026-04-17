from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import joblib
import pandas as pd

from src.hotel_booking_ml.config import IMPROVEMENT_MODEL_PATH, MODELS_REPORT_DIR, TARGET_COLUMN, TRAIN_PATH, VAL_PATH
from src.hotel_booking_ml.evaluation.metrics import calculate_binary_metrics
from src.hotel_booking_ml.models.model_factory import DEFAULT_XGB_PARAMS, get_improvement_model
from src.hotel_booking_ml.preprocessing.pipeline import make_modeling_pipeline
from src.hotel_booking_ml.utils.io import ensure_project_directories, save_json
from src.hotel_booking_ml.utils.logger import setup_logger

logger = setup_logger()

def main() -> None:
    ensure_project_directories()

    train_df = pd.read_csv(TRAIN_PATH)
    val_df = pd.read_csv(VAL_PATH)

    X_train = train_df.drop(columns=[TARGET_COLUMN])
    y_train = train_df[TARGET_COLUMN]
    X_val = val_df.drop(columns=[TARGET_COLUMN])
    y_val = val_df[TARGET_COLUMN]

    pipeline = make_modeling_pipeline(get_improvement_model())
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_val)
    y_proba = pipeline.predict_proba(X_val)[:, 1]
    metrics = calculate_binary_metrics(y_val, y_pred, y_proba)

    IMPROVEMENT_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, IMPROVEMENT_MODEL_PATH)

    save_json(metrics, MODELS_REPORT_DIR / "improvement_metrics.json")
    save_json(DEFAULT_XGB_PARAMS, MODELS_REPORT_DIR / "improvement_params.json")

    logger.info("Improvement model metrics: %s", metrics)
    logger.info("Improvement model saqlandi: %s", IMPROVEMENT_MODEL_PATH)

if __name__ == "__main__":
    main()
