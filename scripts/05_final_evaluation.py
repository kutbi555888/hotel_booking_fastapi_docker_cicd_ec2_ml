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
    EVALUATION_REPORT_DIR,
    FINAL_BASELINE_MODEL_PATH,
    FINAL_IMPROVEMENT_MODEL_PATH,
    IMPROVEMENT_MODEL_PATH,
    TARGET_COLUMN,
    TEST_PATH,
    TRAIN_PATH,
    VAL_PATH,
)
from src.hotel_booking_ml.evaluation.metrics import (
    calculate_binary_metrics,
    metrics_to_frame,
    save_classification_report,
)
from src.hotel_booking_ml.evaluation.plots import (
    save_confusion_matrix_plot,
    save_pr_curve_comparison,
    save_roc_curve_comparison,
)
from src.hotel_booking_ml.utils.io import ensure_project_directories, save_json
from src.hotel_booking_ml.utils.logger import setup_logger

logger = setup_logger()

def main() -> None:
    ensure_project_directories()

    train_df = pd.read_csv(TRAIN_PATH)
    val_df = pd.read_csv(VAL_PATH)
    test_df = pd.read_csv(TEST_PATH)

    train_val_df = pd.concat([train_df, val_df], axis=0, ignore_index=True)
    X_train_val = train_val_df.drop(columns=[TARGET_COLUMN])
    y_train_val = train_val_df[TARGET_COLUMN]

    X_test = test_df.drop(columns=[TARGET_COLUMN])
    y_test = test_df[TARGET_COLUMN]

    final_models = {
        "baseline_logreg": joblib.load(BASELINE_MODEL_PATH),
        "improvement_xgboost": joblib.load(IMPROVEMENT_MODEL_PATH),
    }

    for model_name, pipeline in final_models.items():
        logger.info("%s final refit boshlandi.", model_name)
        pipeline.fit(X_train_val, y_train_val)

    metrics_dict = {}
    curve_inputs = []

    for model_name, pipeline in final_models.items():
        logger.info("%s test evaluation boshlandi.", model_name)
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        metrics = calculate_binary_metrics(y_test, y_pred, y_proba)
        metrics_dict[model_name] = metrics
        curve_inputs.append((model_name, y_test, y_proba))

        save_classification_report(
            y_test,
            y_pred,
            EVALUATION_REPORT_DIR / f"classification_report_{model_name}.txt",
        )

    joblib.dump(final_models["baseline_logreg"], FINAL_BASELINE_MODEL_PATH)
    joblib.dump(final_models["improvement_xgboost"], FINAL_IMPROVEMENT_MODEL_PATH)

    metrics_df = metrics_to_frame(metrics_dict).sort_values("roc_auc", ascending=False)
    metrics_df.to_csv(EVALUATION_REPORT_DIR / "final_test_metrics.csv", index=False)

    best_model_name = metrics_df.iloc[0]["model_name"]
    best_model_path_map = {
        "baseline_logreg": FINAL_BASELINE_MODEL_PATH,
        "improvement_xgboost": FINAL_IMPROVEMENT_MODEL_PATH,
    }

    best_model = final_models[best_model_name]
    best_pred = best_model.predict(X_test)
    best_proba = best_model.predict_proba(X_test)[:, 1]

    save_confusion_matrix_plot(
        y_test,
        best_pred,
        EVALUATION_REPORT_DIR / f"confusion_matrix_{best_model_name}.png",
        f"Confusion Matrix - {best_model_name}",
    )
    save_roc_curve_comparison(curve_inputs, EVALUATION_REPORT_DIR / "roc_curve_comparison.png")
    save_pr_curve_comparison(curve_inputs, EVALUATION_REPORT_DIR / "pr_curve_comparison.png")

    best_summary = {
        "best_model_name": best_model_name,
        "best_model_path": str(best_model_path_map[best_model_name]),
        "best_metrics": metrics_dict[best_model_name],
        "compared_models": list(final_models.keys()),
    }
    save_json(best_summary, EVALUATION_REPORT_DIR / "best_model_summary.json")

    logger.info("Final evaluation tugadi.")
    logger.info("Best model: %s", best_summary)

if __name__ == "__main__":
    main()
