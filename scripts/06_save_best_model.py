from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import shutil

from src.hotel_booking_ml.config import BEST_MODEL_PATH, EVALUATION_REPORT_DIR
from src.hotel_booking_ml.utils.io import ensure_project_directories, load_json, save_json
from src.hotel_booking_ml.utils.logger import setup_logger

logger = setup_logger()

def main() -> None:
    ensure_project_directories()

    best_summary_path = EVALUATION_REPORT_DIR / "best_model_summary.json"
    best_summary = load_json(best_summary_path)

    source_path = best_summary["best_model_path"]
    shutil.copy2(source_path, BEST_MODEL_PATH)

    metadata = {
        "copied_from": source_path,
        "best_model_name": best_summary["best_model_name"],
        "best_metrics": best_summary["best_metrics"],
        "pipeline_stage": "fastapi_ec2_ready",
    }
    save_json(metadata, BEST_MODEL_PATH.with_suffix(".metadata.json"))

    logger.info("Best model nusxalandi: %s", BEST_MODEL_PATH)
    logger.info("Metadata: %s", metadata)

if __name__ == "__main__":
    main()
