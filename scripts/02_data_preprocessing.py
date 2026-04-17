from __future__ import annotations


from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import pandas as pd
from sklearn.model_selection import train_test_split

from src.hotel_booking_ml.config import (
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
    TARGET_COLUMN,
    TEST_SIZE,
    TRAIN_PATH,
    VAL_PATH,
    VAL_SIZE,
    TEST_PATH,
)
from src.hotel_booking_ml.data.dataset import load_raw_data, validate_dataset
from src.hotel_booking_ml.utils.io import ensure_project_directories, save_json
from src.hotel_booking_ml.utils.logger import setup_logger

logger = setup_logger()

def main() -> None:
    ensure_project_directories()
    df = load_raw_data()
    validate_dataset(df)

    train_val_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        stratify=df[TARGET_COLUMN],
        random_state=RANDOM_STATE,
    )

    val_relative_size = VAL_SIZE / (1 - TEST_SIZE)

    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_relative_size,
        stratify=train_val_df[TARGET_COLUMN],
        random_state=RANDOM_STATE,
    )

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(TRAIN_PATH, index=False)
    val_df.to_csv(VAL_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)

    split_summary = {
        "train_shape": list(train_df.shape),
        "val_shape": list(val_df.shape),
        "test_shape": list(test_df.shape),
        "train_target_mean": float(train_df[TARGET_COLUMN].mean()),
        "val_target_mean": float(val_df[TARGET_COLUMN].mean()),
        "test_target_mean": float(test_df[TARGET_COLUMN].mean()),
    }
    save_json(split_summary, PROCESSED_DATA_DIR / "split_summary.json")

    logger.info("Preprocessing tugadi.")
    logger.info("Train shape=%s | Val shape=%s | Test shape=%s", train_df.shape, val_df.shape, test_df.shape)

if __name__ == "__main__":
    main()
