from __future__ import annotations

import pandas as pd

from src.hotel_booking_ml.config import EXPECTED_TARGET_VALUES, RAW_DATA_PATH, TARGET_COLUMN
from src.hotel_booking_ml.utils.logger import setup_logger

logger = setup_logger()

def load_raw_data() -> pd.DataFrame:
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(f"Raw data topilmadi: {RAW_DATA_PATH}")
    df = pd.read_csv(RAW_DATA_PATH)
    logger.info("Raw dataset yuklandi. Shape=%s", df.shape)
    return df

def validate_dataset(df: pd.DataFrame) -> None:
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target ustun topilmadi: {TARGET_COLUMN}")

    target_values = set(pd.Series(df[TARGET_COLUMN]).dropna().unique().tolist())
    if not target_values.issubset(EXPECTED_TARGET_VALUES):
        raise ValueError(
            f"Target qiymatlari 0/1 bo‘lishi kerak. Topilgan qiymatlar: {sorted(target_values)}"
        )

    if df.empty:
        raise ValueError("Dataset bo‘sh.")

    logger.info("Dataset validatsiyasi muvaffaqiyatli tugadi.")
