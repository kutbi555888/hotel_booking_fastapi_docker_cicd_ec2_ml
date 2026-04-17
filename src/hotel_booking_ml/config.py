from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
ENGINEERED_DATA_DIR = DATA_DIR / "engineered"
FINAL_DATA_DIR = DATA_DIR / "final"

REPORTS_DIR = PROJECT_ROOT / "reports"
EDA_REPORT_DIR = REPORTS_DIR / "eda"
MODELS_REPORT_DIR = REPORTS_DIR / "models"
EVALUATION_REPORT_DIR = REPORTS_DIR / "evaluation"

MODELS_DIR = PROJECT_ROOT / "models"

RAW_DATA_PATH = RAW_DATA_DIR / "hotel_bookings_updated_2024.csv"
TRAIN_PATH = PROCESSED_DATA_DIR / "train.csv"
VAL_PATH = PROCESSED_DATA_DIR / "val.csv"
TEST_PATH = PROCESSED_DATA_DIR / "test.csv"

BASELINE_MODEL_PATH = MODELS_DIR / "baseline_logreg.joblib"
DUMMY_MODEL_PATH = MODELS_DIR / "baseline_dummy.joblib"
IMPROVEMENT_MODEL_PATH = MODELS_DIR / "improvement_xgboost.joblib"

FINAL_BASELINE_MODEL_PATH = MODELS_DIR / "final_baseline_logreg.joblib"
FINAL_IMPROVEMENT_MODEL_PATH = MODELS_DIR / "final_improvement_xgboost.joblib"
BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"

TARGET_COLUMN = "is_canceled"
RANDOM_STATE = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15

EXPECTED_TARGET_VALUES = {0, 1}

RAW_INPUT_COLUMNS = [
    "hotel",
    "lead_time",
    "arrival_date_month",
    "arrival_date_week_number",
    "arrival_date_day_of_month",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "adults",
    "children",
    "babies",
    "meal",
    "country",
    "market_segment",
    "distribution_channel",
    "is_repeated_guest",
    "previous_cancellations",
    "previous_bookings_not_canceled",
    "reserved_room_type",
    "assigned_room_type",
    "booking_changes",
    "deposit_type",
    "agent",
    "company",
    "days_in_waiting_list",
    "customer_type",
    "adr",
    "required_car_parking_spaces",
    "total_of_special_requests",
    "city",
]

MONTH_ORDER = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]
