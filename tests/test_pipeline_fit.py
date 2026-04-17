from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.hotel_booking_ml.models.model_factory import get_baseline_model
from src.hotel_booking_ml.preprocessing.pipeline import make_modeling_pipeline

def test_baseline_pipeline_fit_and_predict() -> None:
    df = pd.DataFrame(
        [
            {
                "hotel": "City Hotel - Seoul",
                "lead_time": 10,
                "arrival_date_month": "July",
                "arrival_date_week_number": 29,
                "arrival_date_day_of_month": 18,
                "stays_in_weekend_nights": 2,
                "stays_in_week_nights": 3,
                "adults": 2,
                "children": 1,
                "babies": 0,
                "meal": "BB",
                "country": "KOR",
                "market_segment": "Online TA",
                "distribution_channel": "TA/TO",
                "is_repeated_guest": 0,
                "previous_cancellations": 0,
                "previous_bookings_not_canceled": 1,
                "reserved_room_type": "A",
                "assigned_room_type": "A",
                "booking_changes": 0,
                "deposit_type": "No Deposit",
                "agent": 9,
                "company": None,
                "days_in_waiting_list": 0,
                "customer_type": "Transient",
                "adr": 100.0,
                "required_car_parking_spaces": 0,
                "total_of_special_requests": 0,
                "city": "Seoul",
                "is_canceled": 0,
            },
            {
                "hotel": "Resort Hotel - Busan",
                "lead_time": 150,
                "arrival_date_month": "August",
                "arrival_date_week_number": 31,
                "arrival_date_day_of_month": 7,
                "stays_in_weekend_nights": 1,
                "stays_in_week_nights": 5,
                "adults": 2,
                "children": 0,
                "babies": 0,
                "meal": "HB",
                "country": "KOR",
                "market_segment": "Online TA",
                "distribution_channel": "TA/TO",
                "is_repeated_guest": 0,
                "previous_cancellations": 1,
                "previous_bookings_not_canceled": 0,
                "reserved_room_type": "A",
                "assigned_room_type": "B",
                "booking_changes": 1,
                "deposit_type": "Non Refund",
                "agent": 12,
                "company": None,
                "days_in_waiting_list": 4,
                "customer_type": "Transient",
                "adr": 210.0,
                "required_car_parking_spaces": 0,
                "total_of_special_requests": 0,
                "city": "Busan",
                "is_canceled": 1,
            },
            {
                "hotel": "City Hotel - Seoul",
                "lead_time": 20,
                "arrival_date_month": "March",
                "arrival_date_week_number": 12,
                "arrival_date_day_of_month": 12,
                "stays_in_weekend_nights": 0,
                "stays_in_week_nights": 2,
                "adults": 1,
                "children": 0,
                "babies": 0,
                "meal": "BB",
                "country": "JPN",
                "market_segment": "Direct",
                "distribution_channel": "Direct",
                "is_repeated_guest": 1,
                "previous_cancellations": 0,
                "previous_bookings_not_canceled": 2,
                "reserved_room_type": "A",
                "assigned_room_type": "A",
                "booking_changes": 0,
                "deposit_type": "No Deposit",
                "agent": 1,
                "company": None,
                "days_in_waiting_list": 0,
                "customer_type": "Contract",
                "adr": 80.0,
                "required_car_parking_spaces": 1,
                "total_of_special_requests": 2,
                "city": "Seoul",
                "is_canceled": 0,
            },
            {
                "hotel": "Resort Hotel - Busan",
                "lead_time": 200,
                "arrival_date_month": "December",
                "arrival_date_week_number": 50,
                "arrival_date_day_of_month": 21,
                "stays_in_weekend_nights": 2,
                "stays_in_week_nights": 7,
                "adults": 2,
                "children": 2,
                "babies": 0,
                "meal": "FB",
                "country": "USA",
                "market_segment": "Groups",
                "distribution_channel": "TA/TO",
                "is_repeated_guest": 0,
                "previous_cancellations": 2,
                "previous_bookings_not_canceled": 0,
                "reserved_room_type": "D",
                "assigned_room_type": "D",
                "booking_changes": 2,
                "deposit_type": "Non Refund",
                "agent": 15,
                "company": None,
                "days_in_waiting_list": 12,
                "customer_type": "Transient-Party",
                "adr": 250.0,
                "required_car_parking_spaces": 0,
                "total_of_special_requests": 0,
                "city": "Busan",
                "is_canceled": 1,
            },
        ]
    )

    X = df.drop(columns=["is_canceled"])
    y = df["is_canceled"]

    pipeline = make_modeling_pipeline(get_baseline_model())
    pipeline.fit(X, y)

    probabilities = pipeline.predict_proba(X)[:, 1]
    predictions = pipeline.predict(X)

    assert len(probabilities) == len(df)
    assert len(predictions) == len(df)
