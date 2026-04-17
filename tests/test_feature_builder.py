from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.hotel_booking_ml.features.feature_builder import FeatureBuilder

def test_feature_builder_creates_expected_columns() -> None:
    df = pd.DataFrame(
        {
            "hotel": ["City Hotel - Seoul"],
            "lead_time": [10],
            "arrival_date_month": ["July"],
            "arrival_date_week_number": [29],
            "arrival_date_day_of_month": [18],
            "stays_in_weekend_nights": [2],
            "stays_in_week_nights": [3],
            "adults": [2],
            "children": [1],
            "babies": [0],
            "meal": ["BB"],
            "country": ["KOR"],
            "market_segment": ["Online TA"],
            "distribution_channel": ["TA/TO"],
            "is_repeated_guest": [0],
            "previous_cancellations": [0],
            "previous_bookings_not_canceled": [1],
            "reserved_room_type": ["A"],
            "assigned_room_type": ["B"],
            "booking_changes": [1],
            "deposit_type": ["No Deposit"],
            "agent": [9],
            "company": [None],
            "days_in_waiting_list": [0],
            "customer_type": ["Transient"],
            "adr": [120.0],
            "required_car_parking_spaces": [0],
            "total_of_special_requests": [1],
            "city": ["Seoul"],
            "reservation_status": ["Check-Out"],
            "reservation_status_date": ["00:00.0"],
            "arrival_date_year": [2024],
        }
    )

    transformed = FeatureBuilder().fit_transform(df)

    assert "total_nights" in transformed.columns
    assert "total_guests" in transformed.columns
    assert "hotel_type" in transformed.columns
    assert "has_company" in transformed.columns
    assert "reservation_status" not in transformed.columns
    assert "reservation_status_date" not in transformed.columns
    assert "arrival_date_year" not in transformed.columns
    assert "hotel" not in transformed.columns
