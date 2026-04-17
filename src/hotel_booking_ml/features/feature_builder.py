from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.hotel_booking_ml.config import MONTH_ORDER, RAW_INPUT_COLUMNS

class FeatureBuilder(BaseEstimator, TransformerMixin):
    """Raw booking ma'lumotlaridan yangi feature lar yasaydi."""

    def __init__(self) -> None:
        self.month_map = {month: index for index, month in enumerate(MONTH_ORDER, start=1)}

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "FeatureBuilder":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        X = X.copy()

        for column in RAW_INPUT_COLUMNS:
            if column not in X.columns:
                X[column] = np.nan

        X["hotel"] = X["hotel"].astype("object")
        X["city"] = X["city"].astype("object")
        X["arrival_date_month"] = X["arrival_date_month"].astype("object")

        X["hotel_type"] = (
            X["hotel"]
            .fillna("Unknown Hotel")
            .astype(str)
            .str.split(" - ")
            .str[0]
            .replace("", "Unknown Hotel")
        )

        X["arrival_month_num"] = X["arrival_date_month"].map(self.month_map)
        X["arrival_is_peak_season"] = X["arrival_date_month"].isin(["June", "July", "August"]).astype(int)

        X["children"] = pd.to_numeric(X["children"], errors="coerce")
        X["babies"] = pd.to_numeric(X["babies"], errors="coerce")
        X["adults"] = pd.to_numeric(X["adults"], errors="coerce")
        X["lead_time"] = pd.to_numeric(X["lead_time"], errors="coerce")
        X["booking_changes"] = pd.to_numeric(X["booking_changes"], errors="coerce")
        X["days_in_waiting_list"] = pd.to_numeric(X["days_in_waiting_list"], errors="coerce")
        X["required_car_parking_spaces"] = pd.to_numeric(
            X["required_car_parking_spaces"], errors="coerce"
        )
        X["total_of_special_requests"] = pd.to_numeric(
            X["total_of_special_requests"], errors="coerce"
        )
        X["previous_cancellations"] = pd.to_numeric(
            X["previous_cancellations"], errors="coerce"
        )
        X["previous_bookings_not_canceled"] = pd.to_numeric(
            X["previous_bookings_not_canceled"], errors="coerce"
        )
        X["stays_in_weekend_nights"] = pd.to_numeric(
            X["stays_in_weekend_nights"], errors="coerce"
        )
        X["stays_in_week_nights"] = pd.to_numeric(
            X["stays_in_week_nights"], errors="coerce"
        )
        X["adr"] = pd.to_numeric(X["adr"], errors="coerce")

        X["total_nights"] = X["stays_in_weekend_nights"].fillna(0) + X["stays_in_week_nights"].fillna(0)
        X["total_guests"] = X["adults"].fillna(0) + X["children"].fillna(0) + X["babies"].fillna(0)
        X["is_family"] = ((X["children"].fillna(0) > 0) | (X["babies"].fillna(0) > 0)).astype(int)
        X["room_changed"] = (
            X["reserved_room_type"].fillna("Unknown").astype(str)
            != X["assigned_room_type"].fillna("Unknown").astype(str)
        ).astype(int)
        X["total_previous_bookings"] = (
            X["previous_cancellations"].fillna(0) + X["previous_bookings_not_canceled"].fillna(0)
        )
        X["special_request_ratio"] = X["total_of_special_requests"].fillna(0) / (X["total_nights"] + 1)
        X["booking_change_ratio"] = X["booking_changes"].fillna(0) / (X["lead_time"].fillna(0) + 1)
        X["waiting_ratio"] = X["days_in_waiting_list"].fillna(0) / (X["lead_time"].fillna(0) + 1)
        X["agent_missing"] = X["agent"].isna().astype(int)
        X["country_missing"] = X["country"].isna().astype(int)
        X["has_company"] = X["company"].notna().astype(int)
        X["has_parking_request"] = (X["required_car_parking_spaces"].fillna(0) > 0).astype(int)
        X["is_long_stay"] = (X["total_nights"] >= 5).astype(int)

        X["agent"] = X["agent"].fillna(-1).astype(float).astype(int).astype(str)

        drop_columns = [
            "reservation_status",
            "reservation_status_date",
            "arrival_date_year",
            "company",
            "hotel",
        ]
        X = X.drop(columns=[col for col in drop_columns if col in X.columns], errors="ignore")

        return X
