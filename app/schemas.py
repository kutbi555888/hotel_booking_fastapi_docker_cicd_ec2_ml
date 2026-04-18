from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class BookingFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hotel: str
    lead_time: int = Field(ge=0)
    arrival_date_month: str
    arrival_date_week_number: int = Field(ge=1, le=53)
    arrival_date_day_of_month: int = Field(ge=1, le=31)
    stays_in_weekend_nights: int = Field(ge=0)
    stays_in_week_nights: int = Field(ge=0)
    adults: int = Field(ge=0)
    children: Optional[int] = Field(default=0, ge=0)
    babies: int = Field(default=0, ge=0)
    meal: str
    country: Optional[str] = None
    market_segment: str
    distribution_channel: str
    is_repeated_guest: int = Field(ge=0, le=1)
    previous_cancellations: int = Field(ge=0)
    previous_bookings_not_canceled: int = Field(ge=0)
    reserved_room_type: str
    assigned_room_type: str
    booking_changes: int = Field(ge=0)
    deposit_type: str
    agent: Optional[int] = None
    company: Optional[int] = None
    days_in_waiting_list: int = Field(ge=0)
    customer_type: str
    adr: float = Field(ge=0)
    required_car_parking_spaces: int = Field(ge=0)
    total_of_special_requests: int = Field(ge=0)
    city: str


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    booking: BookingFeatures
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)


class PredictionResult(BaseModel):
    prediction: int
    cancel_probability: float


class PredictionResponse(BaseModel):
    success: bool
    threshold: float
    result: PredictionResult
