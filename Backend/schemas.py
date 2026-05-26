# schemas.py
from pydantic import BaseModel, Field
from typing import Literal

class UserFriendlyInput(BaseModel):
    """User‑friendly input schema."""
    property_type: Literal["Home", "Commercial"] = Field(..., description="Home or Commercial")
    property_size: Literal["Small", "Medium", "Large"] = Field(..., description="Size category")
    people_count: int = Field(..., ge=1, le=100, description="Number of people")
    ac_usage: Literal["Low", "Medium", "High"] = Field(..., description="Air conditioning usage")
    lighting_usage: Literal["Low", "Medium", "High"] = Field(..., description="Lighting usage")
    weather_condition: Literal["Cool", "Normal", "Hot"] = Field(..., description="Weather")
    hour: int = Field(..., ge=0, le=23, description="Hour (0–23)")
    day_of_week: int = Field(..., ge=0, le=6, description="0=Monday, 6=Sunday")
    holiday: Literal[0, 1] = Field(..., description="Holiday flag")
    day: int = Field(..., ge=1, le=31, description="Day of month")
    month: int = Field(..., ge=1, le=12, description="Month")

    class Config:
        json_schema_extra = {  # Changed from 'schema_extra'
            "example": {
                "property_type": "Home",
                "property_size": "Medium",
                "people_count": 4,
                "ac_usage": "Medium",
                "lighting_usage": "Low",
                "weather_condition": "Hot",
                "hour": 14,
                "day_of_week": 2,
                "holiday": 0,
                "day": 23,
                "month": 5
            }
        }

class TechnicalInput(BaseModel):
    """Technical input schema for advanced users."""
    temperature: float = Field(..., ge=-10, le=50, description="Temperature in °C")
    humidity: float = Field(..., ge=0, le=100, description="Humidity in %")
    square_footage: int = Field(..., ge=100, le=10000, description="Area in sq ft")
    occupancy: int = Field(..., ge=1, le=100, description="Number of occupants")
    hvac_usage: Literal[0, 1] = Field(..., description="HVAC on/off")
    lighting_usage: Literal[0, 1] = Field(..., description="Lighting on/off")
    day_of_week: int = Field(..., ge=0, le=6, description="0=Monday, 6=Sunday")
    holiday: Literal[0, 1] = Field(..., description="Holiday flag")
    hour: int = Field(..., ge=0, le=23, description="Hour")
    day: int = Field(..., ge=1, le=31, description="Day of month")
    month: int = Field(..., ge=1, le=12, description="Month")

    class Config:
        json_schema_extra = {  # Changed from 'schema_extra'
            "example": {
                "temperature": 28.5,
                "humidity": 65,
                "square_footage": 750,
                "occupancy": 4,
                "hvac_usage": 1,
                "lighting_usage": 1,
                "day_of_week": 2,
                "holiday": 0,
                "hour": 14,
                "day": 23,
                "month": 5
            }
        }

class PredictionResponse(BaseModel):
    """Comprehensive prediction response."""
    hourly_consumption_kwh: float
    daily_consumption_kwh: float
    monthly_consumption_kwh: float
    hourly_cost_inr: float
    daily_cost_inr: float
    monthly_cost_inr: float
    property_type: str
    electricity_rate_inr: float
    time_period: str
    is_weekend: bool
    recommendation: str
    usage_category: str

class DailyForecastRequest(BaseModel):
    """Request for 24‑hour forecast."""
    property_type: Literal["Home", "Commercial"]
    property_size: Literal["Small", "Medium", "Large"]
    people_count: int = Field(..., ge=1, le=100)
    ac_usage: Literal["Low", "Medium", "High"]
    lighting_usage: Literal["Low", "Medium", "High"]
    weather_condition: Literal["Cool", "Normal", "Hot"]
    day_of_week: int = Field(..., ge=0, le=6)
    holiday: Literal[0, 1]
    day: int = Field(..., ge=1, le=31)
    month: int = Field(..., ge=1, le=12)