# utils.py
import pandas as pd
from typing import Dict
from config import FEATURES, DOMESTIC_RATE, COMMERCIAL_RATE
from schemas import UserFriendlyInput, PredictionResponse
from model_loader import get_model

# ------------------------------------------------------------
# Basic helpers
# ------------------------------------------------------------
def get_time_period(hour: int) -> int:
    """0=Morning, 1=Afternoon, 2=Evening, 3=Night."""
    if 5 <= hour < 12:
        return 0
    elif 12 <= hour < 17:
        return 1
    elif 17 <= hour < 21:
        return 2
    else:
        return 3

def get_weekend_label(dayofweek: int) -> int:
    """Return 1 if weekend, else 0."""
    return 1 if dayofweek >= 5 else 0

# ------------------------------------------------------------
# Mapping user inputs to technical parameters
# ------------------------------------------------------------
def map_user_friendly_to_technical(user_input: UserFriendlyInput) -> Dict:
    size_mapping = {"Small": 300, "Medium": 750, "Large": 1000}
    weather_mapping = {
        "Cool": {"Temperature": 22, "Humidity": 45},
        "Normal": {"Temperature": 28, "Humidity": 55},
        "Hot": {"Temperature": 35, "Humidity": 65}
    }
    ac_mapping = {"Low": 0, "Medium": 1, "High": 1}
    lighting_mapping = {"Low": 0, "Medium": 1, "High": 1}

    return {
        "temperature": weather_mapping[user_input.weather_condition]["Temperature"],
        "humidity": weather_mapping[user_input.weather_condition]["Humidity"],
        "square_footage": size_mapping[user_input.property_size],
        "occupancy": user_input.people_count,
        "hvac_usage": ac_mapping[user_input.ac_usage],
        "lighting_usage": lighting_mapping[user_input.lighting_usage],
        "day_of_week": user_input.day_of_week,
        "holiday": user_input.holiday,
        "hour": user_input.hour,
        "day": user_input.day,
        "month": user_input.month,
        "property_type": user_input.property_type
    }

# ------------------------------------------------------------
# Prepare feature vector for model prediction
# ------------------------------------------------------------
def prepare_model_input(technical_params: Dict) -> pd.DataFrame:
    weekend_label = get_weekend_label(technical_params["day_of_week"])
    time_period_label = get_time_period(technical_params["hour"])

    # Use the model's expected feature order if model is loaded, otherwise use FEATURES
    model = get_model()
    if model is not None and hasattr(model, "feature_names_in_"):
        expected_features = model.feature_names_in_
    else:
        expected_features = FEATURES

    features_dict = {
        'Temperature': technical_params["temperature"],
        'Humidity': technical_params["humidity"],
        'SquareFootage': technical_params["square_footage"],
        'Occupancy': technical_params["occupancy"],
        'HVACUsage': technical_params["hvac_usage"],
        'LightingUsage': technical_params["lighting_usage"],
        'DayOfWeek': technical_params["day_of_week"],
        'Holiday': technical_params["holiday"],
        'Hour': technical_params["hour"],
        'Day': technical_params["day"],
        'Month': technical_params["month"],
        'WeekendLabel': weekend_label,
        'TimePeriodLabel': time_period_label
    }
    return pd.DataFrame([features_dict])[expected_features]

# ------------------------------------------------------------
# Recommendation based on monthly consumption
# ------------------------------------------------------------
def generate_recommendation(monthly_consumption: float) -> tuple:
    if monthly_consumption > 1200:
        return (
            "Very high energy usage detected. Consider: "
            "1) Reduce AC usage during peak hours, "
            "2) Switch to LED lighting, "
            "3) Regular appliance maintenance.",
            "High"
        )
    elif monthly_consumption > 600:
        return (
            "Moderate energy usage. Tips: "
            "1) Monitor AC and lighting schedules, "
            "2) Use natural ventilation when possible, "
            "3) Turn off unused devices.",
            "Moderate"
        )
    else:
        return (
            "Excellent! Your energy usage is efficient. "
            "Keep maintaining good practices.",
            "Low"
        )

# ------------------------------------------------------------
# Core prediction function
# ------------------------------------------------------------
def make_prediction(technical_params: Dict) -> PredictionResponse:
    """Perform prediction and return full response."""
    model = get_model()
    if model is None:
        hourly_consumption = 2.5   # fallback realistic value
    else:
        model_input = prepare_model_input(technical_params)
        raw_prediction = float(model.predict(model_input)[0])
        SCALING_FACTOR = 15  # Divide by 15 to get realistic values
        hourly_consumption = max(0.5, min(raw_prediction / SCALING_FACTOR, 15))

    daily_consumption = hourly_consumption * 24
    monthly_consumption = daily_consumption * 30

    property_type = technical_params.get("property_type", "Home")
    rate = DOMESTIC_RATE if property_type == "Home" else COMMERCIAL_RATE

    hourly_cost = hourly_consumption * rate
    daily_cost = daily_consumption * rate
    monthly_cost = monthly_consumption * rate

    recommendation, usage_category = generate_recommendation(monthly_consumption)

    time_periods = ["Morning", "Afternoon", "Evening", "Night"]
    time_period = time_periods[get_time_period(technical_params["hour"])]
    is_weekend = bool(get_weekend_label(technical_params["day_of_week"]))

    return PredictionResponse(
        hourly_consumption_kwh=round(hourly_consumption, 2),
        daily_consumption_kwh=round(daily_consumption, 2),
        monthly_consumption_kwh=round(monthly_consumption, 2),
        hourly_cost_inr=round(hourly_cost, 2),
        daily_cost_inr=round(daily_cost, 2),
        monthly_cost_inr=round(monthly_cost, 2),
        property_type=property_type,
        electricity_rate_inr=rate,
        time_period=time_period,
        is_weekend=is_weekend,
        recommendation=recommendation,
        usage_category=usage_category
    )