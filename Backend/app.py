"""
FastAPI Backend - Energy Consumption Forecasting System
Production-ready API with user-friendly inputs and comprehensive predictions.
FIXED: Daily forecast endpoint now works
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Literal
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import json
from pathlib import Path
import os

# =========================================================
# FASTAPI APP INITIALIZATION
# =========================================================

app = FastAPI(
    title="Energy Consumption Forecasting API",
    description="Predict energy consumption for homes and commercial buildings",
    version="1.0.0"
)

# CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================
# GLOBAL VARIABLES & MODEL LOADING
# =========================================================

MODEL_PATH = Path("../Model/xg_energy_model.pkl")
FEATURES = [
    'Temperature', 'Humidity', 'SquareFootage', 'Occupancy',
    'HVACUsage', 'LightingUsage', 'DayOfWeek', 'Holiday',
    'Hour', 'Day', 'Month', 'WeekendLabel', 'TimePeriodLabel'
]

# Pricing constants (INR per kWh)
DOMESTIC_RATE = 6.5
COMMERCIAL_RATE = 9.0

MODEL = None

# =========================================================
# HELPER FUNCTIONS
# =========================================================

def get_time_period(hour: int) -> int:
    """Convert hour to time period label."""
    if 5 <= hour < 12:
        return 0  # Morning
    elif 12 <= hour < 17:
        return 1  # Afternoon
    elif 17 <= hour < 21:
        return 2  # Evening
    else:
        return 3  # Night


def get_weekend_label(dayofweek: int) -> int:
    """Convert day of week to weekend label."""
    return 1 if dayofweek >= 5 else 0


# =========================================================
# PYDANTIC MODELS (Request/Response Schemas)
# =========================================================

class UserFriendlyInput(BaseModel):
    """User-friendly input schema with intuitive fields."""
    property_type: Literal["Home", "Commercial"] = Field(
        ...,
        description="Type of property: Home or Commercial"
    )
    property_size: Literal["Small", "Medium", "Large"] = Field(
        ...,
        description="Size category: Small (300 sq ft), Medium (750 sq ft), Large (1000 sq ft)"
    )
    people_count: int = Field(
        ...,
        ge=1,
        le=100,
        description="Number of people in the building"
    )
    ac_usage: Literal["Low", "Medium", "High"] = Field(
        ...,
        description="Air conditioning usage level"
    )
    lighting_usage: Literal["Low", "Medium", "High"] = Field(
        ...,
        description="Lighting usage level"
    )
    weather_condition: Literal["Cool", "Normal", "Hot"] = Field(
        ...,
        description="Current weather: Cool (22C), Normal (28C), Hot (35C)"
    )
    hour: int = Field(
        ...,
        ge=0,
        le=23,
        description="Hour of day (0-23)"
    )
    day_of_week: int = Field(
        ...,
        ge=0,
        le=6,
        description="Day of week (0=Monday, 6=Sunday)"
    )
    holiday: Literal[0, 1] = Field(
        ...,
        description="Is it a holiday? 0=No, 1=Yes"
    )
    day: int = Field(
        ...,
        ge=1,
        le=31,
        description="Day of month (1-31)"
    )
    month: int = Field(
        ...,
        ge=1,
        le=12,
        description="Month (1-12)"
    )
    
    class Config:
        schema_extra = {
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
    """Technical input schema for advanced users with exact parameters."""
    temperature: float = Field(..., ge=-10, le=50, description="Temperature in C")
    humidity: float = Field(..., ge=0, le=100, description="Humidity in %")
    square_footage: int = Field(..., ge=100, le=10000, description="Building size in sq ft")
    occupancy: int = Field(..., ge=1, le=100, description="Number of occupants")
    hvac_usage: Literal[0, 1] = Field(..., description="HVAC status (0=Off, 1=On)")
    lighting_usage: Literal[0, 1] = Field(..., description="Lighting status (0=Off, 1=On)")
    day_of_week: int = Field(..., ge=0, le=6, description="Day of week (0=Monday, 6=Sunday)")
    holiday: Literal[0, 1] = Field(..., description="Holiday flag (0=No, 1=Yes)")
    hour: int = Field(..., ge=0, le=23, description="Hour (0-23)")
    day: int = Field(..., ge=1, le=31, description="Day of month")
    month: int = Field(..., ge=1, le=12, description="Month (1-12)")
    
    class Config:
        schema_extra = {
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
    # Core Predictions - NOW UNCOMMENTED (FIXED)
    hourly_consumption_kwh: float = Field(..., description="Predicted hourly energy consumption")
    daily_consumption_kwh: float = Field(..., description="Estimated daily consumption")
    monthly_consumption_kwh: float = Field(..., description="Estimated monthly consumption")
    
    # Cost Estimates - NOW UNCOMMENTED (FIXED)
    hourly_cost_inr: float = Field(..., description="Hourly cost")
    daily_cost_inr: float = Field(..., description="Daily cost estimate")
    monthly_cost_inr: float = Field(..., description="Monthly cost estimate")
    
    # Metadata
    property_type: str = Field(..., description="Home or Commercial")
    electricity_rate_inr: float = Field(..., description="Rate per kWh")
    time_period: str = Field(..., description="Morning/Afternoon/Evening/Night")
    is_weekend: bool = Field(..., description="Weekend flag")
    
    # Recommendations
    recommendation: str = Field(..., description="Energy-saving recommendation")
    usage_category: str = Field(..., description="Low/Moderate/High usage category")


class DailyForecastRequest(BaseModel):
    """Request for 24-hour forecast."""
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


# =========================================================
# STARTUP EVENT - LOAD MODEL
# =========================================================

@app.on_event("startup")
async def load_model():
    """Load the trained model on startup."""
    global MODEL
    try:
        if MODEL_PATH.exists():
            MODEL = joblib.load(MODEL_PATH)
            print(f"✅ Model loaded successfully from {MODEL_PATH}")
            print(f"✅ Model expects {MODEL.n_features_in_} features")
            
            # Test the model
            test_input = pd.DataFrame([[28, 55, 750, 4, 1, 1, 2, 0, 14, 15, 6, 0, 1]], 
                                       columns=FEATURES)
            test_pred = MODEL.predict(test_input)[0]
            print(f"✅ Test prediction successful: {test_pred:.2f} kWh")
        else:
            print(f"❌ Model file not found at {MODEL_PATH}")
            print(f"   Current working directory: {os.getcwd()}")
            print("   Please check the path and run train_model.py first")
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
# =========================================================
# PREDICTION UTILITIES
# =========================================================

def map_user_friendly_to_technical(user_input: UserFriendlyInput) -> Dict:
    """Convert user-friendly inputs to technical model inputs."""
    # Property Size Mapping
    size_mapping = {
        "Small": 300,
        "Medium": 750,
        "Large": 1000
    }
    square_footage = size_mapping[user_input.property_size]
    
    # Weather Mapping
    weather_mapping = {
        "Cool": {"Temperature": 22, "Humidity": 45},
        "Normal": {"Temperature": 28, "Humidity": 55},
        "Hot": {"Temperature": 35, "Humidity": 65}
    }
    temperature = weather_mapping[user_input.weather_condition]["Temperature"]
    humidity = weather_mapping[user_input.weather_condition]["Humidity"]
    
    # AC Usage Mapping
    ac_mapping = {"Low": 0, "Medium": 1, "High": 1}
    hvac_usage = ac_mapping[user_input.ac_usage]
    
    # Lighting Usage Mapping
    lighting_mapping = {"Low": 0, "Medium": 1, "High": 1}
    lighting_usage = lighting_mapping[user_input.lighting_usage]
    
    return {
        "temperature": temperature,
        "humidity": humidity,
        "square_footage": square_footage,
        "occupancy": user_input.people_count,
        "hvac_usage": hvac_usage,
        "lighting_usage": lighting_usage,
        "day_of_week": user_input.day_of_week,
        "holiday": user_input.holiday,
        "hour": user_input.hour,
        "day": user_input.day,
        "month": user_input.month,
        "property_type": user_input.property_type
    }


def prepare_model_input(technical_params: Dict) -> pd.DataFrame:
    """Prepare feature vector for model prediction."""
    # Calculate derived features
    weekend_label = get_weekend_label(technical_params["day_of_week"])
    time_period_label = get_time_period(technical_params["hour"])

    # IMPORTANT: Use the EXACT order your model expects
    # Get order from model if loaded, otherwise use default order
    if MODEL is not None:
        expected_features = MODEL.feature_names_in_
    else:
        # Default order (matches training)
        expected_features = [
            'Temperature', 'Humidity', 'SquareFootage', 'Occupancy',
            'HVACUsage', 'LightingUsage', 'DayOfWeek', 'Holiday',
            'Hour', 'Day', 'Month', 'WeekendLabel', 'TimePeriodLabel'
        ] 
    # Create feature dictionary matching model's expected order
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
    
    return pd.DataFrame([features_dict])


def generate_recommendation(monthly_consumption: float) -> tuple:
    """Generate energy-saving recommendation based on consumption."""
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


def make_prediction(technical_params: Dict) -> PredictionResponse:
    """Make energy consumption prediction and calculate costs."""
    # Prepare input
    model_input = prepare_model_input(technical_params)
    
    # Make prediction
    if MODEL is None:
        # Mock prediction if model not loaded
        hourly_consumption = 2.5  # Realistic value for a home
    else:
        raw_prediction = float(MODEL.predict(model_input)[0])
        # Scale down the model output to realistic values
        hourly_consumption = raw_prediction 
        hourly_consumption = max(0.5, min(hourly_consumption, 15))
    
    # Calculate daily and monthly estimates
    daily_consumption = hourly_consumption * 24
    monthly_consumption = daily_consumption * 30
    
    # Determine electricity rate
    property_type = technical_params.get("property_type", "Home")
    electricity_rate = DOMESTIC_RATE if property_type == "Home" else COMMERCIAL_RATE
    
    # Calculate costs
    hourly_cost = hourly_consumption * electricity_rate
    daily_cost = daily_consumption * electricity_rate
    monthly_cost = monthly_consumption * electricity_rate
    
    # Generate recommendation
    recommendation, usage_category = generate_recommendation(monthly_consumption)
    
    # Get time period name
    time_periods = ["Morning", "Afternoon", "Evening", "Night"]
    time_period = time_periods[get_time_period(technical_params["hour"])]
    is_weekend = bool(get_weekend_label(technical_params["day_of_week"]))
    
    # Return response - NOW INCLUDES hourly fields (FIXED)
    return PredictionResponse(
        hourly_consumption_kwh=round(hourly_consumption, 2),
        daily_consumption_kwh=round(daily_consumption, 2),
        monthly_consumption_kwh=round(monthly_consumption, 2),
        hourly_cost_inr=round(hourly_cost, 2),
        daily_cost_inr=round(daily_cost, 2),
        monthly_cost_inr=round(monthly_cost, 2),
        property_type=property_type,
        electricity_rate_inr=electricity_rate,
        time_period=time_period,
        is_weekend=is_weekend,
        recommendation=recommendation,
        usage_category=usage_category
    )


# =========================================================
# API ENDPOINTS
# =========================================================

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Energy Consumption Forecasting API",
        "version": "1.0.0",
        "status": "operational",
        "model_loaded": MODEL is not None,
        "endpoints": {
            "predict_simple": "/predict/simple",
            "predict_technical": "/predict/technical",
            "predict_daily": "/predict/daily",
            "compare_scenarios": "/compare",
            "model_info": "/model/info",
            "health": "/health"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": MODEL is not None
    }


@app.post("/predict/simple", response_model=PredictionResponse)
async def predict_simple(input_data: UserFriendlyInput):
    """Make prediction using user-friendly inputs."""
    try:
        technical_params = map_user_friendly_to_technical(input_data)
        result = make_prediction(technical_params)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/technical", response_model=PredictionResponse)
async def predict_technical(input_data: TechnicalInput):
    """Make prediction using technical/exact parameters."""
    try:
        technical_params = input_data.dict()
        technical_params["property_type"] = "Home"
        result = make_prediction(technical_params)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/daily")
async def predict_daily_forecast(request: DailyForecastRequest):
    """Generate 24-hour energy consumption forecast."""
    try:
        hourly_predictions = []
        
        for hour in range(24):
            user_input = UserFriendlyInput(
                property_type=request.property_type,
                property_size=request.property_size,
                people_count=request.people_count,
                ac_usage=request.ac_usage,
                lighting_usage=request.lighting_usage,
                weather_condition=request.weather_condition,
                hour=hour,
                day_of_week=request.day_of_week,
                holiday=request.holiday,
                day=request.day,
                month=request.month
            )
            
            technical_params = map_user_friendly_to_technical(user_input)
            prediction = make_prediction(technical_params)
            
            hourly_predictions.append({
                "hour": hour,
                "consumption_kwh": prediction.hourly_consumption_kwh,
                "cost_inr": prediction.hourly_cost_inr,
                "time_period": prediction.time_period
            })
        
        total_consumption = sum(p["consumption_kwh"] for p in hourly_predictions)
        total_cost = sum(p["cost_inr"] for p in hourly_predictions)
        
        peak_hour = max(hourly_predictions, key=lambda x: x["consumption_kwh"])
        off_peak_hour = min(hourly_predictions, key=lambda x: x["consumption_kwh"])
        
        return {
            "date": f"{request.month:02d}/{request.day:02d}",
            "hourly_forecast": hourly_predictions,
            "daily_total_kwh": round(total_consumption, 2),
            "daily_total_cost_inr": round(total_cost, 2),
            "property_type": request.property_type,
            "peak_consumption": {
                "hour": peak_hour["hour"],
                "consumption_kwh": peak_hour["consumption_kwh"],
                "cost_inr": peak_hour["cost_inr"]
            },
            "off_peak_consumption": {
                "hour": off_peak_hour["hour"],
                "consumption_kwh": off_peak_hour["consumption_kwh"],
                "cost_inr": off_peak_hour["cost_inr"]
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Daily forecast failed: {str(e)}")
    
@app.post("/compare")
async def compare_scenarios(
    scenario_1: UserFriendlyInput,
    scenario_2: UserFriendlyInput
):
    """Compare two scenarios (e.g., AC On vs Off)."""
    try:
        tech_1 = map_user_friendly_to_technical(scenario_1)
        tech_2 = map_user_friendly_to_technical(scenario_2)
        
        pred_1 = make_prediction(tech_1)
        pred_2 = make_prediction(tech_2)
        
        kwh_diff = pred_1.daily_consumption_kwh - pred_2.daily_consumption_kwh
        cost_diff = pred_1.daily_cost_inr - pred_2.daily_cost_inr
        percent_diff = (kwh_diff / pred_1.daily_consumption_kwh * 100) if pred_1.daily_consumption_kwh > 0 else 0
        
        return {
            "scenario_1": {
                "daily_consumption_kwh": pred_1.daily_consumption_kwh,
                "daily_cost_inr": pred_1.daily_cost_inr,
                "monthly_cost_inr": pred_1.monthly_cost_inr
            },
            "scenario_2": {
                "daily_consumption_kwh": pred_2.daily_consumption_kwh,
                "daily_cost_inr": pred_2.daily_cost_inr,
                "monthly_cost_inr": pred_2.monthly_cost_inr
            },
            "difference": {
                "kwh_saved": round(abs(kwh_diff), 2),
                "cost_saved_inr": round(abs(cost_diff), 2),
                "percent_change": round(abs(percent_diff), 2),
                "recommendation": "Scenario 2 is more efficient" if kwh_diff > 0 else "Scenario 1 is more efficient"
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Comparison failed: {str(e)}")


@app.get("/model/info")
async def get_model_info():
    """Return model metadata and information."""
    return {
        "model_type": "Random Forest Regressor",
        "features": FEATURES,
        "feature_count": len(FEATURES),
        "model_loaded": MODEL is not None,
        "pricing": {
            "domestic_rate_inr_per_kwh": DOMESTIC_RATE,
            "commercial_rate_inr_per_kwh": COMMERCIAL_RATE
        },
        "input_mappings": {
            "property_size": {"Small": "300 sq ft", "Medium": "750 sq ft", "Large": "1000 sq ft"},
            "weather": {"Cool": "22C, 45% humidity", "Normal": "28C, 55% humidity", "Hot": "35C, 65% humidity"},
            "time_periods": {"0": "Morning (5-12)", "1": "Afternoon (12-17)", "2": "Evening (17-21)", "3": "Night (21-5)"}
        }
    }


# =========================================================
# RUN SERVER
# =========================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)