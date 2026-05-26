# info.py
from fastapi import APIRouter
from datetime import datetime
from model_loader import get_model
from config import FEATURES, DOMESTIC_RATE, COMMERCIAL_RATE

router = APIRouter(tags=["Info"])

@router.get("/")
async def root():
    """API information."""
    return {
        "message": "Energy Consumption Forecasting API",
        "version": "1.0.0",
        "status": "operational",
        "model_loaded": get_model() is not None,
        "endpoints": {
            "predict_simple": "/predict/simple",
            "predict_technical": "/predict/technical",
            "predict_daily": "/predict/daily",
            "compare_scenarios": "/compare",
            "model_info": "/model/info",
            "health": "/health"
        }
    }

@router.get("/health")
async def health_check():
    """Health check."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": get_model() is not None
    }

@router.get("/model/info")
async def get_model_info():
    """Model metadata."""
    return {
        "model_type": "Random Forest Regressor",
        "features": FEATURES,
        "feature_count": len(FEATURES),
        "model_loaded": get_model() is not None,
        "pricing": {
            "domestic_rate_inr_per_kwh": DOMESTIC_RATE,
            "commercial_rate_inr_per_kwh": COMMERCIAL_RATE
        },
        "input_mappings": {
            "property_size": {"Small": "300 sq ft", "Medium": "750 sq ft", "Large": "1000 sq ft"},
            "weather": {"Cool": "22°C, 45% humidity", "Normal": "28°C, 55% humidity", "Hot": "35°C, 65% humidity"},
            "time_periods": {"0": "Morning (5-12)", "1": "Afternoon (12-17)", "2": "Evening (17-21)", "3": "Night (21-5)"}
        }
    }