# predict.py
from fastapi import APIRouter, HTTPException, status
from schemas import UserFriendlyInput, TechnicalInput, DailyForecastRequest, PredictionResponse
from utils import map_user_friendly_to_technical, make_prediction

router = APIRouter(prefix="/predict", tags=["Prediction"])

@router.post("/simple", response_model=PredictionResponse)
async def predict_simple(input_data: UserFriendlyInput):
    """Make prediction using user-friendly inputs."""
    try:
        technical_params = map_user_friendly_to_technical(input_data)
        result = make_prediction(technical_params)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@router.post("/technical", response_model=PredictionResponse)
async def predict_technical(input_data: TechnicalInput):
    """Make prediction using technical/exact parameters."""
    try:
        technical_params = input_data.dict()
        technical_params["property_type"] = "Home"
        result = make_prediction(technical_params)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@router.post("/daily")
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
            pred = make_prediction(technical_params)
            hourly_predictions.append({
                "hour": hour,
                "consumption_kwh": pred.hourly_consumption_kwh,
                "cost_inr": pred.hourly_cost_inr,
                "time_period": pred.time_period
            })

        total_consumption = sum(p["consumption_kwh"] for p in hourly_predictions)
        total_cost = sum(p["cost_inr"] for p in hourly_predictions)
        peak = max(hourly_predictions, key=lambda x: x["consumption_kwh"])
        off_peak = min(hourly_predictions, key=lambda x: x["consumption_kwh"])

        return {
            "date": f"{request.month:02d}/{request.day:02d}",
            "hourly_forecast": hourly_predictions,
            "daily_total_kwh": round(total_consumption, 2),
            "daily_total_cost_inr": round(total_cost, 2),
            "property_type": request.property_type,
            "peak_consumption": {
                "hour": peak["hour"],
                "consumption_kwh": peak["consumption_kwh"],
                "cost_inr": peak["cost_inr"]
            },
            "off_peak_consumption": {
                "hour": off_peak["hour"],
                "consumption_kwh": off_peak["consumption_kwh"],
                "cost_inr": off_peak["cost_inr"]
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Daily forecast failed: {str(e)}")

@router.post("/compare")
async def compare_scenarios(scenario_1: UserFriendlyInput, scenario_2: UserFriendlyInput):
    """Compare two scenarios."""
    try:
        tech1 = map_user_friendly_to_technical(scenario_1)
        tech2 = map_user_friendly_to_technical(scenario_2)
        pred1 = make_prediction(tech1)
        pred2 = make_prediction(tech2)

        kwh_diff = pred1.daily_consumption_kwh - pred2.daily_consumption_kwh
        cost_diff = pred1.daily_cost_inr - pred2.daily_cost_inr
        percent_diff = (kwh_diff / pred1.daily_consumption_kwh * 100) if pred1.daily_consumption_kwh > 0 else 0

        return {
            "scenario_1": {
                "daily_consumption_kwh": pred1.daily_consumption_kwh,
                "daily_cost_inr": pred1.daily_cost_inr,
                "monthly_cost_inr": pred1.monthly_cost_inr
            },
            "scenario_2": {
                "daily_consumption_kwh": pred2.daily_consumption_kwh,
                "daily_cost_inr": pred2.daily_cost_inr,
                "monthly_cost_inr": pred2.monthly_cost_inr
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