# Energy Consumption Forecasting System

A machine learning-based energy consumption forecasting system designed for Indian homes and commercial buildings. This system predicts hourly, daily, and monthly energy consumption with accompanying cost estimates based on building characteristics, usage patterns, and environmental conditions.

## Overview

The Energy Consumption Forecasting System utilizes a Random Forest machine learning model to provide accurate energy consumption predictions across multiple time horizons. The system includes a FastAPI backend and a React-based frontend, enabling users to forecast energy usage and estimate associated costs.

### Important Notice

**This system uses synthetic data for demonstration and educational purposes.** While the model architecture and API are designed for the Indian context with appropriate electricity tariffs, the training data does not represent actual Indian energy consumption patterns. For production deployment, the model should be retrained using real-world energy consumption data.

## Project Structure

```
Energy Forecasting System/
├── Backend/
│   └── main.py                          # FastAPI backend server
├── Data/
│   └── Energy_consumption_data.csv      # Synthetic training dataset
├── Frontend/
│   ├── public/
│   │   └── index.html
│   └── src/
│       ├── components/
│       ├── pages/
│       ├── services/
│       ├── utils/
│       ├── App.jsx
│       └── index.js
└── Model/
    └── energy_model.pkl                 # Trained Random Forest model
```

## Prerequisites

- Python 3.8 or higher
- Node.js 16 or higher
- npm or yarn

## Installation and Setup

### Backend Configuration

1. **Navigate to the Backend directory**

   ```bash
   cd Backend
   ```

2. **Create a virtual environment**

   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment**

   **Windows:**
   ```bash
   venv\Scripts\activate
   ```

   **macOS/Linux:**
   ```bash
   source venv/bin/activate
   ```

4. **Install dependencies**

   ```bash
   pip install fastapi uvicorn pandas numpy scikit-learn joblib
   ```

5. **Configure the model path**

   Open `main.py` and update the `MODEL_PATH` variable to point to your `energy_model.pkl` file:

   ```python
   MODEL_PATH = Path(r"path/to/your/Model/energy_model.pkl")
   ```

6. **Start the backend server**

   ```bash
   python main.py
   ```

   The backend will be available at `http://localhost:8000`

   API documentation is accessible at `http://localhost:8000/docs`

### Frontend Configuration

1. **Navigate to the Frontend directory**

   ```bash
   cd Frontend
   ```

2. **Install dependencies**

   ```bash
   npm install
   ```

3. **Start the development server**

   ```bash
   npm start
   ```

   The frontend will be available at `http://localhost:3000`

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API information |
| GET | `/health` | Health check |
| POST | `/predict/simple` | Prediction using user-friendly inputs |
| POST | `/predict/technical` | Prediction using technical parameters |
| POST | `/predict/daily` | Generate 24-hour hourly forecast |
| POST | `/compare` | Compare two scenarios |
| GET | `/model/info` | Retrieve model information |

## Request and Response Examples

### Simple Prediction Request

```json
{
  "property_type": "Home",
  "property_size": "Medium",
  "people_count": 4,
  "ac_usage": "Medium",
  "lighting_usage": "Low",
  "weather_condition": "Hot",
  "hour": 14,
  "day_of_week": 2,
  "holiday": 0,
  "day": 15,
  "month": 5
}
```

### Technical Prediction Request

```json
{
  "temperature": 28.5,
  "humidity": 65,
  "square_footage": 750,
  "occupancy": 4,
  "hvac_usage": 1,
  "lighting_usage": 1,
  "day_of_week": 2,
  "holiday": 0,
  "hour": 14,
  "day": 15,
  "month": 5
}
```

### Response Format

```json
{
  "hourly_consumption_kwh": 2.45,
  "daily_consumption_kwh": 58.80,
  "monthly_consumption_kwh": 1764.00,
  "hourly_cost_inr": 15.93,
  "daily_cost_inr": 382.20,
  "monthly_cost_inr": 11466.00,
  "property_type": "Home",
  "electricity_rate_inr": 6.5,
  "time_period": "Afternoon",
  "is_weekend": false,
  "recommendation": "Moderate usage. Monitor AC and lighting schedules.",
  "usage_category": "Moderate"
}
```

## Pricing Structure

| Property Type | Rate (per kWh) |
|---------------|----------------|
| Home (Domestic) | Rs. 6.5 |
| Commercial | Rs. 9.0 |

*Note: These rates are based on standard Indian tariff structures and are used for cost calculations only.*

## Model Specifications

### Features

The Random Forest model operates on the following input features:

- Temperature
- Humidity
- Square Footage
- Occupancy
- HVAC Usage
- Lighting Usage
- Day of Week
- Holiday Flag
- Hour of Day
- Day of Month
- Month
- Weekend Label
- Time Period Label

### Performance Metrics

| Metric | Value |
|--------|-------|
| Model Type | Random Forest Regressor |
| R² Score | 0.55 |
| RMSE | 5.43 kWh |
| MAE | 4.41 kWh |

*Note: These metrics are derived from the synthetic dataset. Real-world performance may vary significantly.*

## Frontend Routes

| Route | Page | Description |
|-------|------|-------------|
| `/` | Dashboard | Overview with quick prediction interface |
| `/predict` | Single Prediction | Detailed prediction form |
| `/daily-forecast` | Daily Forecast | 24-hour hourly breakdown |
| `/compare` | Scenario Comparison | Compare two usage scenarios |
| `/history` | History | View past predictions (localStorage) |
| `/about` | About | Model information and documentation |

## Environment Variables

Create a `.env` file in the `Backend` directory to configure the following optional variables:

```
DOMESTIC_RATE=6.5
COMMERCIAL_RATE=9.0
MODEL_PATH=/path/to/Model/energy_model.pkl
HOST=0.0.0.0
PORT=8000
DEBUG=True
```

## Troubleshooting

### Backend Issues

| Issue | Solution |
|-------|----------|
| Model not found error | Update `MODEL_PATH` in `main.py` to the correct path |
| Port already in use | Change the port number in `main.py` or kill the existing process |
| Module not found | Install missing dependencies: `pip install -r requirements.txt` |

### Frontend Issues

| Issue | Solution |
|-------|----------|
| react-scripts not found | Run `npm install` again |
| Module not found | Verify import paths match your project structure |
| API connection error | Ensure the backend is running on port 8000 |

## Technology Stack

### Backend
- **FastAPI** - Web framework for API development
- **scikit-learn** - Machine learning library
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **Joblib** - Model serialization and persistence

### Frontend
- **React** - JavaScript library for user interfaces
- **React Router** - Client-side navigation
- **Axios** - HTTP client for API requests
- **Recharts** - Data visualization library

## Limitations and Disclaimers

- Training data is synthetic and does not represent real Indian energy consumption patterns
- The model has not been validated against actual Indian household data
- Temperature range in training data (20-30°C) does not account for extreme Indian summer conditions (40°C+)
- Predictions are intended for educational and demonstration purposes
- Do not rely on these predictions for critical financial or energy planning decisions
- Seasonal variations (summer, monsoon, winter) are not fully represented in the current model

## Future Enhancements

- Retrain the model with real Indian energy consumption data
- Incorporate seasonal variations (summer, monsoon, winter)
- Add appliance-specific consumption patterns
- Implement solar generation forecasting
- Support time-of-day (ToD) tariff calculations
- Extend support for multiple Indian states with region-specific tariff structures
- Add predictive maintenance features for energy infrastructure



## Acknowledgments

- Dataset is synthetically generated for demonstration purposes
- Model architecture is designed for the Indian context
- Electricity rates are based on standard Indian tariff structures
