# app/config.py
from pathlib import Path

# Model path (relative to the directory from which the server is started)
MODEL_PATH = Path("../Model/xg_energy_model.pkl")

# Feature list (order must match training)
FEATURES = [
    'Temperature', 'Humidity', 'SquareFootage', 'Occupancy',
    'HVACUsage', 'LightingUsage', 'DayOfWeek', 'Holiday',
    'Hour', 'Day', 'Month', 'WeekendLabel', 'TimePeriodLabel'
]

# Pricing constants (INR per kWh)
DOMESTIC_RATE = 6.5
COMMERCIAL_RATE = 9.0