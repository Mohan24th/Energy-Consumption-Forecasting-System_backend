# app/config.py
from pathlib import Path
import os
# Get the absolute path to the project root (one level up from Backend)
PROJECT_ROOT = Path(__file__).parent.parent
MODEL_PATH = PROJECT_ROOT / "Model" / "xg_energy_model.pkl"

# Alternative: Use environment variable (for Render deployment)
# MODEL_PATH = Path(os.getenv("MODEL_PATH", str(PROJECT_ROOT / "Model" / "xg_energy_model.pkl")))

# Feature list (order must match training)
FEATURES = [
    'Temperature', 'Humidity', 'SquareFootage', 'Occupancy',
    'HVACUsage', 'LightingUsage', 'DayOfWeek', 'Holiday',
    'Hour', 'Day', 'Month', 'WeekendLabel', 'TimePeriodLabel'
]

# Pricing constants (INR per kWh)
DOMESTIC_RATE = 6.5
COMMERCIAL_RATE = 9.0