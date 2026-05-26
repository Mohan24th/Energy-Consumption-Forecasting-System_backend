"""
Generate a new, high-quality synthetic dataset with 8000 rows
that has strong, realistic energy consumption patterns.
Saves to: Data/Energy_consumption_data.csv
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

np.random.seed(42)
N = 8000  # Increased for better accuracy

print("="*60)
print("GENERATING ENHANCED DATASET")
print("="*60)

# Timestamps: 2022-01-01 hourly
start = datetime(2022, 1, 1, 0, 0, 0)
timestamps = [start + timedelta(hours=i) for i in range(N)]

# Time features
hours      = np.array([t.hour       for t in timestamps])
days       = np.array([t.day        for t in timestamps])
months     = np.array([t.month      for t in timestamps])
dow        = np.array([t.weekday()  for t in timestamps])  # 0=Mon
day_names  = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
dow_labels = [day_names[d] for d in dow]

# Weather features with seasonal variation
seasonal_temp = 25 + 6 * np.sin(2 * np.pi * (months - 3) / 12)
daily_temp    = 3 * np.sin(2 * np.pi * (hours - 14) / 24)
temperature   = seasonal_temp + daily_temp + np.random.normal(0, 2, N)
temperature   = np.clip(temperature, 15, 42)

humidity      = 55 - 0.3 * (temperature - 25) + np.random.normal(0, 8, N)
humidity      = np.clip(humidity, 20, 90)

# Building features
square_footage = np.random.choice(
    [800, 1000, 1200, 1500, 1800, 2000, 2500, 3000],
    size=N,
    p=[0.10, 0.15, 0.20, 0.20, 0.15, 0.10, 0.05, 0.05]
)

# Occupancy patterns
is_workday = (dow < 5).astype(int)
hour_occ   = np.clip(np.sin(np.pi * (hours - 8) / 10), 0, 1)
base_occ   = 2 + 8 * hour_occ * is_workday + np.random.randint(0, 3, N)
occupancy  = np.clip(base_occ, 1, 20).astype(int)

# HVAC usage patterns
hvac_prob  = 0.3 + 0.5 * (temperature > 28).astype(float) + 0.2 * (temperature < 20)
hvac_prob += 0.2 * ((hours >= 8) & (hours <= 20)).astype(float)
hvac_usage = (np.random.rand(N) < np.clip(hvac_prob, 0, 0.95)).astype(int)

# Lighting usage patterns
light_prob  = 0.7 * ((hours >= 18) | (hours <= 6)).astype(float)
light_prob += 0.3 * is_workday * ((hours >= 8) & (hours <= 18)).astype(float)
lighting_usage = (np.random.rand(N) < np.clip(light_prob, 0.1, 0.95)).astype(int)

# Renewable energy
renewable = np.clip(
    20 * np.sin(np.pi * np.clip(hours - 6, 0, 12) / 12) + np.random.normal(0, 2, N),
    0, 30
)

# Holidays
holiday = (np.random.rand(N) < 0.10).astype(int)

# ENERGY CONSUMPTION (realistic formula)
base_load      = square_footage / 80
hvac_load      = hvac_usage * (0.4 + 0.04 * np.abs(temperature - 22))
occ_load       = occupancy * 1.2
lighting_load  = lighting_usage * 3.5
renewable_off  = renewable * 0.15
peak_factor    = 1 + 0.3 * ((hours >= 17) & (hours <= 21)).astype(float)
noise          = np.random.normal(0, 3, N)

energy = (base_load + hvac_load + occ_load + lighting_load - renewable_off) * peak_factor + noise
energy = np.clip(energy, 20, 200).round(4)

# Create DataFrame
df = pd.DataFrame({
    'Timestamp'       : [t.strftime('%Y-%m-%d %H:%M:%S') for t in timestamps],
    'Temperature'     : temperature.round(4),
    'Humidity'        : humidity.round(4),
    'SquareFootage'   : square_footage.astype(float),
    'Occupancy'       : occupancy,
    'HVACUsage'       : ['On' if h else 'Off' for h in hvac_usage],
    'LightingUsage'   : ['On' if l else 'Off' for l in lighting_usage],
    'RenewableEnergy' : renewable.round(4),
    'DayOfWeek'       : dow_labels,
    'Holiday'         : ['Yes' if h else 'No' for h in holiday],
    'EnergyConsumption': energy,
})

# Save to Data folder
out_path = "Data/Energy_consumption_data.csv"
os.makedirs("Data", exist_ok=True)
df.to_csv(out_path, index=False)

print(f" Dataset saved to: {out_path}")
print(f"   Rows: {len(df):,}")
print(f"   Energy range: {energy.min():.1f} - {energy.max():.1f} kWh")
print(f"   Energy mean: {energy.mean():.1f} kWh")
print(f"   Energy std: {energy.std():.1f} kWh")
print(f"   Columns: {', '.join(df.columns)}")
print(f"   First 3 rows:")
print(df.head(3))