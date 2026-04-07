import pandas as pd
import numpy as np

np.random.seed(42)

# Generate 3000 timestamps (hourly)
timestamps = pd.date_range(start="2024-01-01", periods=3000, freq="H")

temperature = np.random.normal(loc=30, scale=5, size=3000)
humidity = np.random.normal(loc=65, scale=10, size=3000)
soil_moisture = np.random.normal(loc=45, scale=15, size=3000)
rainfall = np.random.choice([0, 0, 0, 1], size=3000)

data = pd.DataFrame({
    "timestamp": timestamps,
    "temperature": temperature,
    "humidity": humidity,
    "soil_moisture": soil_moisture,
    "rainfall": rainfall
})

data.to_csv("farm_sensor_data.csv", index=False)

print("Dataset Generated Successfully!")