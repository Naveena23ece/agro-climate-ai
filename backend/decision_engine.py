import pandas as pd
import joblib
import numpy as np

# Load models
rf_model = joblib.load("models/random_forest_model.pkl")
anomaly_model = joblib.load("models/isolation_forest_model.pkl")

# Load latest data
data = pd.read_csv("farm_sensor_data.csv")

# Get last 2 rows for lag features
latest = data.tail(2)

temperature = latest.iloc[-1]["temperature"]
humidity = latest.iloc[-1]["humidity"]
rainfall = latest.iloc[-1]["rainfall"]
soil_moisture_lag1 = latest.iloc[-1]["soil_moisture"]
soil_moisture_lag2 = latest.iloc[-2]["soil_moisture"]

# Prepare input for prediction
X_input = X_input = pd.DataFrame([{
    "temperature": temperature,
    "humidity": humidity,
    "rainfall": rainfall,
    "soil_moisture_lag1": soil_moisture_lag1,
    "soil_moisture_lag2": soil_moisture_lag2
}])

# Predict soil moisture
predicted_soil_moisture = rf_model.predict(X_input)[0]

# Check anomaly
anomaly_input = anomaly_input = pd.DataFrame([{
    "temperature": temperature,
    "humidity": humidity,
    "soil_moisture": soil_moisture_lag1,
    "rainfall": rainfall
}])

anomaly_result = anomaly_model.predict(anomaly_input)[0]

# ---------------- Decision Logic ---------------- #

recommendation = ""

if anomaly_result == -1:
    recommendation = "⚠️ Abnormal climate detected. Monitor field conditions."

elif predicted_soil_moisture > 60:
    recommendation = "🚫 Soil moisture high. Skip irrigation."

elif rainfall == 1:
    recommendation = "🌧 Rain detected. Delay fertilizer application."

elif predicted_soil_moisture < 30:
    recommendation = "💧 Low soil moisture predicted. Irrigation recommended."

else:
    recommendation = "✅ Conditions normal. Farm operations can proceed."

# Output Results
print("\n------ DECISION ENGINE OUTPUT ------")
print("Current Temperature:", round(temperature, 2))
print("Current Humidity:", round(humidity, 2))
print("Predicted Soil Moisture:", round(predicted_soil_moisture, 2))
print("Anomaly Status:", "Anomaly Detected" if anomaly_result == -1 else "Normal")
print("Recommendation:", recommendation)