import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib

# Load dataset
data = pd.read_csv("farm_sensor_data.csv")

# Select features for anomaly detection
features = data[["temperature", "humidity", "soil_moisture", "rainfall"]]

# Train Isolation Forest
model = IsolationForest(
    contamination=0.05,   # 5% anomalies
    random_state=42
)

model.fit(features)

# Predict anomalies
data["anomaly"] = model.predict(features)

# -1 = anomaly, 1 = normal
anomaly_count = len(data[data["anomaly"] == -1])

print("Anomaly Model Trained Successfully!")
print("Number of anomalies detected:", anomaly_count)

# Save model
joblib.dump(model, "models/isolation_forest_model.pkl")

print("Anomaly Model Saved Successfully!")