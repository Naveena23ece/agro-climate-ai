import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# Load dataset
data = pd.read_csv("farm_sensor_data.csv")

# Feature Engineering (Lag Features)
data["soil_moisture_lag1"] = data["soil_moisture"].shift(1)
data["soil_moisture_lag2"] = data["soil_moisture"].shift(2)

data.dropna(inplace=True)

# Features and Target
X = data[["temperature", "humidity", "rainfall",
          "soil_moisture_lag1", "soil_moisture_lag2"]]

y = data["soil_moisture"]

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Model Training Completed")
print("MAE:", mae)
print("R2 Score:", r2)

# Save model
joblib.dump(model, "models/random_forest_model.pkl")

print("Model Saved Successfully!")