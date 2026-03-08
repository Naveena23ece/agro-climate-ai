from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import joblib
import numpy as np
import firebase_admin
from firebase_admin import credentials, db
from datetime import datetime
import os
import json

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- Firebase Setup ---------------- #

if not firebase_admin._apps:
    firebase_key = json.loads(os.environ["FIREBASE_KEY"])
    cred = credentials.Certificate(firebase_key)

    firebase_admin.initialize_app(cred, {
        "databaseURL": os.environ["FIREBASE_DB_URL"]
    })

# ---------------- Load Models ---------------- #

rf_model = joblib.load("models/random_forest_model.pkl")
anomaly_model = joblib.load("models/isolation_forest_model.pkl")

# ---------------- PREDICT ENDPOINT ---------------- #

@app.get("/predict")
def predict():

    data = pd.read_csv("./farm_sensor_data.csv")
    latest = data.tail(2)

    temperature = latest.iloc[-1]["temperature"]
    humidity = latest.iloc[-1]["humidity"]
    rainfall = latest.iloc[-1]["rainfall"]
    soil_moisture_lag1 = latest.iloc[-1]["soil_moisture"]
    soil_moisture_lag2 = latest.iloc[-2]["soil_moisture"]

    # Random Forest input
    X_input = pd.DataFrame([{
        "temperature": temperature,
        "humidity": humidity,
        "rainfall": rainfall,
        "soil_moisture_lag1": soil_moisture_lag1,
        "soil_moisture_lag2": soil_moisture_lag2
    }])

    tree_predictions = np.array(
        [tree.predict(X_input)[0] for tree in rf_model.estimators_]
    )

    predicted_soil_moisture = np.mean(tree_predictions)
    prediction_variance = np.var(tree_predictions)
    confidence_score = round(1 / (1 + prediction_variance), 3)

    # Anomaly detection
    anomaly_input = pd.DataFrame([{
        "temperature": temperature,
        "humidity": humidity,
        "soil_moisture": soil_moisture_lag1,
        "rainfall": rainfall
    }])

    anomaly_result = anomaly_model.predict(anomaly_input)[0]

    if anomaly_result == -1:
        recommendation = "Abnormal climate detected. Monitor field."
    elif predicted_soil_moisture > 60:
        recommendation = "Skip irrigation."
    elif rainfall == 1:
        recommendation = "Delay fertilizer application."
    elif predicted_soil_moisture < 30:
        recommendation = "Irrigation recommended."
    else:
        recommendation = "Conditions normal."

    result_data = {
        "timestamp": datetime.now().isoformat(),
        "temperature": round(float(temperature), 2),
        "humidity": round(float(humidity), 2),
        "predicted_soil_moisture": round(float(predicted_soil_moisture), 2),
        "confidence_score": confidence_score,
        "anomaly": "Yes" if anomaly_result == -1 else "No",
        "recommendation": recommendation
    }

    # Store in Firebase
    ref = db.reference("predictions")
    ref.push(result_data)

    return result_data


# ---------------- TREND ENDPOINT ---------------- #

@app.get("/trend")
def trend():

    data = pd.read_csv("./farm_sensor_data.csv")
    last_24 = data.tail(24)

    return {
        "temperature": last_24["temperature"].round(2).tolist(),
        "humidity": last_24["humidity"].round(2).tolist(),
        "soil_moisture": last_24["soil_moisture"].round(2).tolist()
    }


# ---------------- HISTORY ENDPOINT ---------------- #

@app.get("/history")
def history():

    ref = db.reference("predictions")
    data = ref.get()

    if not data:
        return {"history": []}

    records = list(data.values())
    records.sort(key=lambda x: x["timestamp"])

    return {"history": records[-20:]}