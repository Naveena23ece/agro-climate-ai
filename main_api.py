from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import joblib
import firebase_admin
from firebase_admin import credentials, db
from datetime import datetime
import requests
import os
import time
import json

from dotenv import load_dotenv

# -------------------- LOAD ENV --------------------
load_dotenv()

# 🔐 Secure API Key (NO hardcoding)
API_KEY = os.getenv("OPENWEATHER_API_KEY")

if not API_KEY:
    raise ValueError("OPENWEATHER_API_KEY not found. Check your .env file")

app = FastAPI(title="Agro Climate AI", version="2.0")

# ── CORS ──────────────────────────────────────────────────────────────────── #
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── FIREBASE ──────────────────────────────────────────────────────────────── #
if not firebase_admin._apps:
    try:
        if os.path.exists("firebase_key.json"):
            cred = credentials.Certificate("firebase_key.json")
            firebase_admin.initialize_app(cred, {
                "databaseURL": "https://agroclimateai-default-rtdb.asia-southeast1.firebasedatabase.app/"
            })
        else:
            firebase_key = json.loads(os.environ["FIREBASE_KEY"])
            cred = credentials.Certificate(firebase_key)
            firebase_admin.initialize_app(cred, {
                "databaseURL": os.environ["FIREBASE_DB_URL"]
            })
    except Exception as e:
        print("Firebase init error:", e)

# ── MODELS ────────────────────────────────────────────────────────────────── #
rf_model      = joblib.load("models/random_forest_model.pkl")
anomaly_model = joblib.load("models/isolation_forest_model.pkl")


# ══════════════════════════════════════════════════════════════════════════════
# WEATHER FORECAST  (OpenWeatherMap)
# ══════════════════════════════════════════════════════════════════════════════
def get_weather_forecast() -> dict:
    """
    Fetches next-3h forecast from OpenWeatherMap.
    Key is read from .env — NEVER hardcoded.
    Returns safe defaults on any failure so /predict never crashes.
    """
    api_key = os.environ.get("OPENWEATHER_API_KEY", "").strip()
    city    = os.environ.get("OPENWEATHER_CITY", "Coimbatore").strip()

    default = {
        "rain_forecast":      0.0,
        "temp_forecast":      None,
        "humidity_forecast":  None,
        "weather_desc":       "N/A",
    }

    if not api_key or api_key.startswith("your_"):
        print("⚠  OPENWEATHER_API_KEY not set — weather skipped")
        return default

    url = (
        "https://api.openweathermap.org/data/2.5/forecast"
        f"?q={city}&appid={api_key}&units=metric&cnt=3"
    )
    try:
        resp = requests.get(url, timeout=6)
        resp.raise_for_status()
        body = resp.json()

        if "list" not in body or not body["list"]:
            print("Weather API unexpected response:", body)
            return default

        slot = body["list"][0]
        desc = slot.get("weather", [{}])[0].get("description", "N/A").title()

        return {
            "rain_forecast":     round(slot.get("pop", 0) * 100, 1),  # 0-100 %
            "temp_forecast":     round(slot["main"]["temp"], 1),
            "humidity_forecast": slot["main"]["humidity"],
            "weather_desc":      desc,
        }
    except Exception as e:
        print(f"Weather API error: {e}")
        return default


# ══════════════════════════════════════════════════════════════════════════════
# SENSOR DATA  (Firebase realtime → CSV fallback)
# ══════════════════════════════════════════════════════════════════════════════
def get_sensor_data() -> dict:
    """
    Tries Firebase first (data must be < 60 s old).
    Falls back to last row of CSV.
    ESP32 rain sensor is active-low: raw 0 = raining, raw 1 = dry.
    We normalise: rainfall = 1 means IT IS raining.
    """
    try:
        sensor_data = db.reference("sensor").get()
        if sensor_data and "timestamp" in sensor_data:
            age = time.time() - float(sensor_data["timestamp"])
            if age < 60:
                print(f"🔥 Firebase realtime  (age {age:.0f}s)")
                raw_rain = int(sensor_data.get("rain", 1))
                rainfall = 1 if raw_rain == 0 else 0   # active-low normalise
                return {
                    "temperature":   float(sensor_data.get("temperature", 25)),
                    "humidity":      float(sensor_data.get("humidity", 50)),
                    "rainfall":      rainfall,
                    "soil_moisture": float(sensor_data.get("soil", 40)),
                    "source":        "realtime",
                }
            print(f"⚠  Firebase stale ({age:.0f}s) → CSV fallback")
    except Exception as e:
        print(f"Firebase read error: {e}")

    print("📄 CSV fallback")
    df  = pd.read_csv("./farm_sensor_data.csv")
    row = df.tail(1).to_dict(orient="records")[0]
    return {
        "temperature":   float(row["temperature"]),
        "humidity":      float(row["humidity"]),
        "rainfall":      int(row["rainfall"]),
        "soil_moisture": float(row["soil_moisture"]),
        "source":        "csv",
    }


# ══════════════════════════════════════════════════════════════════════════════
# CONFIDENCE SCORE  (fixed formula)
# ══════════════════════════════════════════════════════════════════════════════
def calc_confidence(tree_preds: np.ndarray) -> float:
    """
    Uses std-dev of tree predictions.
    std=0  → confidence 1.00  (all trees agree perfectly)
    std=10 → confidence 0.50
    std=20 → confidence 0.33
    Much more meaningful than the old variance-based formula.
    """
    std = float(np.std(tree_preds))
    return round(1.0 / (1.0 + std / 10.0), 3)


# ══════════════════════════════════════════════════════════════════════════════
# SMART DECISION ENGINE
# ══════════════════════════════════════════════════════════════════════════════
def smart_decision(
    predicted_soil: float,
    rainfall: int,
    humidity: float,
    temperature: float,
    rain_forecast: float,
    anomaly: str,
) -> dict:
    """
    Rule-based smart farming decisions.
    rainfall = 1  →  it IS currently raining.
    Returns: recommendation (str), tags (list), alert_level (str).
    """
    advice = []
    tags   = []
    alert  = "normal"   # normal | warning | critical

    # ── Anomaly override ─────────────────────────────────────────────────── #
    if anomaly == "Yes":
        return {
            "recommendation": "⚠ Abnormal climate detected. Inspect field immediately.",
            "tags":           ["Alert", "Monitoring"],
            "alert_level":    "critical",
        }

    # ── Irrigation decisions ──────────────────────────────────────────────── #
    if rainfall == 1:
        advice.append("🌧 Rain detected — do not irrigate now.")
        tags.append("Irrigation")
        alert = "warning"

    elif predicted_soil < 30:
        if rain_forecast > 60:
            advice.append(
                f"🌦 Rain expected ({rain_forecast:.0f}%) — skip irrigation, wait for rain."
            )
        else:
            advice.append("🚰 Soil critically dry — start irrigation immediately.")
            alert = "critical"
        tags.append("Irrigation")

    elif predicted_soil < 45:
        if rain_forecast > 60:
            advice.append(
                f"🌦 Rain likely ({rain_forecast:.0f}%) — hold irrigation for now."
            )
        elif rain_forecast > 40:
            advice.append(
                f"🌤 Moderate rain chance ({rain_forecast:.0f}%) — light irrigation if needed."
            )
        else:
            advice.append("💧 Soil moisture low — irrigation recommended.")
            if alert == "normal":
                alert = "warning"
        tags.append("Irrigation")

    elif predicted_soil > 70:
        advice.append("🚫 Soil moisture high — skip irrigation.")
        tags.append("Irrigation")

    # ── Fertilizer decisions ──────────────────────────────────────────────── #
    if rainfall == 1 or rain_forecast > 60:
        advice.append("🧪 Avoid fertilizer — rain will wash it away.")
        tags.append("Fertilizer")
    elif 40 <= predicted_soil <= 70 and rainfall == 0:
        advice.append("✅ Good conditions for fertilizer application.")
        tags.append("Fertilizer")

    # ── Pest / Disease alerts ─────────────────────────────────────────────── #
    if humidity > 85:
        advice.append("🦠 Very high humidity — high fungal disease risk. Apply fungicide.")
        tags.append("Pest/Disease")
        alert = "critical"
    elif humidity > 75:
        advice.append("🌫 Elevated humidity — monitor for early fungal signs.")
        tags.append("Pest/Disease")
        if alert == "normal":
            alert = "warning"

    # ── Temperature extremes ──────────────────────────────────────────────── #
    if temperature > 38:
        advice.append("🌡 Extreme heat — heat stress risk. Consider shade nets.")
        tags.append("Monitoring")
        if alert == "normal":
            alert = "warning"
    elif temperature < 15:
        advice.append("❄ Low temperature — frost risk. Protect crops.")
        tags.append("Monitoring")
        if alert == "normal":
            alert = "warning"

    # ── Default (all good) ────────────────────────────────────────────────── #
    if not advice:
        advice.append("✅ All conditions normal. Farm operations can proceed.")
        tags.append("Monitoring")

    return {
        "recommendation": " | ".join(advice),
        "tags":           list(dict.fromkeys(tags)),   # deduplicate, keep order
        "alert_level":    alert,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 6-HOUR IRRIGATION FORECAST
# ══════════════════════════════════════════════════════════════════════════════
def irrigation_forecast_6h(predicted_soil: float, rain_forecast: float) -> dict:
    if rain_forecast > 60:
        return {"irrigation_need": "Not Required", "reason": f"Rain expected ({rain_forecast:.0f}%)"}
    if predicted_soil < 30:
        return {"irrigation_need": "Urgent",       "reason": "Soil critically dry"}
    if predicted_soil < 45:
        return {"irrigation_need": "Recommended",  "reason": "Soil moisture below optimal"}
    if predicted_soil > 70:
        return {"irrigation_need": "Not Required", "reason": "Soil moisture sufficient"}
    return     {"irrigation_need": "Optional",     "reason": "Conditions borderline"}


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/predict")
def predict():
    # 1. Sensor data
    data        = get_sensor_data()
    temperature = float(data["temperature"])
    humidity    = float(data["humidity"])
    rainfall    = int(data["rainfall"])
    soil_lag1   = float(data["soil_moisture"])

    # 2. Weather forecast
    weather       = get_weather_forecast()
    rain_forecast = weather["rain_forecast"]

    # 3. Lag feature from CSV
    df        = pd.read_csv("./farm_sensor_data.csv")
    soil_lag2 = float(df.iloc[-2]["soil_moisture"]) if len(df) >= 2 else soil_lag1

    # 4. Random Forest prediction
    X = pd.DataFrame([{
        "temperature":        temperature,
        "humidity":           humidity,
        "rainfall":           rainfall,
        "soil_moisture_lag1": soil_lag1,
        "soil_moisture_lag2": soil_lag2,
    }])
    tree_preds     = np.array([t.predict(X)[0] for t in rf_model.estimators_])
    predicted_soil = float(np.mean(tree_preds))
    confidence     = calc_confidence(tree_preds)

    # 5. Anomaly detection
    X_anom = pd.DataFrame([{
        "temperature":   temperature,
        "humidity":      humidity,
        "soil_moisture": soil_lag1,
        "rainfall":      rainfall,
    }])
    anomaly_status = "Yes" if anomaly_model.predict(X_anom)[0] == -1 else "No"

    # 6. Smart decision
    decision = smart_decision(
        predicted_soil=predicted_soil,
        rainfall=rainfall,
        humidity=humidity,
        temperature=temperature,
        rain_forecast=rain_forecast,
        anomaly=anomaly_status,
    )

    # 7. Irrigation forecast
    irr = irrigation_forecast_6h(predicted_soil, rain_forecast)

    result = {
        "timestamp":               datetime.now().isoformat(),
        "temperature":             round(temperature, 2),
        "humidity":                round(humidity, 2),
        "rainfall":                rainfall,
        "predicted_soil_moisture": round(predicted_soil, 2),
        "confidence_score":        confidence,
        "anomaly":                 anomaly_status,
        # Smart decision outputs
        "recommendation":          decision["recommendation"],
        "tags":                    decision["tags"],
        "alert_level":             decision["alert_level"],
        # Weather
        "rain_forecast":           rain_forecast,
        "temp_forecast":           weather["temp_forecast"],
        "humidity_forecast":       weather["humidity_forecast"],
        "weather_desc":            weather["weather_desc"],
        # Irrigation
        "irrigation_next_6h":      irr["irrigation_need"],
        "irrigation_reason":       irr["reason"],
        # Meta
        "data_source":             data["source"],
    }

    # 8. Persist to Firebase
    try:
        db.reference("predictions").push(result)
    except Exception:
        pass

    return result


@app.get("/weather")
def weather_endpoint():
    """Standalone weather forecast endpoint."""
    return get_weather_forecast()


@app.get("/trend")
def trend():
    """Last 24 prediction records for trend charts."""
    try:
        data = db.reference("predictions").get()
        if not data:
            return {"temperature": [], "humidity": [], "soil_moisture": [],
                    "rain_forecast": [], "labels": []}

        records = sorted(data.values(), key=lambda x: x.get("timestamp", ""))[-24:]

        return {
            "labels":        [r.get("timestamp", "")[-8:-3] for r in records],  # HH:MM
            "temperature":   [r.get("temperature", 0)             for r in records],
            "humidity":      [r.get("humidity", 0)                for r in records],
            "soil_moisture": [r.get("predicted_soil_moisture", 0) for r in records],
            "rain_forecast": [r.get("rain_forecast", 0)           for r in records],
        }
    except Exception as e:
        return {"error": str(e), "temperature": [], "humidity": [],
                "soil_moisture": [], "rain_forecast": [], "labels": []}


@app.get("/history")
def history():
    """Last 20 full prediction records."""
    try:
        data = db.reference("predictions").get()
        if not data:
            return {"history": []}

        records = sorted(data.values(), key=lambda x: x.get("timestamp", ""))[-20:]

        # Normalise fields so old records (pre-upgrade) don't break the frontend
        clean = []
        for r in records:
            clean.append({
                "timestamp":               r.get("timestamp", ""),
                "predicted_soil_moisture": r.get("predicted_soil_moisture", 0),
                "confidence_score":        r.get("confidence_score", 0),
                "recommendation":          r.get("recommendation", ""),
                "anomaly":                 r.get("anomaly", "No"),
                "rain_forecast":           r.get("rain_forecast", 0),
                "alert_level":             r.get("alert_level", "normal"),
            })
        return {"history": clean}
    except Exception as e:
        return {"history": [], "error": str(e)}


@app.get("/")
def home():
    return {
        "status":    "running",
        "project":   "Agro Climate AI v2.0",
        "mode":      "Smart Farming Decision System",
        "endpoints": ["/predict", "/weather", "/trend", "/history"],
    }
