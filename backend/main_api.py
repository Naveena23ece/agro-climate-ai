"""
Agro Climate AI — FastAPI Backend v2.1
Production-hardened: every operation wrapped in try-except,
no crash can bring down the server.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import joblib
import firebase_admin
from firebase_admin import credentials, db
from datetime import datetime
import requests, os, time, json
from dotenv import load_dotenv

load_dotenv()

# ── APP ───────────────────────────────────────────────────────────────────── #
app = FastAPI(title="Agro Climate AI", version="2.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── FIREBASE (safe init) ──────────────────────────────────────────────────── #
try:
    if not firebase_admin._apps:
        if os.path.exists("firebase_key.json"):
            cred = credentials.Certificate("firebase_key.json")
        else:
            cred = credentials.Certificate(
                json.loads(os.environ.get("FIREBASE_KEY", "{}"))
            )
        firebase_admin.initialize_app(cred, {
            "databaseURL": os.environ.get(
                "FIREBASE_DB_URL",
                "https://agroclimateai-default-rtdb.asia-southeast1.firebasedatabase.app/"
            )
        })
    print("Firebase initialised OK")
except Exception as e:
    print("ERROR: Firebase init failed:", e)

# ── MODELS (load once at startup) ─────────────────────────────────────────── #
rf_model      = None
anomaly_model = None
try:
    rf_model      = joblib.load("models/random_forest_model.pkl")
    anomaly_model = joblib.load("models/isolation_forest_model.pkl")
    print("Models loaded OK")
except Exception as e:
    print("ERROR: Model loading failed:", e)

# ── CSV FALLBACK (load once, re-read only if needed) ──────────────────────── #
_csv_cache = None

def get_csv_row() -> dict:
    global _csv_cache
    try:
        df = pd.read_csv("./farm_sensor_data.csv")
        row = df.tail(1).to_dict(orient="records")[0]
        _csv_cache = {
            "temperature":       float(row.get("temperature", 28)),
            "humidity":          float(row.get("humidity", 60)),
            "rainfall":          int(row.get("rainfall", 0)),
            "soil_moisture":     float(row.get("soil_moisture", 50)),
            "soil_moisture_raw": None,
            "source":            "csv",
        }
    except Exception as e:
        print("ERROR: CSV read failed:", e)
        if _csv_cache is None:
            _csv_cache = {
                "temperature": 28.0, "humidity": 60.0,
                "rainfall": 0, "soil_moisture": 50.0,
                "soil_moisture_raw": None, "source": "fallback",
            }
    return _csv_cache


# ── SOIL ADC → % ──────────────────────────────────────────────────────────── #
DRY_VAL = 2800   # ADC in dry air  → 0 %
WET_VAL  = 1000  # ADC in water    → 100 %

def adc_to_soil_pct(raw) -> float:
    try:
        raw = float(raw)
        raw = max(WET_VAL, min(DRY_VAL, raw))
        return round((DRY_VAL - raw) / (DRY_VAL - WET_VAL) * 100.0, 1)
    except Exception:
        return 50.0   # safe default


# ── SENSOR DATA ───────────────────────────────────────────────────────────── #
def get_sensor_data() -> dict:
    try:
        sensor_data = db.reference("sensor").get()
        if sensor_data and "timestamp" in sensor_data:
            age = time.time() - float(sensor_data["timestamp"])
            if age < 60:
                print(f"Firebase realtime (age {age:.0f}s)")
                raw_rain = int(sensor_data.get("rain", 1))
                raw_soil = float(sensor_data.get("soil", 2000))
                return {
                    "temperature":       float(sensor_data.get("temperature", 28)),
                    "humidity":          float(sensor_data.get("humidity", 60)),
                    "rainfall":          1 if raw_rain == 0 else 0,
                    "soil_moisture":     adc_to_soil_pct(raw_soil),
                    "soil_moisture_raw": raw_soil,
                    "source":            "realtime",
                }
            print(f"Firebase stale ({age:.0f}s) → CSV")
    except Exception as e:
        print("ERROR: Firebase read:", e)
    return get_csv_row()


# ── WEATHER FORECAST ──────────────────────────────────────────────────────── #
_WEATHER_DEFAULT = {
    "rain_forecast": 0.0, "temp_forecast": None,
    "humidity_forecast": None, "weather_desc": "N/A",
}

def get_weather_forecast() -> dict:
    api_key = os.environ.get("OPENWEATHER_API_KEY", "").strip()
    city    = os.environ.get("OPENWEATHER_CITY", "Coimbatore").strip()

    if not api_key or api_key.startswith("your_"):
        return _WEATHER_DEFAULT.copy()

    try:
        url  = (f"https://api.openweathermap.org/data/2.5/forecast"
                f"?q={city}&appid={api_key}&units=metric&cnt=3")
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        body = resp.json()

        if "list" not in body or not body["list"]:
            print("ERROR: Weather API bad response:", body.get("message", ""))
            return _WEATHER_DEFAULT.copy()

        slot = body["list"][0]
        return {
            "rain_forecast":     round(float(slot.get("pop", 0)) * 100, 1),
            "temp_forecast":     round(float(slot["main"]["temp"]), 1),
            "humidity_forecast": int(slot["main"]["humidity"]),
            "weather_desc":      slot.get("weather", [{}])[0]
                                     .get("description", "N/A").title(),
        }
    except requests.exceptions.Timeout:
        print("ERROR: Weather API timeout")
    except requests.exceptions.RequestException as e:
        print("ERROR: Weather API request:", e)
    except Exception as e:
        print("ERROR: Weather API parse:", e)
    return _WEATHER_DEFAULT.copy()


# ── CONFIDENCE SCORE ──────────────────────────────────────────────────────── #
def calc_confidence(tree_preds: np.ndarray) -> float:
    try:
        std = float(np.std(tree_preds))
        return round(1.0 / (1.0 + std / 10.0), 3)
    except Exception:
        return 0.5


# ── SMART DECISION ENGINE ─────────────────────────────────────────────────── #
def smart_decision(soil: float, rainfall: int, humidity: float,
                   temperature: float, rain_forecast: float, anomaly: str) -> dict:
    """
    All decisions based on REAL sensor soil moisture (0–100 %).
    Thresholds: <30 critical dry | 30-50 low | 50-70 optimal | >70 high
    """
    if anomaly == "Yes":
        return {
            "recommendation": "⚠ Abnormal climate detected. Inspect field immediately.",
            "tags": ["Alert", "Monitoring"], "alert_level": "critical",
        }

    advice, tags, alert = [], [], "normal"

    # Irrigation
    if rainfall == 1:
        advice.append("🌧 Rain detected — irrigation not needed.")
        tags.append("Irrigation"); alert = "warning"
    elif soil < 30:
        if rain_forecast > 60:
            advice.append(f"🌦 Soil critically dry but rain expected ({rain_forecast:.0f}%) — wait 2h before irrigating.")
            if alert == "normal": alert = "warning"
        else:
            advice.append("🚰 Soil critically dry — start irrigation immediately.")
            alert = "critical"
        tags.append("Irrigation")
    elif soil < 50:
        if rain_forecast > 60:
            advice.append(f"🌦 Rain likely ({rain_forecast:.0f}%) — hold irrigation for now.")
        elif rain_forecast > 40:
            advice.append(f"🌤 Moderate rain chance ({rain_forecast:.0f}%) — light irrigation if needed.")
        else:
            advice.append("💧 Soil moisture low — irrigation recommended.")
            if alert == "normal": alert = "warning"
        tags.append("Irrigation")
    elif soil <= 70:
        if rainfall == 0:
            advice.append("✅ Soil moisture optimal — no irrigation needed.")
        tags.append("Monitoring")
    else:
        advice.append("🚫 Soil over-saturated — skip irrigation, allow soil to dry.")
        tags.append("Irrigation")

    # Fertilizer
    if rainfall == 1 or rain_forecast > 60:
        advice.append("🧪 Avoid fertilizer — rain will wash it away.")
        tags.append("Fertilizer")
    elif 40 <= soil <= 70 and rainfall == 0:
        advice.append("✅ Good conditions for fertilizer application.")
        tags.append("Fertilizer")

    # Pest / Disease
    if humidity > 85:
        advice.append("🦠 Very high humidity — fungal disease risk. Apply fungicide.")
        tags.append("Pest/Disease"); alert = "critical"
    elif humidity > 75:
        advice.append("🌫 Elevated humidity — monitor for fungal signs.")
        tags.append("Pest/Disease")
        if alert == "normal": alert = "warning"

    # Temperature
    if temperature > 38:
        advice.append("🌡 Extreme heat — heat stress risk. Consider shade nets.")
        tags.append("Monitoring")
        if alert == "normal": alert = "warning"
    elif temperature < 15:
        advice.append("❄ Low temperature — frost risk. Protect crops.")
        tags.append("Monitoring")
        if alert == "normal": alert = "warning"

    if not advice:
        advice.append("✅ All conditions normal. Farm operations can proceed.")
        tags.append("Monitoring")

    return {
        "recommendation": " | ".join(advice),
        "tags":           list(dict.fromkeys(tags)),
        "alert_level":    alert,
    }


# ── IRRIGATION FORECAST ───────────────────────────────────────────────────── #
def irrigation_forecast_6h(soil: float, rain_forecast: float) -> dict:
    if rain_forecast > 60:
        return {"irrigation_need": "Not Required", "reason": f"Rain expected ({rain_forecast:.0f}%)"}
    if soil < 30:
        return {"irrigation_need": "Urgent",       "reason": "Soil critically dry"}
    if soil < 50:
        return {"irrigation_need": "Recommended",  "reason": "Soil moisture below optimal"}
    if soil <= 70:
        return {"irrigation_need": "Not Required", "reason": "Soil moisture optimal"}
    return     {"irrigation_need": "Not Required", "reason": "Soil too wet — allow to dry"}


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

# Safe fallback response — returned if /predict crashes completely
FALLBACK_RESPONSE = {
    "timestamp": "", "temperature": 0, "humidity": 0, "rainfall": 0,
    "soil_moisture_realtime": 0, "soil_moisture_raw_adc": None,
    "predicted_soil_moisture": 0, "confidence_score": 0,
    "anomaly": "No", "recommendation": "System error — please refresh.",
    "tags": ["Monitoring"], "alert_level": "normal",
    "rain_forecast": 0, "temp_forecast": None,
    "humidity_forecast": None, "weather_desc": "N/A",
    "irrigation_next_6h": "Unknown", "irrigation_reason": "Data unavailable",
    "data_source": "error",
}


@app.get("/predict")
def predict():
    try:
        # 1. Sensor data (never raises — has its own fallback)
        data        = get_sensor_data()
        temperature = float(data.get("temperature", 28))
        humidity    = float(data.get("humidity", 60))
        rainfall    = int(data.get("rainfall", 0))
        soil_real   = float(data.get("soil_moisture", 50))

        # 2. Weather (never raises — returns defaults on failure)
        weather       = get_weather_forecast()
        rain_forecast = float(weather.get("rain_forecast", 0))

        # 3. ML prediction (wrapped — models may be None if load failed)
        predicted_soil = soil_real   # default: use real value
        confidence     = 0.5
        if rf_model is not None:
            try:
                # Lag feature — use real soil as both lags if CSV unavailable
                try:
                    df        = pd.read_csv("./farm_sensor_data.csv")
                    soil_lag2 = float(df.iloc[-2]["soil_moisture"]) if len(df) >= 2 else soil_real
                except Exception:
                    soil_lag2 = soil_real

                X = pd.DataFrame([{
                    "temperature":        temperature,
                    "humidity":           humidity,
                    "rainfall":           rainfall,
                    "soil_moisture_lag1": soil_real,
                    "soil_moisture_lag2": soil_lag2,
                }])
                tree_preds     = np.array([t.predict(X)[0] for t in rf_model.estimators_])
                predicted_soil = float(np.mean(tree_preds))
                confidence     = calc_confidence(tree_preds)
            except Exception as e:
                print("ERROR: RF prediction:", e)

        # 4. Anomaly detection (wrapped)
        anomaly_status = "No"
        if anomaly_model is not None:
            try:
                X_anom = pd.DataFrame([{
                    "temperature":   temperature,
                    "humidity":      humidity,
                    "soil_moisture": soil_real,
                    "rainfall":      rainfall,
                }])
                anomaly_status = "Yes" if anomaly_model.predict(X_anom)[0] == -1 else "No"
            except Exception as e:
                print("ERROR: Anomaly detection:", e)

        # 5. Smart decision — always uses REAL soil moisture
        try:
            decision = smart_decision(
                soil=soil_real, rainfall=rainfall, humidity=humidity,
                temperature=temperature, rain_forecast=rain_forecast,
                anomaly=anomaly_status,
            )
        except Exception as e:
            print("ERROR: Smart decision:", e)
            decision = {"recommendation": "System processing error.",
                        "tags": ["Monitoring"], "alert_level": "normal"}

        # 6. Irrigation forecast
        try:
            irr = irrigation_forecast_6h(soil_real, rain_forecast)
        except Exception as e:
            print("ERROR: Irrigation forecast:", e)
            irr = {"irrigation_need": "Unknown", "reason": "Error"}

        result = {
            "timestamp":               datetime.now().isoformat(),
            "temperature":             round(temperature, 2),
            "humidity":                round(humidity, 2),
            "rainfall":                rainfall,
            "soil_moisture_realtime":  round(soil_real, 1),
            "soil_moisture_raw_adc":   data.get("soil_moisture_raw"),
            "predicted_soil_moisture": round(predicted_soil, 2),
            "confidence_score":        confidence,
            "anomaly":                 anomaly_status,
            "recommendation":          decision["recommendation"],
            "tags":                    decision["tags"],
            "alert_level":             decision["alert_level"],
            "rain_forecast":           rain_forecast,
            "temp_forecast":           weather.get("temp_forecast"),
            "humidity_forecast":       weather.get("humidity_forecast"),
            "weather_desc":            weather.get("weather_desc", "N/A"),
            "irrigation_next_6h":      irr["irrigation_need"],
            "irrigation_reason":       irr["reason"],
            "data_source":             data.get("source", "unknown"),
        }

        # 7. Persist to Firebase (non-blocking, never crashes predict)
        try:
            db.reference("predictions").push(result)
        except Exception as e:
            print("ERROR: Firebase push:", e)

        return result

    except Exception as e:
        # Last-resort catch — server NEVER crashes
        print("ERROR: /predict top-level:", e)
        fallback = FALLBACK_RESPONSE.copy()
        fallback["timestamp"] = datetime.now().isoformat()
        return fallback


@app.get("/weather")
def weather_endpoint():
    try:
        return get_weather_forecast()
    except Exception as e:
        print("ERROR: /weather:", e)
        return _WEATHER_DEFAULT.copy()


@app.get("/trend")
def trend():
    try:
        data = db.reference("predictions").get()
        if not data:
            return {"labels": [], "temperature": [], "humidity": [],
                    "soil_moisture": [], "rain_forecast": []}
        records = sorted(data.values(), key=lambda x: x.get("timestamp", ""))[-24:]
        return {
            "labels":        [r.get("timestamp", "")[-8:-3] for r in records],
            "temperature":   [r.get("temperature", 0)             for r in records],
            "humidity":      [r.get("humidity", 0)                for r in records],
            "soil_moisture": [r.get("soil_moisture_realtime",
                               r.get("predicted_soil_moisture", 0)) for r in records],
            "rain_forecast": [r.get("rain_forecast", 0)           for r in records],
        }
    except Exception as e:
        print("ERROR: /trend:", e)
        return {"labels": [], "temperature": [], "humidity": [],
                "soil_moisture": [], "rain_forecast": []}


@app.get("/history")
def history():
    try:
        data = db.reference("predictions").get()
        if not data:
            return {"history": []}
        records = sorted(data.values(), key=lambda x: x.get("timestamp", ""))[-20:]
        clean = [{
            "timestamp":               r.get("timestamp", ""),
            "predicted_soil_moisture": r.get("soil_moisture_realtime",
                                        r.get("predicted_soil_moisture", 0)),
            "confidence_score":        r.get("confidence_score", 0),
            "recommendation":          r.get("recommendation", ""),
            "anomaly":                 r.get("anomaly", "No"),
            "rain_forecast":           r.get("rain_forecast", 0),
            "alert_level":             r.get("alert_level", "normal"),
        } for r in records]
        return {"history": clean}
    except Exception as e:
        print("ERROR: /history:", e)
        return {"history": []}


@app.get("/")
@app.head("/")
def home():
    return {
        "status":    "running",
        "project":   "Agro Climate AI v2.1",
        "mode":      "Smart Farming Decision System",
        "endpoints": ["/predict", "/weather", "/trend", "/history"],
    }
