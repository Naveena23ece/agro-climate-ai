#include <WiFi.h>
#include <FirebaseESP32.h>
#include "DHT.h"
#include <time.h>

// ---------------- WIFI ----------------
#define WIFI_SSID "KATHIR_HOME"
#define WIFI_PASSWORD "Namo@123"

// ---------------- FIREBASE ----------------
#define FIREBASE_HOST "agroclimateai-default-rtdb.asia-southeast1.firebasedatabase.app"
#define FIREBASE_AUTH "AIzaSyARlICahSf7whIIJMqn8CJExx4XDj6PaqA"

// ---------------- PINS ----------------
#define DHTPIN 4
#define DHTTYPE DHT11
#define SOIL_PIN 34
#define RAIN_PIN 27

DHT dht(DHTPIN, DHTTYPE);

FirebaseData fbdo;
FirebaseAuth auth;
FirebaseConfig config;

// ---------------- SETUP ----------------
void setup() {
  Serial.begin(115200);
  dht.begin();

  pinMode(SOIL_PIN, INPUT);
  pinMode(RAIN_PIN, INPUT_PULLUP);

  // WIFI CONNECT
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.print("Connecting to WiFi");

  while (WiFi.status() != WL_CONNECTED) {
    Serial.print(".");
    delay(500);
  }
  Serial.println("\nWiFi Connected!");

  // 🔥 TIME SYNC (IMPORTANT)
  configTime(0, 0, "pool.ntp.org");

  Serial.print("Syncing time");
  while (time(nullptr) < 100000) {
    Serial.print(".");
    delay(1000);
  }
  Serial.println("\nTime synced!");

  // FIREBASE INIT
  config.database_url = FIREBASE_HOST;
  config.signer.tokens.legacy_token = FIREBASE_AUTH;

  Firebase.begin(&config, &auth);
  Firebase.reconnectWiFi(true);

  Serial.println("Firebase initialized!");
}

// ---------------- LOOP ----------------
void loop() {

  float temperature = dht.readTemperature();
  float humidity = dht.readHumidity();
  int soil = analogRead(SOIL_PIN);
  int rain = digitalRead(RAIN_PIN);

  if (isnan(temperature) || isnan(humidity)) {
    Serial.println("❌ DHT Error!");
    delay(2000);
    return;
  }

  time_t now = time(nullptr);   // ✅ REAL UNIX TIME

  Serial.println("\n📡 Sending data to Firebase...");
  Serial.println("Temp: " + String(temperature));
  Serial.println("Humidity: " + String(humidity));
  Serial.println("Soil: " + String(soil));
  Serial.println("Rain: " + String(rain));
  Serial.println("Timestamp: " + String(now));

  // 🔥 SEND DATA (WITH ERROR CHECK)
  if (!Firebase.setFloat(fbdo, "/sensor/temperature", temperature))
    Serial.println("❌ Temp failed: " + fbdo.errorReason());

  if (!Firebase.setFloat(fbdo, "/sensor/humidity", humidity))
    Serial.println("❌ Humidity failed: " + fbdo.errorReason());

  if (!Firebase.setInt(fbdo, "/sensor/soil", soil))
    Serial.println("❌ Soil failed: " + fbdo.errorReason());

  if (!Firebase.setInt(fbdo, "/sensor/rain", rain))
    Serial.println("❌ Rain failed: " + fbdo.errorReason());

  if (!Firebase.setInt(fbdo, "/sensor/timestamp", now))
    Serial.println("❌ Timestamp failed: " + fbdo.errorReason());

  Serial.println("✅ Data sent successfully!");

  delay(5000); // send every 5 sec
}