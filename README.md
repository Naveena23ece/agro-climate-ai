🌱 AI-Enabled Agro-Climate Intelligence System
An intelligent agriculture decision support system that analyzes environmental data using Machine Learning and provides adaptive recommendations for irrigation, fertilizer application, and farm operations.
This system integrates IoT concepts, Machine Learning models, cloud APIs, and a real-time dashboard to help farmers make data-driven decisions.

🚀 Live Deployment
🌐 Dashboard
https://naveena23ece.github.io/agro-climate-ai/
⚙️ API Backend
https://agro-climate-ai-1.onrender.com
📘 API Documentation
https://agro-climate-ai-1.onrender.com/docs

🧠 Project Motivation
Agricultural decisions depend heavily on environmental conditions such as temperature, humidity, rainfall, and soil moisture. Farmers often rely on generalized weather forecasts, which do not represent farm-specific micro-climate conditions.
This project aims to build an AI-driven climate intelligence platform that transforms raw environmental data into actionable farming recommendations.

System Architecture:
Farm Sensors / Simulated Data
        ↓
FastAPI Backend (Cloud - Render)
        ↓
Machine Learning Layer(Random Forest + Isolation Forest)
        ↓
Decision Engine
        ↓
Firebase Realtime Database
        ↓
Interactive Web Dashboard (GitHub Pages)

🔧 Technologies Used
Backend:
-Python
-FastAPI
-Scikit-learn
-Pandas
-NumPy
Machine Learning:
-Random Forest Regression
-Isolation Forest (Anomaly Detection)
Cloud Services:
-Render (API deployment)
-Firebase Realtime Database
Frontend:
-HTML
-CSS
-JavaScript
-Chart.js
Deployment:GitHub Pages (Dashboard hosting)

⚙️ Machine Learning Models
🌿 Random Forest Regression
Used for predicting future soil moisture levels based on environmental conditions.
Features used:
-Temperature
-Humidity
-Rainfall
-Previous soil moisture levels

Advantages:
-Handles nonlinear relationships
-Works well with small datasets
-Robust and stable predictions

⚠️ Isolation Forest
Used for climate anomaly detection.
Detects abnormal patterns such as:
-Sudden temperature spikes
-Rapid humidity drops
-Unexpected soil moisture changes

🧠 Decision Engine
The decision engine combines:
-Current environmental conditions
-ML prediction results
-Anomaly detection output

to generate adaptive recommendations such as:
-Skip irrigation
-Delay fertilizer application
-Monitor abnormal climate conditions

📊 Dashboard Features
The web dashboard provides:
-Real-time environmental monitoring
-Soil moisture prediction results
-Anomaly detection alerts
-Climate trend visualization
-Historical prediction analysis
Charts are powered by Chart.js.

👩‍💻 Author
Naveena N
B.E Electronics and Communication Engineering
AI / Cloud / Software Systems Enthusiast
