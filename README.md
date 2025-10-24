


#  Smart Borewell Water Quality Monitoring System

### Real-time IoT + Machine Learning + Streamlit Dashboard

* This project simulates a **Smart Borewell Monitoring System** that collects **water quality data** from IoT sensors, predicts **potability (safe or unsafe)** using ML models, and visualizes live readings via a **Streamlit dashboard**.  
* It uses **FastAPI** as the backend, **MongoDB Atlas** as the database, and **Random Forest/XGBoost** for predictive analytics.

---

## ⚙️ System Architecture

```

IoT Sensors → FastAPI Backend → ML Model (Random Forest / XGBoost)
       ↓                            ↑
Simulated Data (Python)             |
       ↓                            |
   MongoDB Atlas  ←------------------
       ↓
Streamlit Dashboard (Visualization)


```

---

## 🧠 Features

✅ Real-time IoT data simulation  
✅ Automatic potability prediction (Safe / Unsafe)  
✅ MongoDB Atlas cloud storage  
✅ Interactive dashboard with region & time trends  
✅ Auto-refresh visualization  
✅ Machine learning training (RF, XGBoost)  
✅ Performance reports & feature importance charts  

---

## 📁 Project Structure

```

smart_borewell_monitor/
│
├── backend/
│   ├── app.py                 # FastAPI backend API
│
├── dashboard/
│   ├── app.py                 # Streamlit visualization dashboard
│
├── ml_models/
│   ├── train_model.py         # Train & save ML models (RF, XGBoost)
│   ├── rf_model.pkl           # Saved Random Forest model
│   ├── xgb_model.pkl          # Saved XGBoost model
│
├── simulate_data.py           # IoT data simulator
├── generate_results_visuals.py# Generates charts & metrics
├── water_potability.csv       # Dataset
└── README.md

````

---

## 🧩 Tech Stack

| Component | Technology Used |
|------------|-----------------|
| **Backend** | FastAPI |
| **Database** | MongoDB Atlas |
| **Frontend/Dashboard** | Streamlit + Plotly |
| **Machine Learning** | Scikit-learn, XGBoost |
| **Data Handling** | Pandas, NumPy |
| **Model Persistence** | Joblib |
| **Visualization** | Plotly, Matplotlib |

---

## 🚀 How to Run the Project

### 1️⃣ Clone the repository

```bash
git clone https://github.com/magnabenita/smart-borewell-monitor.git
cd smart-borewell-monitor
````

---

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

### 3️⃣ Train the ML models

```bash
cd ml_models
python train_model.py
```

✅ This will train **Random Forest** and **XGBoost** models, saving them as:

* `rf_model.pkl`
* `xgb_model.pkl`

---

### 4️⃣ Start the backend (FastAPI)

```bash
cd ../backend
uvicorn app:app --reload
```

✅ Backend API runs on:
👉 `http://127.0.0.1:8000/`

Endpoints:

* `/` → API status
* `/add_data` → POST simulated IoT data
* `/get_data` → Fetch all data from MongoDB

---

### 5️⃣ Start IoT Data Simulation

```bash
python simulate_data.py
```

This script continuously sends **region-wise water quality readings** to the backend every few seconds.

Example output:

```
 Starting *realistic* IoT data simulator...
 [Chennai] Sent: {'ph': 7.1, 'Hardness': 180, ...}
 Response: {"status": "Data stored successfully", "prediction": 1}
```

---

### 6️⃣ Launch the Dashboard

```bash
cd dashboard
streamlit run app.py
```

✅ Dashboard URL:
 `http://localhost:8501`

Features:

* Auto-refresh at configurable intervals
* Region-wise Safe/Unsafe bar charts
* Line charts showing time trends for all parameters
* Full tabular view of recent readings

---

## 📊 Generated Reports & Visuals

Run:

```bash
python generate_results_visuals.py
```

It produces:

| File                        | Description                           |
| --------------------------- | ------------------------------------- |
| `model_results.csv`         | Accuracy, Precision, Recall, F1-Score |
| `feature_importance_rf.png` | Random Forest feature importance      |
| `potability_pie.png`        | Safe vs Unsafe water distribution     |

---

## 🧮 Example Output

**Model Results (sample):**

| Model         | Accuracy | Precision | Recall | F1-Score |
| ------------- | -------- | --------- | ------ | -------- |
| Random Forest | 0.93     | 0.91      | 0.90   | 0.90     |
| XGBoost       | 0.91     | 0.90      | 0.88   | 0.89     |

**Feature Importance Chart:**

<img width="500" height="300" alt="image" src="https://github.com/user-attachments/assets/28ae4efd-ce4a-4ac2-85ee-597f4df9e807" />


---

## ☁️ MongoDB Atlas Setup

1. Create a free cluster at [MongoDB Atlas](https://www.mongodb.com/atlas).
2. Add your IP to the access list.
3. Create a database named `borewell_db` and a collection named `readings`.
4. Update the connection string in `backend/app.py`:

```python
MONGO_URI = "mongodb+srv://<username>:<password>@cluster0.mongodb.net/"
```

---

## 🧾 Example API Request (Manual)

```bash
curl -X POST http://127.0.0.1:8000/add_data \
  -H "Content-Type: application/json" \
  -d '{
        "region": "Chennai",
        "ph": 7.2,
        "Hardness": 180,
        "Solids": 1050,
        "Turbidity": 2.5,
        "Conductivity": 580
      }'
```

Response:

```json
{
  "status": "Data stored successfully",
  "prediction": 1
}
```

---

## 📦 Future Enhancements

* 🔔 Email/SMS alerts for unsafe readings
* 🌡️ Real IoT sensor integration (NodeMCU, ESP32)
* 📱 Mobile app for farmers/local authorities
* 🧠 Auto ML model retraining from live data

---

## 👨‍💻 Contributors

**Magna Benita P** – Lead Developer
M.Tech (Integrated) – VIT
Contact: magnabenita123@gmail.com

---

## 🪪 License

This project is licensed under the MIT License — free to use and modify with attribution.

---

**💧 Smart Borewell — Safer Water, Smarter Future.**




