#backend/app.py
from fastapi import FastAPI, Request
from pymongo import MongoClient
from joblib import load
import pandas as pd
import random

# Initialize FastAPI app
app = FastAPI()

# Connect to MongoDB Atlas
MONGO_URI = "mongodb+srv://<username>:<password>@cluster0.mongodb.net/"
client = MongoClient(MONGO_URI)
db = client["borewell_db"]
collection = db["readings"]

# Load trained model
model = load("../ml_models/rf_model.pkl")

@app.get("/")
def home():
    return {"message": "✅ Smart Borewell API is running"}

# Endpoint to receive IoT (simulated) data
@app.post("/add_data")
async def add_data(request: Request):
    try:
        data = await request.json()

        # Extract safely with defaults
        region = data.get("region", "Unknown")
        timestamp = data.get("timestamp", None)

        # Generate defaults only if missing
        ph = float(data.get("ph", 7.0))
        turbidity = float(data.get("Turbidity", 3.0))
        TDS = float(data.get("TDS", random.uniform(100, 800)))
        temperature = float(data.get("temperature", random.uniform(20, 35)))

        # Prepare DataFrame for model prediction
        X = pd.DataFrame([{
            "ph": ph,
            "Hardness": data.get("Hardness", 100),
            "Solids": data.get("Solids", 500),
            "Chloramines": data.get("Chloramines", 2),
            "Sulfate": data.get("Sulfate", 200),
            "Conductivity": data.get("Conductivity", 500),
            "Organic_carbon": data.get("Organic_carbon", 5),
            "Trihalomethanes": data.get("Trihalomethanes", 20),
            "Turbidity": turbidity
        }])

        prediction = int(model.predict(X)[0])

        # ✅ Clean record — no duplicate columns
        record = {
            "region": region,
            "ph": ph,
            "Hardness": data.get("Hardness"),
            "Solids": data.get("Solids"),
            "Chloramines": data.get("Chloramines"),
            "Sulfate": data.get("Sulfate"),
            "Conductivity": data.get("Conductivity"),
            "Organic_carbon": data.get("Organic_carbon"),
            "Trihalomethanes": data.get("Trihalomethanes"),
            "Turbidity": turbidity,
            "TDS": TDS,
            "temperature": temperature,
            "timestamp": timestamp,
            "potable": prediction
        }

        collection.insert_one(record)

        return {
            "status": "Data stored successfully",
            "prediction": prediction
        }

    except Exception as e:
        print("❌ ERROR in /add_data:", e)
        return {"status": "Error", "prediction": -1, "error": str(e)}


@app.get("/get_data")
def get_data():
    try:
        records = list(collection.find({}, {"_id": 0}))
        return records
    except Exception as e:
        print("❌ ERROR in /get_data:", e)
        return {"error": str(e)}
