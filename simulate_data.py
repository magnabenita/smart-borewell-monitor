#simulate_data.py
import requests
import random
import time
import datetime
import math

# Backend API URL
API_URL = "http://127.0.0.1:8000/add_data"

# Simulated borewell regions
REGIONS = ["Chennai", "Madurai", "Coimbatore", "Tirunelveli"]

# Each region starts with a baseline "real" value set
BASELINES = {
    region: {
        "ph": random.uniform(6.8, 7.5),
        "Hardness": random.uniform(100, 300),
        "Solids": random.uniform(500, 2000),
        "Chloramines": random.uniform(1.0, 3.5),
        "Sulfate": random.uniform(100, 300),
        "Conductivity": random.uniform(200, 800),
        "Organic_carbon": random.uniform(2, 10),
        "Trihalomethanes": random.uniform(20, 60),
        "Turbidity": random.uniform(1, 5),
        "TDS": random.uniform(200, 600),
        "temperature": random.uniform(24, 32)
    }
    for region in REGIONS
}

def drift_value(value, min_val, max_val, step=0.05, noise=0.1):
    """
    Simulate smooth drift: small directional change + random noise.
    """
    drift = random.uniform(-step, step)
    noisy_value = value + drift + random.gauss(0, noise)
    return max(min(noisy_value, max_val), min_val)

def generate_data(region):
    """Generate realistic correlated sensor data for one region."""
    base = BASELINES[region]

    # Slowly drift baseline values
    for key in base:
        if key in ["ph", "Turbidity", "temperature", "TDS", "Hardness", "Solids"]:
            base[key] = drift_value(base[key], 0, 8000, step=0.2, noise=0.05)

    # Correlations
    base["Hardness"] = base["TDS"] * random.uniform(0.4, 0.6)
    base["Solids"] = base["TDS"] * random.uniform(2.0, 3.0)
    base["Conductivity"] = base["TDS"] * random.uniform(0.5, 1.5)

    # Occasionally inject a turbidity spike (rain or pump)
    if random.random() < 0.05:
        base["Turbidity"] *= random.uniform(1.5, 3.0)

    # Occasionally simulate missing data
    if random.random() < 0.02:
        missing_field = random.choice(list(base.keys()))
        base[missing_field] = None

    # Build data packet
    data = {
        "region": region,
        "timestamp": datetime.datetime.now().isoformat(),
        "ph": round(base["ph"], 2) if base["ph"] else None,
        "Hardness": round(base["Hardness"], 2) if base["Hardness"] else None,
        "Solids": round(base["Solids"], 2) if base["Solids"] else None,
        "Chloramines": round(base["Chloramines"], 2),
        "Sulfate": round(base["Sulfate"], 2),
        "Conductivity": round(base["Conductivity"], 2),
        "Organic_carbon": round(base["Organic_carbon"], 2),
        "Trihalomethanes": round(base["Trihalomethanes"], 2),
        "Turbidity": round(base["Turbidity"], 2) if base["Turbidity"] else None,
        "TDS": round(base["TDS"], 2) if base["TDS"] else None,
        "temperature": round(base["temperature"], 2)
    }

    return data

if __name__ == "__main__":
    print("🌍 Starting *realistic* IoT data simulator...")
    while True:
        region = random.choice(REGIONS)
        data = generate_data(region)
        try:
            response = requests.post(API_URL, json=data, timeout=5)
            print(f"📡 [{region}] Sent:", data)
            print("🧠 Response:", response.text)
        except Exception as e:
            print("🚫 Error sending data:", e)
        time.sleep(5)
