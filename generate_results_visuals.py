#generate_results_visuals.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ===========================
# 1️⃣ Load dataset
# ===========================
df = pd.read_csv("water_potability.csv")

# Drop missing target rows
df = df.dropna(subset=['Potability'])

# Fill missing values with median for clean visualization
feature_cols = [
    'ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate',
    'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity'
]
df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median())

X = df[feature_cols]
y = df['Potability'].astype(int)

# ===========================
# 2️⃣ Load trained models
# ===========================
print("🔍 Loading trained models...")
rf_model = joblib.load("ml_models/rf_model.pkl")
xgb_model = joblib.load("ml_models/xgb_model.pkl")

# ===========================
# 3️⃣ Evaluate models
# ===========================
print("📊 Evaluating models on full dataset...")

y_pred_rf = rf_model.predict(X)
y_pred_xgb = xgb_model.predict(X)

rf_metrics = {
    "Model": "Random Forest",
    "Accuracy": accuracy_score(y, y_pred_rf),
    "Precision": precision_score(y, y_pred_rf),
    "Recall": recall_score(y, y_pred_rf),
    "F1-Score": f1_score(y, y_pred_rf)
}

xgb_metrics = {
    "Model": "XGBoost",
    "Accuracy": accuracy_score(y, y_pred_xgb),
    "Precision": precision_score(y, y_pred_xgb),
    "Recall": recall_score(y, y_pred_xgb),
    "F1-Score": f1_score(y, y_pred_xgb)
}

results_df = pd.DataFrame([rf_metrics, xgb_metrics])
results_df.to_csv("model_results.csv", index=False)
print("\n✅ Saved model_results.csv\n")
print(results_df)

# ===========================
# 4️⃣ Feature Importance (RF)
# ===========================
print("📈 Generating feature importance chart...")
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.bar(range(len(importances)), importances[indices], color='skyblue')
plt.xticks(range(len(importances)), np.array(feature_cols)[indices], rotation=45)
plt.title("Feature Importance - Random Forest")
plt.tight_layout()
plt.savefig("feature_importance_rf.png")
plt.close()
print("✅ Saved feature_importance_rf.png")

# ===========================
# 5️⃣ Potability Distribution
# ===========================
print("🥤 Generating Potability distribution chart...")
plt.figure(figsize=(5, 5))
df['Potability'].value_counts().plot(
    kind='pie', autopct='%1.1f%%', startangle=90,
    colors=['#90EE90', '#FF9999'], labels=['Safe', 'Unsafe']
)
plt.title("Potable vs Non-Potable Water Distribution")
plt.ylabel('')
plt.tight_layout()
plt.savefig("potability_pie.png")
plt.close()
print("✅ Saved potability_pie.png")

print("\n🎯 All visuals and tables generated successfully!")
