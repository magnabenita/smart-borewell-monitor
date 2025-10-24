#ml_models/train_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
import joblib

# ========================
# Load and preprocess data
# ========================
df = pd.read_csv("../water_potability.csv")

# Drop rows with missing target
df = df.dropna(subset=['Potability'])

# Fill missing values in features with median
feature_cols = ['ph','Hardness','Solids','Chloramines','Sulfate',
                'Conductivity','Organic_carbon','Trihalomethanes','Turbidity']
df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median())

X = df[feature_cols]
y = df['Potability'].astype(int)

# ========================
# Split dataset
# ========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ========================
# Handle class imbalance
# ========================
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

# ========================
# Train Random Forest
# ========================
print("Training Random Forest with hyperparameter tuning...")

rf_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 8, 12, 16],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None],
    'class_weight': ['balanced']
}

rf = RandomForestClassifier(random_state=42)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

rf_grid = GridSearchCV(rf, rf_param_grid, cv=cv, scoring='f1', n_jobs=-1, verbose=1)
rf_grid.fit(X_train_res, y_train_res)

rf_best = rf_grid.best_estimator_
y_pred_rf = rf_best.predict(X_test)

print("Random Forest Classification Report:")
print(classification_report(y_test, y_pred_rf))

# Save RF model
joblib.dump(rf_best, "rf_model.pkl")
print("Random Forest model saved as rf_model.pkl")

# ========================
# Train XGBoost
# ========================
print("\nTraining XGBoost with hyperparameter tuning...")

xgb_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.7, 0.8, 1],
    'colsample_bytree': [0.7, 0.8, 1],
    'scale_pos_weight': [1, sum(y_train==0)/sum(y_train==1)]
}

xgb = XGBClassifier(
    eval_metric='logloss',  # suppress deprecated warning
    use_label_encoder=False,  # optional; ignored in latest versions
    random_state=42
)

xgb_grid = GridSearchCV(xgb, xgb_param_grid, cv=cv, scoring='f1', n_jobs=4, verbose=1)
xgb_grid.fit(X_train_res, y_train_res)

xgb_best = xgb_grid.best_estimator_
y_pred_xgb = xgb_best.predict(X_test)

print("XGBoost Classification Report:")
print(classification_report(y_test, y_pred_xgb))

# Save XGB model
joblib.dump(xgb_best, "xgb_model.pkl")
print("XGBoost model saved as xgb_model.pkl")
