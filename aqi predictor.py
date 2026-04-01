import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np
import os
import joblib
import json

# Setup Exports Directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EXPORTS_DIR = os.path.join(BASE_DIR, 'exports')
if not os.path.exists(EXPORTS_DIR):
    os.makedirs(EXPORTS_DIR)

print(f"[OK] All packages imported! Working in: {BASE_DIR}")

# --- Step 3: Load Data ---
# Search for dataset in local directories
DATASET_NAME = 'india_aqi_v6_2024-2025.csv'
possible_paths = [
    os.path.join(BASE_DIR, DATASET_NAME),           # Inside Model folder
    os.path.join(os.path.dirname(BASE_DIR), DATASET_NAME), # One level up (AIRQ root)
    DATASET_NAME                                    # Current working dir
]

dataset_path = None
for path in possible_paths:
    if os.path.exists(path):
        dataset_path = path
        break

if not dataset_path:
    print(f"[ERROR] {DATASET_NAME} not found. Please place it in the {BASE_DIR} directory.")
    exit()

print(f"[OK] Found dataset at: {dataset_path}")
df = pd.read_csv(dataset_path)
df['datetime'] = pd.to_datetime(df['datetime'])
df = df.sort_values(by=['city', 'datetime'])

# --- Step 4: Feature Engineering ---
print("[INFO] Engineering features...")
df['AQI_lag1'] = df.groupby('city')['AQI'].shift(1)
df['AQI_lag24'] = df.groupby('city')['AQI'].shift(24)

df['Target_Next_1h_AQI'] = df.groupby('city')['AQI'].shift(-1)
df['Target_Next_24h_AQI'] = df.groupby('city')['AQI'].shift(-24)

df = df.dropna(subset=['Target_Next_1h_AQI', 'Target_Next_24h_AQI', 'AQI_lag1', 'AQI_lag24'])
df_encoded = pd.get_dummies(df, columns=['city'], drop_first=False)

features = [
    'AQI', 'Temperature_C', 'Precipitation_mm', 'WindSpeed_kmh',
    'hour', 'month', 'AQI_lag1', 'AQI_lag24',
    'city_Delhi', 'city_Hyderabad', 'city_Kolkata', 'city_Mumbai'
]

X = df_encoded[features]
y_1h = df_encoded['Target_Next_1h_AQI']
y_24h = df_encoded['Target_Next_24h_AQI']

# --- Step 5: Training ---
X_train, X_test, y_1h_train, y_1h_test = train_test_split(X, y_1h, test_size=0.2, random_state=42, shuffle=False)
_, _, y_24h_train, y_24h_test = train_test_split(X, y_24h, test_size=0.2, random_state=42, shuffle=False)

print("[TRAINING] Training models... please wait.")
rf_1h = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_1h.fit(X_train, y_1h_train)

rf_24h = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_24h.fit(X_train, y_24h_train)

# --- Step 6: Accuracy Reports ---
preds_1h = rf_1h.predict(X_test)
r2_1h = r2_score(y_1h_test, preds_1h)
print(f"[RESULT] 1-Hour Model R2: {r2_1h:.4f}")

preds_24h = rf_24h.predict(X_test)
r2_24h = r2_score(y_24h_test, preds_24h)
print(f"[RESULT] 24-Hour Model R2: {r2_24h:.4f}")

# --- Step 7: EXPORTING AS .PKL ---
print("\n[EXPORT] Exporting files to 'exports/' folder...")

# Save Models
joblib.dump(rf_1h, os.path.join(EXPORTS_DIR, 'rf_1h_model.pkl'))
joblib.dump(rf_24h, os.path.join(EXPORTS_DIR, 'rf_24h_model.pkl'))

# Save Metadata (Features list)
metadata = {
    'features': features,
    'cities': ['Delhi', 'Hyderabad', 'Kolkata', 'Mumbai'],
    'accuracy_1h': float(r2_1h),
    'accuracy_24h': float(r2_24h)
}
with open(os.path.join(EXPORTS_DIR, 'metadata.json'), 'w') as f:
    json.dump(metadata, f, indent=4)

# --- Step 8: Visuals ---
plt.figure(figsize=(10, 6))
feat_df = pd.DataFrame({'Feature': features, 'Importance': rf_1h.feature_importances_}).sort_values(by='Importance', ascending=False)
sns.barplot(x='Importance', y='Feature', data=feat_df)
plt.title('Feature Importance')
plt.savefig(os.path.join(EXPORTS_DIR, 'feature_importance.png'))

print("\n[DONE] ALL DONE!")
print(f"Files saved in: {EXPORTS_DIR}/")
print("1. rf_1h_model.pkl")
print("2. rf_24h_model.pkl")
print("3. metadata.json")