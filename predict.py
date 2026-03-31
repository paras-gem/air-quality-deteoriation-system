import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

print("1. Loading and cleaning data...")
# Load Data
df = pd.read_csv('india_aqi_v6_2024-2025.csv')
df['datetime'] = pd.to_datetime(df['datetime'])
df = df.sort_values(by=['city', 'datetime'])

print("2. Feature Engineering (Now Using Delta Strategy)...")
# Create Lag Features (Looking at the past)
df['AQI_lag1'] = df.groupby('city')['AQI'].shift(1)
df['AQI_lag24'] = df.groupby('city')['AQI'].shift(24)

# Grab the actual future AQI (we need this to measure our final accuracy)
df['Future_1h_AQI'] = df.groupby('city')['AQI'].shift(-1)
df['Future_24h_AQI'] = df.groupby('city')['AQI'].shift(-24)

# NEW: Create TARGET DELTAS (What the AI will actually predict: The Change in AQI)
df['Target_1h_Delta'] = df['Future_1h_AQI'] - df['AQI']
df['Target_24h_Delta'] = df['Future_24h_AQI'] - df['AQI']

# Drop missing values caused by shifting
df = df.dropna(subset=['Future_1h_AQI', 'Future_24h_AQI', 'AQI_lag1', 'AQI_lag24'])

# One-Hot Encode Cities
df_encoded = pd.get_dummies(df, columns=['city'], drop_first=False)

# Define our feature set
features = [
    'AQI', 'Temperature_C', 'Precipitation_mm', 'WindSpeed_kmh', 
    'hour', 'month', 'AQI_lag1', 'AQI_lag24',
    'city_Delhi', 'city_Hyderabad', 'city_Kolkata', 'city_Mumbai'
]

X = df_encoded[features]
y_1h_delta = df_encoded['Target_1h_Delta']
y_24h_delta = df_encoded['Target_24h_Delta']

# We also keep the actual future AQI specifically for evaluating our final accuracy
y_1h_actual = df_encoded['Future_1h_AQI']
y_24h_actual = df_encoded['Future_24h_AQI']

# Train-Test Split (80% training, 20% testing - shuffle must be False for time series)
# We split X, the deltas (for training), and the actuals (for checking our work)
X_train, X_test, y_1h_delta_train, y_1h_delta_test, y_1h_actual_train, y_1h_actual_test = train_test_split(
    X, y_1h_delta, y_1h_actual, test_size=0.2, random_state=42, shuffle=False)

_, _, y_24h_delta_train, y_24h_delta_test, y_24h_actual_train, y_24h_actual_test = train_test_split(
    X, y_24h_delta, y_24h_actual, test_size=0.2, random_state=42, shuffle=False)

print("3. Training Random Forest Models to predict AQI changes...")
rf_1h = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_1h.fit(X_train, y_1h_delta_train)

rf_24h = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_24h.fit(X_train, y_24h_delta_train)

print("\n--- MODEL ACCURACY EVALUATION ---")
# 1. Predict the DELTA (how much the AQI will change)
preds_1h_delta = rf_1h.predict(X_test)
preds_24h_delta = rf_24h.predict(X_test)

# 2. Reconstruct the final predicted AQI (Current AQI + Predicted Change)
preds_1h_abs = X_test['AQI'].values + preds_1h_delta
preds_24h_abs = X_test['AQI'].values + preds_24h_delta

# Calculate Accuracy Metrics comparing our reconstructed guesses to the actual future AQI
r2_1h = r2_score(y_1h_actual_test, preds_1h_abs)
mae_1h = mean_absolute_error(y_1h_actual_test, preds_1h_abs)
r2_24h = r2_score(y_24h_actual_test, preds_24h_abs)
mae_24h = mean_absolute_error(y_24h_actual_test, preds_24h_abs)

print(f"\n1-Hour Prediction Accuracy -> R2: {r2_1h:.4f}, MAE: {mae_1h:.2f}")
print(f"24-Hour Prediction Accuracy -> R2: {r2_24h:.4f}, MAE: {mae_24h:.2f}")

# Example comparison
print("\n--- TEST PREDICTION SAMPLE (First 5 test points) ---")
results = pd.DataFrame({
    'Actual_1h': y_1h_actual_test.values[:5],
    'Predicted_1h': preds_1h_abs[:5],
    'Actual_24h': y_24h_actual_test.values[:5],
    'Predicted_24h': preds_24h_abs[:5]
})
print(results)