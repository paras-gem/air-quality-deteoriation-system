import os
import joblib
import json
import logging
import subprocess
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EXPORTS_DIR = os.path.join(BASE_DIR, '..', 'Model', 'exports')
TRAIN_SCRIPT = os.path.join(BASE_DIR, '..', 'Model', 'aqi predictor.py')

def generate_dummy_models():
    """Generates an ultra-fast dummy Random Forest model if the massive CSV is missing."""
    logging.warning("Generating fallback dummy model because CSV or script failed...")
    
    features = [
        'AQI', 'Temperature_C', 'Precipitation_mm', 'WindSpeed_kmh',
        'hour', 'month', 'AQI_lag1', 'AQI_lag24',
        'city_Delhi', 'city_Hyderabad', 'city_Kolkata', 'city_Mumbai'
    ]
    
    # Tiny 2-row dataset just to fit a dummy Random Forest
    data = pd.DataFrame(0, index=[0, 1], columns=features)
    y_1h = pd.Series([50, 60])
    y_24h = pd.Series([55, 65])
    
    rf_1h = RandomForestRegressor(n_estimators=10, random_state=42)
    rf_1h.fit(data, y_1h)
    
    rf_24h = RandomForestRegressor(n_estimators=10, random_state=42)
    rf_24h.fit(data, y_24h)
    
    if not os.path.exists(EXPORTS_DIR):
        os.makedirs(EXPORTS_DIR)
        
    joblib.dump(rf_1h, os.path.join(EXPORTS_DIR, 'rf_1h_model.pkl'))
    joblib.dump(rf_24h, os.path.join(EXPORTS_DIR, 'rf_24h_model.pkl'))
    
    metadata = {
        'features': features,
        'cities': ['Delhi', 'Hyderabad', 'Kolkata', 'Mumbai'],
        'accuracy_1h': 0.85,  # Dummy accuracy
        'accuracy_24h': 0.80, # Dummy accuracy
        'is_dummy': True
    }
    with open(os.path.join(EXPORTS_DIR, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=4)
        
    logging.info("Fully functional Dummy Random Forest model saved to 'exports/'.")


def load_models():
    """Loads models from disk, automatically generating them if completely missing."""
    model_1h_path = os.path.join(EXPORTS_DIR, 'rf_1h_model.pkl')
    model_24h_path = os.path.join(EXPORTS_DIR, 'rf_24h_model.pkl')
    metadata_path = os.path.join(EXPORTS_DIR, 'metadata.json')

    # If models are entirely missing, try to build them
    if not os.path.exists(model_1h_path) or not os.path.exists(model_24h_path):
        logging.warning("AI Models missing! Attempting to launch the automated training script...")
        
        try:
            # Try to run the actual training python script
            result = subprocess.run(['python', TRAIN_SCRIPT], cwd=os.path.dirname(TRAIN_SCRIPT), text=True, capture_output=True)
            
            if result.returncode != 0:
                logging.error(f"Training script failed (likely missing CSV data). Error output:\n{result.stderr}")
                generate_dummy_models()
            else:
                logging.info("Training script completed successfully!")
                logging.info(f"Output: {result.stdout.strip()[-500:]}")
                
        except Exception as e:
            logging.error(f"Failed to launch subprocess: {e}")
            generate_dummy_models()
    
    # Now they're guaranteed to exist, so load them
    logging.info("Loading Random Forest models into memory...")
    rf_1h = joblib.load(model_1h_path)
    rf_24h = joblib.load(model_24h_path)
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
        
    return rf_1h, rf_24h, metadata

