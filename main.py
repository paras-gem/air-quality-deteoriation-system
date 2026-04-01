import logging
import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import uvicorn
from model_manager import load_models
from surroundings import get_surroundings_data, get_nearest_city, CITIES

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="AIRQ Prediction API", version="2.0.0")

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Models
rf_1h, rf_24h, metadata = load_models()
FEATURES = metadata['features']

class PredictionRequest(BaseModel):
    city: str
    currentAqi: float
    aqi1h: float
    aqi24h: float
    temperature: float
    precipitation: float
    windSpeed: float
    hour: int
    month: int

class GeolocationRequest(BaseModel):
    lat: float
    lon: float

def run_prediction(data_dict: dict, city_name: str):
    """Core prediction logic helper."""
    # Construct dataframe matching trained features
    # Note: features list looks like ['AQI', 'Temperature_C', 'Precipitation_mm', 'WindSpeed_kmh', 'hour', 'month', 'AQI_lag1', 'AQI_lag24', 'city_Delhi', ...]
    input_data = {
        'AQI':              [data_dict['currentAqi']],
        'Temperature_C':    [data_dict['temperature']],
        'Precipitation_mm': [data_dict['precipitation']],
        'WindSpeed_kmh':    [data_dict['windSpeed']],
        'hour':             [data_dict['hour']],
        'month':            [data_dict['month']],
        'AQI_lag1':         [data_dict['aqi1h']],
        'AQI_lag24':        [data_dict['aqi24h']],
        'city_Delhi':       [1 if city_name == 'Delhi' else 0],
        'city_Hyderabad':   [1 if city_name == 'Hyderabad' else 0],
        'city_Kolkata':     [1 if city_name == 'Kolkata' else 0],
        'city_Mumbai':      [1 if city_name == 'Mumbai' else 0],
    }
    
    input_df = pd.DataFrame(input_data)[FEATURES]
    
    pred_1h = round(float(rf_1h.predict(input_df)[0]), 1)
    pred_24h = round(float(rf_24h.predict(input_df)[0]), 1)
    
    return pred_1h, pred_24h

@app.post("/predict")
async def predict_aqi(req: PredictionRequest):
    try:
        p1, p24 = run_prediction(req.dict(), req.city)
        return {
            "city": req.city,
            "oneHour": p1,
            "twentyFourHour": p24,
            "accuracy_1h": metadata.get('accuracy_1h', 0.85),
            "accuracy_24h": metadata.get('accuracy_24h', 0.80)
        }
    except Exception as e:
        logger.error(f"Prediction Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/detect-surroundings")
async def detect_surroundings(req: GeolocationRequest):
    """Fetches real data and runs prediction instantly for one set of coordinates."""
    try:
        raw_data = await get_surroundings_data(req.lat, req.lon)
        city_name = get_nearest_city(req.lat, req.lon)
        
        from datetime import datetime
        now = datetime.now()
        
        pred_input = {
            "currentAqi": raw_data['current_aqi'],
            "aqi1h": raw_data['aqi_1h'],
            "aqi24h": raw_data['aqi_24h'],
            "temperature": raw_data['temperature'],
            "precipitation": raw_data['precipitation'],
            "windSpeed": raw_data['wind_speed'],
            "hour": now.hour,
            "month": now.month
        }
        
        p1, p24 = run_prediction(pred_input, city_name)
        
        return {
            "location": raw_data['city_guess'],
            "nearest_model_city": city_name,
            "raw_sensors": raw_data,
            "predictions": {"oneHour": p1, "twentyFourHour": p24}
        }
    except Exception as e:
        logger.error(f"Detection Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/multi-predict")
async def multi_predict():
    """Returns predictions for all 4 cities for the home page table."""
    results = []
    from datetime import datetime
    now = datetime.now()
    
    for city_name, coords in CITIES.items():
        try:
            # We fetch mock/live data for each city
            raw_data = await get_surroundings_data(coords['lat'], coords['lon'])
            
            pred_input = {
                "currentAqi": raw_data['current_aqi'],
                "aqi1h": raw_data['aqi_1h'],
                "aqi24h": raw_data['aqi_24h'],
                "temperature": raw_data['temperature'],
                "precipitation": raw_data['precipitation'],
                "windSpeed": raw_data['wind_speed'],
                "hour": now.hour,
                "month": now.month
            }
            
            p1, p24 = run_prediction(pred_input, city_name)
            results.append({
                "city": city_name,
                "oneHour": p1,
                "twentyFourHour": p24
            })
        except Exception as e:
            logger.warning(f"Failed city {city_name}: {e}")
            
    return results

@app.get("/health")
def health():
    return {"status": "ok", "fastapi": True, "is_dummy": metadata.get('is_dummy', False)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
