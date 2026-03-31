import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from model_manager import load_models

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Allows your React frontend to call this server

logger.info("Starting AIRQ Flask Server...")

# --- Load Models On Startup automatically via model_manager ---
rf_1h, rf_24h, metadata = load_models()
FEATURES = metadata['features']
logger.info(f"Loaded Random Forest models successfully! Expected features: {len(FEATURES)}")


def get_aqi_category(aqi):
    if aqi <= 50:   return "Good"
    elif aqi <= 100:return "Satisfactory"
    elif aqi <= 200:return "Moderate"
    elif aqi <= 300:return "Poor"
    elif aqi <= 400:return "Very Poor"
    else:           return "Severe"


# --- API Route ---
@app.route('/predict', methods=['POST'])
def predict():
    """
    Expects frontend JSON payload with camelCase matching React component:
    {
        "city": "Delhi",
        "currentAqi": "145",
        "aqi1h": "132",
        "aqi24h": "160",
        "temperature": "32",
        "precipitation": "0",
        "windSpeed": "12",
        "hour": "8",
        "month": "11"
    }
    """
    try:
        data = request.get_json()
        if not data:
            logger.error("Empty JSON payload received.")
            return jsonify({'error': 'Missing JSON payload.'}), 400
            
        # Parse inputs, mapping camelCase (React frontend) into floats
        city = str(data.get('city', 'Delhi'))
        
        try:
            current_aqi = float(data['currentAqi'])
            aqi_1h = float(data['aqi1h'])
            aqi_24h = float(data['aqi24h'])
            temp = float(data['temperature'])
            precip = float(data['precipitation'])
            wind_speed = float(data['windSpeed'])
            hour = int(data['hour'])
            month = int(data['month'])
        except (KeyError, ValueError) as ve:
            logger.error(f"Input validation failed: {ve}")
            return jsonify({'error': f"Invalid or missing numeric parameter: {str(ve)}"}), 400

        # Construct dataframe dynamically matching trained features
        input_data = {
            'AQI':              [current_aqi],
            'Temperature_C':    [temp],
            'Precipitation_mm': [precip],
            'WindSpeed_kmh':    [wind_speed],
            'hour':             [hour],
            'month':            [month],
            'AQI_lag1':         [aqi_1h],
            'AQI_lag24':        [aqi_24h],
            'city_Delhi':       [1 if city == 'Delhi' else 0],
            'city_Hyderabad':   [1 if city == 'Hyderabad' else 0],
            'city_Kolkata':     [1 if city == 'Kolkata' else 0],
            'city_Mumbai':      [1 if city == 'Mumbai' else 0],
        }

        input_df = pd.DataFrame(input_data)[FEATURES]

        # Model Inference
        pred_1h  = round(float(rf_1h.predict(input_df)[0]), 1)
        pred_24h = round(float(rf_24h.predict(input_df)[0]), 1)

        # Output schema match for frontend expected shapes ("oneHour", "twentyFourHour")
        response_payload = {
            'city': city,
            'oneHour':            pred_1h,
            'twentyFourHour':     pred_24h,
            'category_1h':        get_aqi_category(pred_1h),
            'category_24h':       get_aqi_category(pred_24h),
            'model_accuracy_1h':  metadata.get('accuracy_1h', 0),
            'model_accuracy_24h': metadata.get('accuracy_24h', 0),
            'is_dummy_model':     metadata.get('is_dummy', False)
        }
        
        logger.info(f"Successful prediction for {city}: 1H={pred_1h}, 24H={pred_24h}")
        return jsonify(response_payload), 200

    except Exception as e:
        logger.exception("Unexpected error occurred during inference")
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok', 
        'models_loaded': True, 
        'is_dummy_model': metadata.get('is_dummy', False)
    }), 200


if __name__ == '__main__':
    logger.info("🚀 AQI Prediction Backend running on http://localhost:5000")
    app.run(debug=True, port=5000, use_reloader=False)

