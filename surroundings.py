import httpx
import os
import asyncio
from datetime import datetime, timedelta

# --- OpenWeatherMap Configuration ---
# To use real data, set 'OPENWEATHER_API_KEY' in your environment or replace 'DUMMY'
API_KEY = os.getenv("OPENWEATHER_API_KEY", "DUMMY_KEY_REPLACE_ME")
BASE_URL = "http://api.openweathermap.org/data/2.5"

# Supported Cities Coordinates
CITIES = {
    "Delhi": {"lat": 28.6139, "lon": 77.2090},
    "Mumbai": {"lat": 19.0760, "lon": 72.8777},
    "Kolkata": {"lat": 22.5726, "lon": 88.3639},
    "Hyderabad": {"lat": 17.3850, "lon": 78.4867},
}

async def get_surroundings_data(lat: float, lon: float):
    """
    Fetches real-time weather and AQI (including 1h and 24h lags) for a location.
    If API_KEY is DUMMY, returns realistic mock data.
    """
    if "DUMMY" in API_KEY or API_KEY == "":
        return get_mock_data(lat, lon)

    try:
        async with httpx.AsyncClient() as client:
            # 1. Current Weather (Temp, Wind)
            weather_task = client.get(
                f"{BASE_URL}/weather", params={"lat": lat, "lon": lon, "appid": API_KEY, "units": "metric"}
            )
            
            # 2. Current Air Pollution
            aqi_task = client.get(
                f"{BASE_URL}/air_pollution", params={"lat": lat, "lon": lon, "appid": API_KEY}
            )
            
            # 3. Historical Air Pollution (for 1h and 24h lags)
            # We need data for 24h ago to get aqi24h
            now = datetime.now()
            start = int((now - timedelta(hours=25)).timestamp())
            end = int(now.timestamp())
            history_task = client.get(
                f"{BASE_URL}/air_pollution/history", 
                params={"lat": lat, "lon": lon, "start": start, "end": end, "appid": API_KEY}
            )

            w_res, a_res, h_res = await asyncio.gather(weather_task, aqi_task, history_task)
            
            weather = w_res.json()
            aqi_now = a_res.json()
            aqi_hist = h_res.json()

            # Process Historical (Lags)
            # h_res returns a list of hourly points. We find the one closest to -1h and -24h.
            history_list = aqi_hist.get('list', [])
            aqi_1h = aqi_now['list'][0]['main']['aqi'] * 50 # rough conversion to Indian AQI scale if needed, or use raw
            aqi_24h = aqi_1h
            
            if len(history_list) > 24:
                aqi_1h = history_list[-2]['main']['aqi'] * 60 # Mocking scale for now
                aqi_24h = history_list[0]['main']['aqi'] * 60

            return {
                "temperature": weather['main']['temp'],
                "wind_speed": weather['wind']['speed'] * 3.6, # m/s to km/h
                "current_aqi": aqi_now['list'][0]['main']['aqi'] * 60,
                "aqi_1h": aqi_1h,
                "aqi_24h": aqi_24h,
                "precipitation": weather.get('rain', {}).get('1h', 0),
                "city_guess": weather.get('name', 'Detected Area')
            }

    except Exception as e:
        print(f"External API Error: {e}")
        return get_mock_data(lat, lon)

def get_mock_data(lat: float, lon: float):
    """Generates realistic mock data for UI testing without an API key."""
    import random
    base_aqi = 120 + random.randint(-40, 60)
    return {
        "temperature": 28 + random.randint(-5, 8),
        "wind_speed": 10 + random.uniform(0, 15),
        "current_aqi": base_aqi,
        "aqi_1h": base_aqi - 5,
        "aqi_24h": base_aqi + 15,
        "precipitation": 0,
        "city_guess": "Simulated Area",
        "is_mock": True
    }

def get_nearest_city(lat: float, lon: float):
    """Basic distance mapper to find which model city is closest."""
    import math
    min_dist = float('inf')
    nearest = "Delhi"
    for name, coords in CITIES.items():
        dist = math.sqrt((lat - coords['lat'])**2 + (lon - coords['lon'])**2)
        if dist < min_dist:
            min_dist = dist
            nearest = name
    return nearest
