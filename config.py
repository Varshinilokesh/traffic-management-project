# config.py

import os
import joblib
from ultralytics import YOLO
import pytz 
from typing import Dict, Any

# ------------------------------
# File paths & constants
# ------------------------------
DATA_DIR = 'data'
CSV_FILE = os.path.join(DATA_DIR, "traffic_counts.csv")
GEOJSON_PATH = os.path.join(DATA_DIR, 'map.geojson')
MODEL_PATH = 'model.joblib' 
cookies_file = "C:/Users/Lenovo/OneDrive/Desktop/yolo model/cookies.txt"

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

# City mapping for each video index
CITY_MAP = {
    0: "Tokyo",
    1: "Colorado",
    2: "New York",
    3: "Roswell"
}

# ADDED: Timezone mapping for local time conversion
CITY_TIMEZONES = {
    "Tokyo": pytz.timezone("Asia/Tokyo"),
    "Colorado": pytz.timezone("America/Denver"),
    "New York": pytz.timezone("America/New_York"),
    "Roswell": pytz.timezone("America/Denver")
}

ALERT_LEVEL_SCORE = {
    'Low': 1,
    'Medium': 2,
    'High': 3,
    'Heavy': 4
}

youtube_urls = [
    "https://www.youtube.com/watch?v=6dp-bvQ7RWo", # Tokyo
    "https://www.youtube.com/watch?v=B0YjuKbVZ5w", # Colorado
    "https://www.youtube.com/watch?v=rnXIjl_Rzy4", # New York
    "https://www.youtube.com/watch?v=__S1lZ6t1qg", # Roswell
]

# ------------------------------
# Models
# ------------------------------
model_yolo = YOLO('yolov8/yolov8s.pt')

# Try load user's pre-trained classifier
model_predict = None
try:
    if os.path.exists("traffic_model.pkl"):
        model_predict = joblib.load("traffic_model.pkl")
except Exception:
    print("Warning: Could not load traffic_model.pkl")
    model_predict = None

# ------------------------------
# Streaming Configuration
# ------------------------------
CAPTURE_WIDTH = 640
CAPTURE_HEIGHT = 360
PROCESS_FPS = 2
QUEUE_MAXSIZE = 3
PRED_INTERVAL_SECONDS = 10

# Color/label mapping
COLOR_MAP = {0: 'red', 1: 'green', 2: 'blue', 3: 'yellow'}


def load_geojson(path=GEOJSON_PATH):
    """Helper to load GeoJSON data."""
    import json
    with open(path) as f:
        return json.load(f)
    
# 🟢 ACTION REQUIRED: REPLACE THIS with your actual Bot Token!
TELEGRAM_BOT_TOKEN = "8329599182:AAH17nkqLAcwoybfGQu7je9ru3C0ryzNyMU"  
TELEGRAM_API_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"

# ------------------------------
# Dummy Users (Update mobile_id to use Telegram Chat IDs)
# ------------------------------
# NOTE: These User IDs must match the keys in user_management.py USER_LOCATIONS
DUMMY_USERS = {
    # 1. Your ID is already here and correct
    101: {'name': 'Project Lead', 'mobile_id': '5835037205', 'city': 'Tokyo', 'alert_level': 'Heavy'},
    
    # 2. FIX: Replace 'FRIEND_A_CHAT_ID' with your ID for testing
    102: {'name': 'Friend A', 'mobile_id': '5835037205', 'city': 'Colorado', 'alert_level': 'High'}, 
    
    # 3. FIX: Replace 'FRIEND_B_CHAT_ID' with your ID for testing (this is the one failing in your log!)
    103: {'name': 'Friend B', 'mobile_id': '5835037205', 'city': 'New York', 'alert_level': 'Medium'},
    
    # 4. Your ID is already here and correct
    104: {'name': 'Project Lead (Roswell)', 'mobile_id': '5835037205', 'city': 'Roswell', 'alert_level': 'High'},
}

# ------------------------------
# Geographical Boundaries for Simulation
# ------------------------------
# Define a simulated 'alert radius' for each city (Latitude/Longitude ranges)
CITY_BOUNDARIES = {
    "Tokyo": {"lat_min": 35.69, "lat_max": 35.70, "lon_min": 139.69, "lon_max": 139.71},
    "Colorado": {"lat_min": 39.54, "lat_max": 39.56, "lon_min": -107.33, "lon_max": -107.31},
    "New York": {"lat_min": 40.75, "lat_max": 40.77, "lon_min": -73.99, "lon_max": -73.97},
    "Roswell": {"lat_min": 33.39, "lat_max": 33.40, "lon_min": -104.53, "lon_max": -104.51},
}