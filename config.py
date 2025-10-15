# config.py

import os
import joblib
from ultralytics import YOLO

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