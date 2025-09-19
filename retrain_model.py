import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# ===============================
# Config
# ===============================
DATA_FILE = "traffic_counts.csv"
MODEL_FILE = "traffic_model.pkl"
ROW_THRESHOLD = 500  # retrain only if >= 500 new rows

# ===============================
# Load dataset
# ===============================
if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(f"❌ Dataset {DATA_FILE} not found")

data = pd.read_csv(DATA_FILE)
print(f"📊 Loaded dataset with {len(data)} rows")

# ===============================
# Check if retraining is needed
# ===============================
meta_file = "last_train_meta.txt"
last_trained_rows = 0

if os.path.exists(meta_file):
    with open(meta_file, "r") as f:
        last_trained_rows = int(f.read().strip())

new_rows = len(data) - last_trained_rows
if new_rows < ROW_THRESHOLD:
    print(f"⚠️ Only {new_rows} new rows since last training. Skipping retrain.")
    exit(0)

# ===============================
# Features and Target
# ===============================
# Features DO NOT include total (to avoid leakage)
X = data[["city", "video_index", "car", "bus", "truck"]]

# Target label comes from total
def label(row):
    if row < 10:
        return "low"
    elif row < 20:
        return "medium"
    else:
        return "high"

y = data["total"].apply(label)

# ===============================
# Preprocessing
# ===============================
preprocessor = ColumnTransformer(
    transformers=[
        ("city", OneHotEncoder(handle_unknown="ignore"), ["city"]),
        ("num", "passthrough", ["video_index", "car", "bus", "truck"]),
    ]
)

pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
])

# ===============================
# Train-test split & training
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

pipeline.fit(X_train, y_train)

# ===============================
# Save model & update metadata
# ===============================
joblib.dump(pipeline, MODEL_FILE)
with open(meta_file, "w") as f:
    f.write(str(len(data)))

print(f"✅ Model retrained on {len(data)} rows and saved as {MODEL_FILE}")
