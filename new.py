import pandas as pd
from datetime import datetime

# Your fixed file
CSV_FILE = "data/traffic_counts_fixed.csv"

# Load data (already has datetime_local column)
df = pd.read_csv(CSV_FILE)

# Just ensure datetime_local is string in the right format
df['datetime_local'] = pd.to_datetime(df['datetime_local']).dt.strftime("%Y-%m-%d %H:%M:%S")

# Keep only the required columns
df = df[['timestamp','city','video_index','car','bus','truck','total']]

# Save back to old style
df.to_csv("data/traffic_counts.csv", index=False)
print("✅ Restored old format as traffic_counts.csv")
