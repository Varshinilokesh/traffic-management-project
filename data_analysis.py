# data_analysis.py

import pandas as pd
import os
import time
import numpy as np
from datetime import datetime, timedelta 
from sklearn.linear_model import LinearRegression
import pytz 
import user_management 
import alert_system 

# Import constants from config.py
from config import CSV_FILE, CITY_MAP, PRED_INTERVAL_SECONDS, CITY_TIMEZONES, ALERT_LEVEL_SCORE 


# ====================================================================
# NEW FUNCTION: TRAFFIC LEVEL CLASSIFICATION (REQUIRED FOR ALERT SYSTEM)
# ====================================================================
def get_traffic_level(total_count: int) -> str:
    """
    Classifies the total vehicle count into a traffic level.
    The thresholds are arbitrary and should be defined based on your data/needs.
    """
    if total_count >= 15000:
        return 'Heavy'
    elif total_count >= 5000:
        return 'High'
    elif total_count >= 1000:
        return 'Medium'
    else:
        return 'Low'

# ====================================================================
# 🟢 NEW: UPDATE PREDICTIONS & TRIGGER ALERTS
# ====================================================================
def update_and_alert(city: str, prediction_results: dict):
    """
    Updates the LATEST_PREDICTIONS dictionary and triggers the alert system check.
    This function should be called after a successful prediction.
    """
    
    if not prediction_results.get('ok'):
        print(f"Skipping alert update for {city}: Prediction failed.")
        return

    predictions = prediction_results['predictions']
    if not predictions:
        return

    # 1. Calculate the 24-hour summary level
    # Sum up the total predicted vehicles for the next 24 intervals
    total_24h_predicted = sum(p['predicted_total'] for p in predictions)
    
    # Classify the total traffic volume
    level = get_traffic_level(total_24h_predicted)

    # 2. Store the summary prediction in the shared state
    user_management.LATEST_PREDICTIONS[city] = {
        'level': level,
        'total': total_24h_predicted,
        'timestamp': time.time()
    }
    
    print(f"\n[PREDICTION SAVED] {city}: Level={level}, Total={total_24h_predicted}")

    # 3. Trigger the alert check instantly (since a new prediction is ready)
    try:
        # Note: alert_system.check_and_send_alerts() will now use this updated data
        alert_system.check_and_send_alerts() 
    except Exception as e:
        print(f"Error during alert check triggered by {city}: {e}")

# ====================================================================
# EXISTING FUNCTIONS (NO CHANGE NEEDED IN LOGIC)
# ====================================================================

def _ensure_counts_csv():
    """Ensures the CSV file exists with the correct header."""
    if not os.path.exists(CSV_FILE):
        pd.DataFrame(
            columns=['timestamp', 'city', 'video_index', 'car', 'bus', 'truck', 'total']
        ).to_csv(CSV_FILE, index=False)

def save_counts_row(ts: float, video_index: int, car: int, bus: int, truck: int, total: int) -> None:
    """Appends a new count row to the CSV file."""
    _ensure_counts_csv()
    city = CITY_MAP.get(video_index, f"City_{video_index}")
    row = pd.DataFrame([{
        'timestamp': int(ts),
        'city': city,
        'video_index': int(video_index),
        'car': int(car),
        'bus': int(bus),
        'truck': int(truck),
        'total': int(total),
    }])
    row.to_csv(CSV_FILE, mode='a', header=False, index=False)

def read_last_samples(city: str, now_utc: float | None = None) -> pd.DataFrame:
    """Reads historical samples, prioritizing the last 7 days, then falling back to 365 days."""
    _ensure_counts_csv()
    df = pd.read_csv(CSV_FILE)
    if df.empty:
        return pd.DataFrame(columns=['timestamp', 'city', 'total'])

    df = df[df['city'] == city] 
    if df.empty:
        return pd.DataFrame(columns=['timestamp', 'city', 'total'])

    if now_utc is None:
        now_utc = time.time()
        
    df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['timestamp'])
    
    recent_limit = now_utc - 7 * 86400 
    fallback_limit = now_utc - 365 * 86400 

    recent_df = df[df['timestamp'] >= recent_limit]
    
    if not recent_df.empty:
        return recent_df.reset_index(drop=True)
    
    fallback_df = df[df['timestamp'] >= fallback_limit]
    
    return fallback_df.reset_index(drop=True)

def last_week_interval_totals(city: str, now_utc: float | None = None) -> pd.DataFrame:
    """Aggregates traffic totals into 1-hour intervals for historical analysis."""
    samples = read_last_samples(city, now_utc)
    if samples.empty:
        return pd.DataFrame(columns=['date', 'interval', 'interval_total'])
        
    tz = CITY_TIMEZONES.get(city)
    if tz is None:
        tz = pytz.utc # Fallback just in case
        
    # 1. Convert timestamp to UTC-aware datetime objects
    samples['dt'] = pd.to_datetime(samples['timestamp'], unit='s', utc=True)
    
    # 2. Convert UTC to the City's local timezone
    samples['dt'] = samples['dt'].dt.tz_convert(tz)
    
    samples['date'] = samples['dt'].dt.date
    samples['hour'] = samples['dt'].dt.hour
    
    # Groups by 1-hour interval (hour 0 to 23)
    samples['interval'] = samples['hour'] 
    
    interval_df = (
        samples.groupby(['date', 'interval'], as_index=False)['total']
        .sum()
        .rename(columns={'total': 'interval_total'})
    )
    return interval_df.sort_values(['date', 'interval']).reset_index(drop=True)


def predict_next_24_hours(city: str, now_utc: float | None = None, selected_dates_str: str = '') -> dict:
    """
    Predicts traffic totals for the next 24 1-hour intervals, 
    using all historical data or a specific set of historical dates if provided.
    
    🟢 MODIFIED: Calls update_and_alert() before returning.
    """
    if now_utc is None:
        now_utc = time.time()
        
    tz = CITY_TIMEZONES.get(city)
    if tz is None:
        tz = pytz.utc # Fallback
        
    utc_datetime = datetime.fromtimestamp(now_utc, pytz.utc)
    current_datetime = utc_datetime.astimezone(tz)
    
    current_hour = current_datetime.hour
    
    # Load all historical data
    interval_df = last_week_interval_totals(city, now_utc)

    # NEW LOGIC: Filter data based on selected historical dates
    if selected_dates_str:
        try:
            # Parse the comma-separated dates (e.g., '2025-10-02, 2025-10-05')
            selected_dates = [datetime.strptime(d.strip(), '%Y-%m-%d').date() for d in selected_dates_str.split(',') if d.strip()]
            
            if selected_dates:
                # Filter interval_df to include only the selected dates
                interval_df = interval_df[interval_df['date'].isin(selected_dates)]
                
                if interval_df.empty:
                    # Return error if no data matches the selected dates
                    return {'ok': False, 'predictions': [], 'error': 'No data found for the selected dates.'}
            
        except ValueError as e:
            print(f"Date parsing error: {e}. Falling back to all available data.")
            pass 
            
    preds = []
    
    # Loop 24 times for the next 24 hours
    for i in range(24):
        # Calculate the hour (0-23) being predicted
        hour_to_predict = (current_hour + i) % 24
        
        # history for THIS specific hour-of-day across the (potentially filtered) historical data
        hist = (
            interval_df[interval_df['interval'] == hour_to_predict]
            .sort_values('date')
        )

        if hist.empty:
            y_next = 0.0 # No data for this time-of-day (after filtering)
        else:
            y = hist['interval_total'].to_numpy(dtype=float)
            n = len(y)

            # Use a simple trend or average based on sample count
            if n >= 3:
                X = np.arange(n).reshape(-1, 1)
                lr = LinearRegression().fit(X, y)
                # Predict the next value (index n) in the trend
                y_next = float(lr.predict([[n]])[0]) 
            else:
                # With 1–2 samples, just average THIS interval’s values
                y_next = float(np.mean(y))
        
        end_hour = (hour_to_predict + 1) % 24
        prediction_dt = current_datetime + timedelta(hours=i)
        
        preds.append({
            'interval': f"{hour_to_predict:02d}:00-{end_hour:02d}:00",
            'predicted_total': int(round(max(0, y_next))),
            'date': prediction_dt.strftime('%b %d'),
            'day_of_week': prediction_dt.strftime('%a')
        })

    results = {'ok': True, 'predictions': preds, 'history': interval_df.to_dict(orient='records')}

    # 🟢 CRUCIAL STEP: Store prediction data and trigger the alert check
    update_and_alert(city, results)

    return results