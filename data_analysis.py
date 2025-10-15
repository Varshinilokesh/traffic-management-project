# data_analysis.py

import pandas as pd
import os
import time
import numpy as np
from datetime import datetime, timedelta # <-- Added timedelta for date calculations
from sklearn.linear_model import LinearRegression

# Import constants from config.py
from config import CSV_FILE, CITY_MAP, PRED_INTERVAL_SECONDS 

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
    
    samples['dt'] = pd.to_datetime(samples['timestamp'], unit='s')
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


def predict_next_24_hours(city: str, now_utc: float | None = None) -> dict:
    """
    Predicts traffic totals for the next 24 1-hour intervals, 
    starting with the current hour. (REPLACING predict_today_intervals_remaining)
    """
    if now_utc is None:
        now_utc = time.time()
        
    current_datetime = datetime.fromtimestamp(now_utc)
    current_hour = current_datetime.hour
    
    interval_df = last_week_interval_totals(city, now_utc)

    preds = []
    
    # Loop 24 times for the next 24 hours
    for i in range(24):
        # Calculate the hour (0-23) being predicted
        hour_to_predict = (current_hour + i) % 24
        
        # history for THIS specific hour-of-day across the historical data
        hist = (
            interval_df[interval_df['interval'] == hour_to_predict]
            .sort_values('date')
        )

        if hist.empty:
            y_next = 0.0 # No data for this time-of-day
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
        
        # Calculate the end hour (handling 23:00-24:00 correctly as 00:00)
        end_hour = (hour_to_predict + 1) % 24

        # Calculate the actual date for this prediction (Day 0, Day 1)
        prediction_dt = current_datetime + timedelta(hours=i)
        
        preds.append({
            'interval': f"{hour_to_predict:02d}:00-{end_hour:02d}:00",
            'predicted_total': int(round(max(0, y_next))),
            # Add the date context for display in the table
            'date': prediction_dt.strftime('%b %d'),
            'day_of_week': prediction_dt.strftime('%a')
        })

    return {'ok': True, 'predictions': preds, 'history': interval_df.to_dict(orient='records')}