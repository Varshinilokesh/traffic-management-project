# alert_system.py

import time
import threading
import requests
from typing import Dict, Any

# Import configurations and state management
import config 
import user_management 

# --- NEW GLOBAL TRACKER ---
# Format: {(user_id, city): sent_count}
ALERT_SENT_COUNT = {}
# --------------------------

# --- Function to send messages via Telegram ---
def send_telegram_message(chat_id: str, message: str) -> bool:
    """Sends a message via the Telegram Bot API."""
    
    # Check if Telegram is configured (token is not empty/placeholder)
    is_configured = config.TELEGRAM_BOT_TOKEN != "YOUR_TELEGRAM_BOT_TOKEN" and config.TELEGRAM_BOT_TOKEN != "YOUR_ACTUAL_SECRET_TOKEN_HERE"
    
    if not is_configured:
        print("\n*** TELEGRAM BOT IS NOT CONFIGURED. FALLING BACK TO CONSOLE LOG. ***\n")
        return False
        
    # NOTE: The message is clean, as the formatting was previously fixed.
    payload = {
        'chat_id': chat_id,
        'text': message,
    }
    
    try:
        # Use the fully constructed URL from config.py
        response = requests.post(config.TELEGRAM_API_URL, data=payload)
        response.raise_for_status() 
        
        # NOTE: Print will be handled by the caller function (check_and_send_alerts)
        return True
        
    except requests.exceptions.RequestException as e:
        # The user's ID (mobile_id) is passed as chat_id
        print(f"\n[ALERT FAILED] Could not send Telegram message to {chat_id}. Error: {e}")
        return False

# --- Core alert function, updated to choose output method ---
def send_alert_to_mobile(mobile_id: str, message: str, method: str = 'Simulated Mobile Alert') -> bool:
    """Decides where to send the final message (Telegram or Console)."""
    
    # Attempt to send via Telegram if configured and successful
    if send_telegram_message(mobile_id, message):
        return True
        
    # Fallback to the console log if Telegram fails or is not configured
    print("\n" + "="*50)
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 🚨 {method} 🚨")
    print(f"-> To: {mobile_id}")
    print(f"-> Message: {message}")
    print("="*50 + "\n")
    return True

# --- Main alert logic, now with 2-time limit ---
def check_and_send_alerts():
    """Checks all active predictions and all users for real-time, location-based alerts."""
    
    for user_id, user_data in config.DUMMY_USERS.items():
        
        # 1. User and Location Check
        user_id_int = int(user_id) 
        city_near = user_management.is_user_near_city(user_id_int, config.CITY_BOUNDARIES)
        
        if not city_near:
            continue # User is too far away, skip alert.

        # 2. Prediction Check
        current_prediction = user_management.LATEST_PREDICTIONS.get(city_near)
        if not current_prediction:
            continue
            
        predicted_level = current_prediction['level']
        total_24h = current_prediction['total']

        current_score = config.ALERT_LEVEL_SCORE.get(predicted_level, 0)
        required_score = config.ALERT_LEVEL_SCORE.get(user_data['alert_level'], 0)
            
        # Check if prediction meets user's alert level
        if current_score >= required_score and current_score > 0: 
            
            # --- START ALERT LIMIT LOGIC ---
            alert_key = (user_id_int, city_near) # e.g., (103, 'New York')
            current_count = ALERT_SENT_COUNT.get(alert_key, 0)
            
            # Check 1: If the count is less than the limit (2)
            if current_count < 2:
                
                # Construct the clean, unformatted message for Telegram
                message = (
                    f"🚗 Live Traffic Alert for {city_near}! 🚨 "
                    f"You are currently NEAR this location. "
                    f"Predicted 24h traffic level is {predicted_level.upper()} "
                    f"(Total: {total_24h} vehicles). "
                    f"Please check alternative routes NOW."
                )
                
                # Attempt to send the alert
                if send_alert_to_mobile(
                    mobile_id=user_data['mobile_id'], 
                    message=message, 
                    method=f'Simulated Mobile Alert (Attempt {current_count + 1})'
                ):
                    # SUCCESS: Increment the counter only if Telegram send was successful
                    ALERT_SENT_COUNT[alert_key] = current_count + 1
                    print(f"[ALERT SUCCESS] Telegram Alert Sent (Count: {ALERT_SENT_COUNT[alert_key]}/2) for User {user_id} near {city_near}")

            # Check 2: If the count is equal to the limit
            elif current_count == 2:
                # Do nothing, the alert has been sent the maximum number of times
                print(f"[ALERT LIMIT REACHED] User {user_id} near {city_near} has already received 2 alerts.")
                
            # --- END ALERT LIMIT LOGIC ---

# --- Thread function to run the alert check constantly ---
def start_alert_monitoring(stop_event: threading.Event):
    """Monitors predictions and user locations to send real-time alerts."""
    print("\n=== Real-Time Alert Monitoring Started (Checking every 10s) ===\n")
    while not stop_event.is_set():
        # Check every 10 seconds 
        time.sleep(10) 
        try:
            check_and_send_alerts()
        except Exception as e:
            print(f"Alert Monitoring Thread Error: {e}")