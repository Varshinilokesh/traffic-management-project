import time
import threading
import requests
from typing import Dict, Any, Optional

# Import configurations and state management
import config 
import user_management 

# --- NEW GLOBAL TRACKER ---
# Format: {(user_id, city): sent_count}
ALERT_SENT_COUNT = {}
# --------------------------

# ====================================================================
# send_telegram_message (Unchanged)
# ====================================================================
def send_telegram_message(bot_token: str, chat_id: str, message: str) -> bool:
    """Sends a message via the Telegram Bot API using a specific bot token."""
    
    if not bot_token or "YOUR_" in bot_token:
        return False
        
    try:
        # Requires config.TELEGRAM_BASE_URL
        api_url = config.TELEGRAM_BASE_URL.format(bot_token)
    except AttributeError:
        print("Error: TELEGRAM_BASE_URL not defined in config.")
        return False
        
    payload = {
        'chat_id': chat_id,
        # Use Markdown formatting for bold text in public alerts
        'text': message,
        'parse_mode': 'Markdown' 
    }
    
    try:
        response = requests.post(api_url, data=payload)
        response.raise_for_status() 
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"\n[ALERT FAILED] Could not send Telegram message to {chat_id} (via bot token: {bot_token[-8:]}...)\nError: {e}")
        return False

# ====================================================================
# send_alert_to_mobile (Unchanged)
# ====================================================================
def send_alert_to_mobile(bot_token: str, mobile_id: str, message: str, method: str = 'Simulated Mobile Alert') -> bool:
    """Decides where to send the final message (Telegram or Console)."""
    
    if send_telegram_message(bot_token, mobile_id, message): 
        return True
        
    # Fallback to the console log if Telegram fails or is not configured
    print("\n" + "="*50)
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 🚨 {method} 🚨")
    print(f"-> To: {mobile_id}")
    print(f"-> Message: {message}")
    print("="*50 + "\n")
    return True

# ====================================================================
# NEW: send_public_alert function for broadcasting to the channel
# ====================================================================
def send_public_alert(city: str, level: str, total: int):
    """Broadcasts a critical alert to the single public Telegram channel."""
    
    # Use the token from the 'Tokyo' bot for the broadcast (it must be an admin of the public channel)
    bot_config = config.BOT_CREDENTIALS.get("Tokyo")
    
    if not bot_config or not bot_config['token']:
        print("ERROR: Cannot send public alert; 'Tokyo' bot token is missing.")
        return False
        
    bot_token = bot_config['token']
    
    # Define the threshold for public alerts (only 'Heavy' will trigger a public message)
    if config.ALERT_LEVEL_SCORE.get(level, 0) < config.ALERT_LEVEL_SCORE.get('Heavy', 4):
        return False 

    # Note: Using Markdown syntax for bolding
    message = (
        f"🚨 📢 *PUBLIC TRAFFIC WARNING: {city.upper()}* 📢 🚨\n\n"
        f"A **{level.upper()}** traffic event is predicted for **{city}** in the next 24 hours (Total: {total} vehicles).\n"
        f"Please check alternative routes NOW."
    )
    
    print(f"\n[PUBLIC ALERT] Attempting to send '{level}' alert for {city} to public channel...")
    return send_telegram_message(
        bot_token=bot_token,
        chat_id=config.PUBLIC_ALERT_CHANNEL_ID,
        message=message
    )

# ====================================================================
# check_and_send_alerts (Modified for Dual Logic)
# ====================================================================
def check_and_send_alerts():
    """Checks all active predictions and all users for real-time, location-based alerts."""
    
    # -----------------------------------------------------------
    # 1. Private Alert Logic (Iterates over USERS)
    # -----------------------------------------------------------
    for user_id, user_data in config.DUMMY_USERS.items():
        
        # ... (Location and Prediction Checks - Unchanged)
        user_id_int = int(user_id) 
        city_near = user_management.is_user_near_city(user_id_int, config.CITY_BOUNDARIES)
        
        if not city_near:
            continue
            
        current_prediction = user_management.LATEST_PREDICTIONS.get(city_near)
        if not current_prediction:
            continue
            
        # ... (Get Bot Credentials - Unchanged)
        city_bot_credentials = config.BOT_CREDENTIALS.get(city_near)
        if not city_bot_credentials:
            print(f"[ERROR] No bot credentials found in config.BOT_CREDENTIALS for city: {city_near}")
            continue
            
        bot_token = city_bot_credentials['token']
        bot_name = city_bot_credentials['name']
            
        predicted_level = current_prediction['level']
        total_24h = current_prediction['total']

        current_score = config.ALERT_LEVEL_SCORE.get(predicted_level, 0)
        required_score = config.ALERT_LEVEL_SCORE.get(user_data['alert_level'], 0)
            
        alert_key = (user_id_int, city_near) 
            
        # CONDITION 1: Check if alert needs to be sent (Private Alert)
        if current_score >= required_score and current_score > 0: 
            
            # --- START ALERT LIMIT LOGIC ---
            current_count = ALERT_SENT_COUNT.get(alert_key, 0)
            
            if current_count < 2:
                
                message = (
                    f"🚗 Live Traffic Alert for {city_near}! 🚨 (via {bot_name})\n\n"
                    f"You are currently NEAR this location. "
                    f"Predicted 24h traffic level is {predicted_level.upper()} "
                    f"(Total: {total_24h} vehicles). "
                    f"Please check alternative routes NOW."
                )
                
                if send_alert_to_mobile(
                    bot_token=bot_token,
                    mobile_id=user_data['mobile_id'], 
                    message=message, 
                    method=f'Alert via {bot_name} (Attempt {current_count + 1})'
                ):
                    ALERT_SENT_COUNT[alert_key] = current_count + 1
                    print(f"[ALERT SUCCESS] Telegram Alert Sent (Count: {ALERT_SENT_COUNT[alert_key]}/2) for User {user_id} near {city_near}")

            elif current_count == 2:
                print(f"[ALERT LIMIT REACHED] User {user_id} near {city_near} has already received 2 alerts.")
                
        # CONDITION 2: Check if alert counter needs to be reset (Private Alert Reset)
        elif current_score < required_score:
            if ALERT_SENT_COUNT.get(alert_key, 0) > 0:
                ALERT_SENT_COUNT[alert_key] = 0
                print(f"[ALERT RESET] User {user_id} near {city_near}. Traffic dropped below required level. Alert counter reset to 0.") 
                
        # --- END ALERT LIMIT LOGIC ---

    # -----------------------------------------------------------
    # 2. Public Alert Logic (Iterates over CITIES/PREDICTIONS)
    # -----------------------------------------------------------
    for city_name, prediction in user_management.LATEST_PREDICTIONS.items():
        # Skip internal tracking keys used by the public logic itself
        if city_name.startswith('public_'):
            continue
            
        predicted_level = prediction.get('level')
        total_24h = prediction.get('total')
        
        if not predicted_level:
            continue
            
        public_alert_key = f"public_{city_name}"
        
        # CRITICAL: Only send a public alert if the level is 'Heavy'
        if config.ALERT_LEVEL_SCORE.get(predicted_level, 0) >= config.ALERT_LEVEL_SCORE.get('Heavy', 4):
            
            # Check if we have NOT sent the public alert for this specific 'Heavy' event yet
            if user_management.LATEST_PREDICTIONS.get(public_alert_key) != predicted_level:
                
                if send_public_alert(city_name, predicted_level, total_24h):
                    # Mark the public alert as sent for this level
                    user_management.LATEST_PREDICTIONS[public_alert_key] = predicted_level
                    
        # If the level drops below 'Heavy', clear the public alert state for that city
        elif user_management.LATEST_PREDICTIONS.get(public_alert_key) == 'Heavy':
             user_management.LATEST_PREDICTIONS.pop(public_alert_key, None)


# --- Thread function to run the alert check constantly (No change needed) ---
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