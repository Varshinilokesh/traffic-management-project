# user_management.py

from typing import Dict, Any, Optional
import config # Import config to access CITY_BOUNDARIES

# A thread-safe dictionary to hold the current simulated location for each user
# Structure: {user_id: {'lat': float, 'lon': float}}
USER_LOCATIONS: Dict[int, Dict[str, float]] = {
    # 🟢 Set user 101's location to be within the Tokyo boundary
    101: {'lat': 35.695, 'lon': 139.70},  
    # Locations far from all other cities (default)
    102: {'lat': 28.0, 'lon': 77.0},
    103: {'lat': 28.0, 'lon': 77.0},
    104: {'lat': 28.0, 'lon': 77.0},
}

# A simple thread-safe dictionary to hold the latest traffic prediction for all cities
LATEST_PREDICTIONS: Dict[str, Dict[str, Any]] = {
    # 🟢 Mock prediction for Tokyo (level 'Heavy' will trigger user 101's alert)
    "Tokyo": {'level': 'Medium', 'total': 4000}, 
    "Colorado": {'level': 'Low', 'total': 100},
    "New York": {'level': 'Medium', 'total': 3000},
    "Roswell": {'level': 'High', 'total': 5000},
}

# 🟢 MODIFIED FUNCTION: Checks ALL boundaries and returns the city name
def is_user_near_city(user_id: int, boundaries: Dict[str, Any]) -> Optional[str]:
    """
    Checks if a user's GPS is within ANY city's alert radius.
    Returns the city name (str) if near, or None if too far.
    """
    user_loc = USER_LOCATIONS.get(user_id)
    if not user_loc:
        return None

    # Loop through ALL defined city boundaries from config
    for city, city_boundary in boundaries.items():
        # Check if user's lat/lon is within the defined boundary
        near = (city_boundary['lat_min'] <= user_loc['lat'] <= city_boundary['lat_max']) and \
               (city_boundary['lon_min'] <= user_loc['lon'] <= city_boundary['lon_max'])
        
        if near:
            return city  # Found the city! Return its name.

    return None # User is not near any tracked city