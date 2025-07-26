
from fastapi import FastAPI, HTTPException, Query
from fastapi_mcp import FastApiMCP
from datetime import datetime
import pytz
import re
import uvicorn

app = FastAPI(title="Time Zone REST API", version="1.0.0")

# Common timezone mappings for locations
LOCATION_TIMEZONES = {
    'new_york': 'America/New_York',
    'london': 'Europe/London',
    'tokyo': 'Asia/Tokyo',
    'sydney': 'Australia/Sydney',
    'paris': 'Europe/Paris',
    'berlin': 'Germany',
    'moscow': 'Europe/Moscow',
    'beijing': 'Asia/Shanghai',
    'mumbai': 'Asia/Kolkata',
    'los_angeles': 'America/Los_Angeles',
    'chicago': 'America/Chicago',
    'toronto': 'America/Toronto',
    'singapore': 'Asia/Singapore',
    'dubai': 'Asia/Dubai',
    'cairo': 'Africa/Cairo',
    'lagos': 'Africa/Lagos',
    'sao_paulo': 'America/Sao_Paulo',
    'mexico_city': 'America/Mexico_City',
    'vancouver': 'America/Vancouver',
    'hong_kong': 'Asia/Hong_Kong'
}

def normalize_location(location):
    """Normalize location name for lookup"""
    return re.sub(r'[^a-zA-Z]', '_', location.lower().strip())

@app.get('/time/{location}')
def get_time_by_location(location: str):
    """Get current time for a specific location"""
    try:
        normalized_location = normalize_location(location)
        
        # Check if location is in our predefined mappings
        if normalized_location in LOCATION_TIMEZONES:
            timezone_str = LOCATION_TIMEZONES[normalized_location]
        else:
            # Try to use the location as a timezone directly
            timezone_str = location.replace('_', '/').replace('-', '/')
        
        # Get timezone
        try:
            timezone = pytz.timezone(timezone_str)
        except pytz.exceptions.UnknownTimeZoneError:
            raise HTTPException(
                status_code=400,
                detail={
                    'error': f'Unknown location or timezone: {location}',
                    'available_locations': list(LOCATION_TIMEZONES.keys()),
                    'note': 'You can also use standard timezone names like America/New_York'
                }
            )
        
        # Get current time in the specified timezone
        utc_now = datetime.now(pytz.UTC)
        local_time = utc_now.astimezone(timezone)
        
        return {
            'location': location,
            'timezone': str(timezone),
            'current_time': local_time.strftime('%Y-%m-%d %H:%M:%S %Z'),
            'iso_format': local_time.isoformat(),
            'utc_time': utc_now.strftime('%Y-%m-%d %H:%M:%S UTC')
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail={'error': str(e)})

@app.get('/time')
def get_time_by_query(location: str = Query(..., description="Location name or timezone")):
    """Get current time for a location specified in query parameter"""
    return get_time_by_location(location)

@app.get('/locations')
def get_available_locations():
    """Get list of available predefined locations"""
    return {
        'available_locations': list(LOCATION_TIMEZONES.keys()),
        'note': 'You can also use standard timezone names like America/New_York, Europe/Berlin, etc.'
    }

@app.get('/')
def home():
    """API documentation"""
    return {
        'message': 'Time Zone REST API',
        'endpoints': {
            'GET /time/{location}': 'Get current time for a specific location',
            'GET /time?location={location}': 'Get current time using query parameter', 
            'GET /locations': 'Get list of available predefined locations',
            'GET /': 'This help message'
        },
        'examples': [
            '/time/new_york',
            '/time/london', 
            '/time?location=tokyo',
            '/time/America/New_York'
        ]
    }


#what it means is that, it iternally do app.run(transport="streamable-http")
mcp = FastApiMCP(app)
mcp.mount()

def main():
    print("Starting REST server for timezone information...")
    print("Available endpoints:")
    print("- GET /time/{location} - Get time for location")
    print("- GET /time?location={location} - Get time using query param")
    print("- GET /locations - List available locations")  
    print("- GET / - API documentation")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
