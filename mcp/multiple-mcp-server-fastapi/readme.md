# Time Zone REST Server

A simple REST API server that returns the current time for any given location or timezone.

## Features

- Get current time for predefined locations (cities)
- Support for standard timezone names (e.g., America/New_York)
- Multiple endpoint formats for flexibility
- JSON responses with detailed time information
- List available predefined locations

## Installation

1. Install dependencies:
```bash
pip install -e .
```

## Usage

Start the server:
```bash
python main.py
```

The server will start on `http://localhost:5000`

## API Endpoints

### Get Time by Location (Path Parameter)
```
GET /time/<location>
```

Examples:
- `GET /time/new_york` - Get time for New York
- `GET /time/london` - Get time for London
- `GET /time/America/Chicago` - Using timezone name

### Get Time by Location (Query Parameter)
```
GET /time?location=<location>
```

Example:
- `GET /time?location=tokyo`

### List Available Locations
```
GET /locations
```

Returns a list of predefined location names.

### API Documentation
```
GET /
```

Returns API documentation and examples.

## Response Format

```json
{
  "location": "new_york",
  "timezone": "America/New_York",
  "current_time": "2025-07-26 10:30:45 EDT",
  "iso_format": "2025-07-26T10:30:45.123456-04:00",
  "utc_time": "2025-07-26 14:30:45 UTC"
}
```

## Supported Locations

The server includes predefined mappings for major cities:
- new_york, london, tokyo, sydney, paris, berlin, moscow
- beijing, mumbai, los_angeles, chicago, toronto, singapore
- dubai, cairo, lagos, sao_paulo, mexico_city, vancouver, hong_kong

You can also use any standard timezone name (e.g., `America/New_York`, `Europe/London`, etc.)

## Error Handling

If an unknown location is provided, the server returns a 400 error with available options:

```json
{
  "error": "Unknown location or timezone: invalid_location",
  "available_locations": ["new_york", "london", "tokyo", "..."],
  "note": "You can also use standard timezone names like America/New_York"
}
```