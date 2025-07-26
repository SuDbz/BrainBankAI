# AI Coding Instructions

## Project Architecture

This is a dual-service Python project implementing both a REST API server and Model Context Protocol (MCP) integration:

- **`rest-server/`**: FastAPI-based timezone service running on port 8000
- **`mcp-server/`**: MCP server stub (currently minimal implementation)
- **`.vscode/mcp.json`**: VS Code MCP configuration pointing to `http://127.0.0.1:8000/mcp`

## Key Technical Patterns

### FastAPI + MCP Integration
The `rest-server` uses `fastapi-mcp` to expose REST endpoints as MCP tools:
```python
from fastapi_mcp import FastApiMCP
mcp = FastApiMCP(app)
mcp.mount()  # Exposes endpoints at /mcp
```

### Location-to-Timezone Resolution
The core business logic uses a two-tier approach:
1. Normalize user input with `normalize_location()` (strips non-alpha, lowercases)
2. Check `LOCATION_TIMEZONES` dict for city mappings, fallback to direct timezone lookup
3. Use pytz for timezone conversion with comprehensive error handling

### Error Handling Convention
Use FastAPI's `HTTPException` with structured detail objects:
```python
raise HTTPException(status_code=400, detail={
    'error': 'descriptive message',
    'available_locations': list(LOCATION_TIMEZONES.keys()),
    'note': 'additional context'
})
```

## Development Workflow

### Running Services
- **REST Server**: `cd rest-server && python main.py` (starts uvicorn on 0.0.0.0:8000)
- **Dependencies**: Use `pip install -e .` in each service directory
- **Python Version**: Fixed at 3.13 (see `.python-version` files)

### Project Structure Conventions
- Each service is a separate Python package with its own `pyproject.toml`
- No shared dependencies or monorepo tooling - services are independent
- MCP integration happens through HTTP endpoint mounting, not shared code

### API Design Patterns
- Dual endpoint approach: both path params (`/time/{location}`) and query params (`/time?location=`)
- Self-documenting root endpoint (`/`) returns API structure
- Helper endpoints (`/locations`) for discoverability
- Consistent JSON response format with multiple time representations

## Integration Points

### VS Code MCP Configuration
The `.vscode/mcp.json` expects the rest-server's MCP endpoint at `http://127.0.0.1:8000/mcp`. Start rest-server before testing MCP integration.

### Timezone Data Dependencies
- `pytz` for timezone calculations (not Python's built-in `zoneinfo`)
- `LOCATION_TIMEZONES` dict maps friendly names to IANA timezone identifiers
- Fallback logic allows direct IANA timezone usage (e.g., "America/New_York")

## Code Quality Patterns

- Use type hints for FastAPI path/query parameters
- Docstrings follow Google style for API endpoints
- Location normalization handles edge cases (spaces, hyphens, case insensitivity)
- Comprehensive error responses include actionable alternatives
