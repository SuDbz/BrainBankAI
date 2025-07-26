# MCP Time Calculator Server

A Model Context Protocol (MCP) server implementation that provides time information for different locations around the world.

## Features

- Uses official MCP Python SDK (https://github.com/modelcontextprotocol/python-sdk)
- Provides time information based on location names or timezone identifiers
- Supports both friendly location names and standard IANA timezone identifiers
- Returns comprehensive time information including local time, UTC time, and timezone details

## Available Functions

1. `get_time`: Get the current time for a specific location or timezone
   - Parameter: `location` (string) - Location name or timezone identifier
   
2. `list_locations`: Get a list of all supported location names
   - No parameters required

## Installation

Install the package and its dependencies:

```bash
cd mcp-server
pip install -e .
```

## Usage

### Running the Server

```bash
python main.py
```

The server uses stdio (standard input/output) to communicate according to the Model Context Protocol.

### Example MCP Messages

Request available functions:
```json
{"type": "functions_request"}
```

Call `get_time` function to get time for New York:
```json
{"type": "function_call", "function_call": {"name": "get_time", "arguments": {"location": "new_york"}}}
```

Call `list_locations` function:
```json
{"type": "function_call", "function_call": {"name": "list_locations", "arguments": {}}}
```

## Supported Locations

The server supports various locations including:
- new_york, london, tokyo, sydney, paris
- berlin, moscow, beijing, mumbai, los_angeles
- chicago, toronto, singapore, dubai, cairo
- lagos, sao_paulo, mexico_city, vancouver, hong_kong

You can also use standard IANA timezone identifiers like:
- America/New_York
- Europe/London
- Asia/Tokyo
- etc.

## Development

This implementation uses the official MCP Python SDK and follows the project's architecture patterns with error handling conventions and location normalization.

## Integration with VS Code

This MCP server is designed to work with VS Code's MCP integration. The project's configuration in `.vscode/mcp.json` currently points to the REST server's MCP endpoint, but can be updated to use this stdio-based server if needed.
