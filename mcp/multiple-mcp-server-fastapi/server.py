#!/usr/bin/env python3
"""
Multiple MCP Server FastAPI Integration

This module demonstrates how to combine multiple MCP (Model Context Protocol) servers
into a single FastAPI application. It showcases:
1. Combining multiple MCP servers (greet_server and bye_server)
2. Managing server lifecycles with FastAPI lifespan events
3. Mounting MCP servers as HTTP endpoints for external access
4. Proper resource management using AsyncExitStack

Architecture:
- Each MCP server (greet, bye) runs independently with its own tools
- FastAPI acts as a gateway, routing requests to appropriate MCP servers
- HTTP endpoints allow external clients to interact with MCP tools
- Lifespan management ensures servers start/stop cleanly with the main app
"""

import contextlib
import uvicorn
from greet_server import mcp as greet_mcp  # Import MCP server for greeting functionality
from bye_server import mcp as bye_mcp      # Import MCP server for farewell functionality
from fastapi import FastAPI



# Create a combined MCP lifespan manager
@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI Lifespan Event Handler for Multiple MCP Servers
    
    This async context manager handles the startup and shutdown of multiple MCP servers
    within the FastAPI application lifecycle. It ensures that:
    
    1. **Startup Phase**: Both MCP servers are initialized and started before the app serves requests
    2. **Runtime Phase**: All servers remain active and handle requests concurrently  
    3. **Shutdown Phase**: All servers are gracefully stopped when the app shuts down
    
    Why use AsyncExitStack:
    - Manages multiple async context managers (MCP servers) together
    - Ensures proper cleanup even if one server fails to start
    - Handles resources in reverse order of creation (LIFO - Last In, First Out)
    - Provides exception safety - if startup fails, already-started servers are cleaned up
    
    Flow:
    1. Create AsyncExitStack for resource management
    2. Start greet_mcp server and register it for cleanup
    3. Start bye_mcp server and register it for cleanup  
    4. Yield control back to FastAPI (app is now ready to serve requests)
    5. When app shuts down, AsyncExitStack automatically stops servers in reverse order
    """
    async with contextlib.AsyncExitStack() as stack:
        # Register and start the greet server
        # This server handles greeting-related MCP tools (like "echoHai")
        # session_manager.run() starts the MCP server and returns a context manager
        await stack.enter_async_context(greet_mcp.session_manager.run())
        
        # Register and start the bye server  
        # This server handles farewell-related MCP tools (like "echoBye")
        # Both servers run concurrently and independently
        await stack.enter_async_context(bye_mcp.session_manager.run())
        
        # Yield control back to FastAPI
        # At this point, both MCP servers are running and ready to handle requests
        # The app will serve HTTP requests until shutdown is requested
        yield
        
        # When the context exits (app shutdown), AsyncExitStack automatically:
        # 1. Stops bye_mcp server (last registered, first stopped)
        # 2. Stops greet_mcp server (first registered, last stopped)
        # 3. Cleans up all associated resources


# Create the main FastAPI application instance
app = FastAPI(
    lifespan=lifespan,  # Attach our custom lifespan manager for MCP servers
    title="Multiple MCP Server FastAPI Gateway",  # API documentation title
    description="A FastAPI gateway that combines multiple MCP servers for greeting and farewell functionality",
    version="1.0.0",  # API version for documentation
    docs_url="/docs",  # Swagger UI endpoint (default)
    redoc_url="/redoc"  # ReDoc UI endpoint (default)
)

# Mount MCP servers as HTTP endpoints
# This creates HTTP routes that external clients can call to interact with MCP tools

# Mount the greet MCP server at /greet path
# Creates endpoints like: POST /greet/tools/echoHai
# streamable_http_app() converts MCP server to ASGI-compatible HTTP application
# This allows HTTP clients to call MCP tools via REST API instead of MCP protocol
app.mount("/greet", greet_mcp.streamable_http_app())

# Mount the bye MCP server at /bye path  
# Creates endpoints like: POST /bye/tools/echoBye
# Both servers operate independently on different URL paths
# Clients can choose which server to interact with based on the path
app.mount("/bye", bye_mcp.streamable_http_app())

# Optional: Add a root endpoint for API information
@app.get("/")
async def root():
    """
    Root endpoint providing information about available MCP servers and their endpoints.
    
    Returns:
        dict: Information about mounted MCP servers and their available tools
    """
    return {
        "message": "Multiple MCP Server FastAPI Gateway",
        "servers": {
            "greet": {
                "path": "/greet",
                "description": "Greeting MCP server with tools for saying hello",
                "tools": ["echoHai"],
                "example_url": "POST /greet/tools/echoHai"
            },
            "bye": {
                "path": "/bye", 
                "description": "Farewell MCP server with tools for saying goodbye",
                "tools": ["echoBye"],
                "example_url": "POST /bye/tools/echoBye"
            }
        },
        "documentation": {
            "swagger": "/docs",
            "redoc": "/redoc"
        }
    }


if __name__ == "__main__":
    """
    Application Entry Point
    
    This block runs only when the script is executed directly (not imported as a module).
    It starts the FastAPI application using Uvicorn ASGI server.
    
    Uvicorn Configuration:
    - app: The FastAPI application instance to serve
    - host="0.0.0.0": Listen on all network interfaces (allows external connections)
    - port=8000: HTTP port to listen on
    - reload=False: Disable auto-reload for production (set to True for development)
    - log_level="info": Set logging level for server output
    
    Server URLs after startup:
    - Main API: http://localhost:8000/
    - Greet MCP tools: http://localhost:8000/greet/mcp
    - Bye MCP tools: http://localhost:8000/bye/mcp
    - API Documentation: http://localhost:8000/docs
    - Alternative docs: http://localhost:8000/redoc
    
    Example usage:
    python server.py  # Starts the server
    curl -X POST http://localhost:8000/greet/tools/echoHai -d '{"userName": "Alice"}'
    """
    print("🚀 Starting Multiple MCP Server FastAPI Gateway...")
    print("📍 Server will be available at: http://localhost:8000")
    print("📚 API Documentation: http://localhost:8000/docs")
    print("🔧 Greet tools: http://localhost:8000/greet/mcp")
    print("👋 Bye tools: http://localhost:8000/bye/mcp")
    
    # Run the FastAPI app with Uvicorn ASGI server
    uvicorn.run(
        app,              # FastAPI application instance
        host="0.0.0.0",   # Listen on all interfaces (0.0.0.0 = all IPs)
        port=8000,        # HTTP port
        log_level="info"  # Logging verbosity
    )
