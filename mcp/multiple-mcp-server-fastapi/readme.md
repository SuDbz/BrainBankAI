# Multiple MCP Server FastAPI Integration

A comprehensive example demonstrating how to combine multiple Model Context Protocol (MCP) servers into a single FastAPI application. This project showcases advanced MCP patterns including server lifecycle management, HTTP endpoint mounting, and resource management.

## 🏗️ Project Overview

This application demonstrates a **gateway pattern** where multiple independent MCP servers are unified under a single FastAPI application. Each MCP server operates independently with its own tools, while FastAPI acts as a routing layer that exposes these tools via HTTP endpoints.

### Key Architecture Components

```
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI Gateway                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────┐              ┌─────────────────────┐   │
│  │   Greet Server  │              │    Bye Server       │   │
│  │                 │              │                     │   │
│  │ • echoHai tool  │              │ • echoBye tool      │   │
│  │ • HTTP: /greet  │              │ • HTTP: /bye        │   │
│  │ • MCP Protocol  │              │ • MCP Protocol      │   │
│  └─────────────────┘              └─────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                              │
                              │ HTTP/REST API
                              ▼
                    ┌─────────────────────┐
                    │   External Clients  │
                    │                     │
                    │ • Web Apps          │
                    │ • CLI Tools         │
                    │ • Other Services    │
                    └─────────────────────┘
```

## 🎯 What This Project Demonstrates

### **1. Multiple MCP Server Integration**
- **Independent Servers**: Each MCP server runs with its own tools and namespace
- **Concurrent Operation**: Multiple servers run simultaneously without interference
- **Isolated Functionality**: Greet server handles greetings, Bye server handles farewells

### **2. FastAPI Lifecycle Management**
- **Lifespan Events**: Proper startup and shutdown of MCP servers
- **Resource Safety**: Guaranteed cleanup using AsyncExitStack
- **Exception Handling**: Graceful degradation if servers fail to start

### **3. HTTP Gateway Pattern**
- **Protocol Translation**: Converts MCP protocol to HTTP REST endpoints
- **Multiple Endpoints**: Each server mounted on different URL paths
- **API Documentation**: Auto-generated Swagger/OpenAPI docs

### **4. Production-Ready Patterns**
- **Error Handling**: Comprehensive error management and logging
- **Resource Management**: Proper cleanup of async resources
- **Scalability**: Architecture supports adding more MCP servers

## 📁 Project Structure

```
multiple-mcp-server-fastapi/
├── server.py              # Main FastAPI application and gateway
├── greet_server.py        # MCP server for greeting functionality
├── bye_server.py          # MCP server for farewell functionality
├── pyproject.toml         # Python dependencies and metadata
├── requirements.txt       # Alternative dependency specification
├── README.md             # This comprehensive documentation
└── .venv/                # Virtual environment (if using venv)
```

## 🔧 Core Components Explained

### **server.py - FastAPI Gateway**
The main application that orchestrates multiple MCP servers:

```python
# Key responsibilities:
1. Import and initialize multiple MCP servers
2. Manage server lifecycles with FastAPI lifespan events
3. Mount MCP servers as HTTP endpoints
4. Provide API documentation and routing
5. Handle startup/shutdown gracefully
```

### **greet_server.py - Greeting MCP Server**
Independent MCP server focused on greeting functionality:

```python
# Features:
- FastMCP server with stateless HTTP support
- echoHai tool for personalized greetings
- Returns: "Hello, {userName}!"
- Accessible via: /greet/* endpoints
```

### **bye_server.py - Farewell MCP Server**
Independent MCP server focused on farewell functionality:

```python
# Features:
- FastMCP server with stateless HTTP support  
- echoBye tool for personalized farewells
- Returns: "Bye, {userName}!"
- Accessible via: /bye/* endpoints
```

## 🚀 Installation & Setup

### **Prerequisites**
- Python 3.13+
- uv package manager (recommended) or pip

### **1. Clone and Navigate**
```bash
cd multiple-mcp-server-fastapi
```

### **2. Install Dependencies**

**Using uv (recommended):**
```bash
# Install all dependencies
uv sync

# Add missing dependencies if needed
uv add fastapi uvicorn mcp
```

**Using pip:**
```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### **3. Verify Installation**
```bash
# Check if MCP servers can be imported
python -c "from greet_server import mcp as greet_mcp; print('✅ Greet server imported successfully')"
python -c "from bye_server import mcp as bye_mcp; print('✅ Bye server imported successfully')"
```

## 🏃 Running the Application

### **Start the Server**
```bash
# Using Python directly
python server.py

# Using uv
uv run server.py

# Using uvicorn directly (with auto-reload for development)
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```

### **Expected Startup Output**
```
🚀 Starting Multiple MCP Server FastAPI Gateway...
📍 Server will be available at: http://localhost:8000
📚 API Documentation: http://localhost:8000/docs
🔧 Greet tools: http://localhost:8000/greet/
👋 Bye tools: http://localhost:8000/bye/

INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### **4. Interactive API Documentation**

Visit these URLs in your browser for interactive API exploration:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

Both provide:
- Interactive tool testing
- Request/response examples
- Parameter documentation
- Authentication options (if configured)

## 🧠 Advanced Concepts Explained

### **1. AsyncExitStack Pattern**

```python
async with contextlib.AsyncExitStack() as stack:
    await stack.enter_async_context(greet_mcp.session_manager.run())
    await stack.enter_async_context(bye_mcp.session_manager.run())
    yield
```

**Why this pattern?**
- **Resource Safety**: Guarantees cleanup even if exceptions occur
- **Multiple Resources**: Manages several async context managers together
- **Reverse Cleanup**: Resources cleaned up in LIFO (Last In, First Out) order
- **Exception Propagation**: Proper error handling during startup/shutdown

### **2. FastAPI Lifespan Events**

```python
app = FastAPI(lifespan=lifespan)
```

**Benefits:**
- **Startup Tasks**: Initialize resources before serving requests
- **Shutdown Tasks**: Clean up resources when app terminates
- **State Management**: Share resources across request handlers
- **Health Checks**: Ensure all services are ready before accepting traffic

### **3. MCP Server Mounting**

```python
app.mount("/greet", greet_mcp.streamable_http_app())
```

**What this does:**
- **Protocol Translation**: Converts MCP protocol to HTTP
- **Namespace Isolation**: Each server gets its own URL prefix
- **Independent Scaling**: Servers can be scaled/modified independently
- **Client Flexibility**: Supports both MCP and HTTP clients

### **4. Stateless HTTP Mode**

```python
mcp = FastMCP("ServerName", stateless_http=True)
```

**Advantages:**
- **Scalability**: No server-side session state to manage
- **Load Balancing**: Requests can go to any server instance
- **Fault Tolerance**: Server restarts don't lose state
- **Cloud-Friendly**: Works well with containerized deployments

## 🔄 Development Workflow

### **Adding New MCP Servers**

1. **Create New Server File** (e.g., `math_server.py`):
```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("MathServer", stateless_http=True)

@mcp.tool(description="Add two numbers", name="add")
def add(a: int, b: int) -> int:
    return a + b
```

2. **Import in main server.py**:
```python
from math_server import mcp as math_mcp
```

3. **Add to lifespan manager**:
```python
await stack.enter_async_context(math_mcp.session_manager.run())
```

4. **Mount HTTP endpoint**:
```python
app.mount("/math", math_mcp.streamable_http_app())
```

### **Testing New Tools**
```bash
# Test the new math server
curl -X POST http://localhost:8000/math/tools/add \
  -H "Content-Type: application/json" \
  -d '{"a": 5, "b": 3}'
```

## 🐛 Troubleshooting

### **Common Issues and Solutions**

#### **1. Import Errors**
```
ModuleNotFoundError: No module named 'mcp'
```
**Solution:**
```bash
uv add mcp
# or
pip install mcp
```

#### **2. Server Startup Failures**
```
RuntimeError: Server failed to start
```
**Solution:**
- Check that all MCP servers are properly configured
- Verify no port conflicts (default: 8000)
- Check server logs for specific error messages

#### **3. Tool Not Found**
```
{"error": "Tool 'echoHai' not found"}
```
**Solution:**
- Verify tool is properly decorated with `@mcp.tool()`
- Check server mounting path in URL
- Ensure server started successfully during lifespan

#### **4. Connection Refused**
```
Connection refused on localhost:8000
```
**Solution:**
```bash
# Check if server is running
lsof -i :8000

# Start server if not running
python server.py
```

## 📊 Performance Considerations

### **Scaling Strategies**

1. **Horizontal Scaling**: Run multiple instances behind a load balancer
2. **Vertical Scaling**: Increase server resources (CPU, memory)
3. **Service Separation**: Split servers into microservices
4. **Caching**: Add Redis/Memcached for frequently accessed data

### **Monitoring & Observability**

```python
# Add logging
import logging
logging.basicConfig(level=logging.INFO)

# Add metrics (example with Prometheus)
from prometheus_client import Counter, Histogram
tool_calls = Counter('mcp_tool_calls_total', 'Total tool calls', ['server', 'tool'])
```

## 🔒 Security Considerations

### **Production Deployment Checklist**

- [ ] **Authentication**: Add API keys or OAuth
- [ ] **Rate Limiting**: Prevent abuse with request limits
- [ ] **Input Validation**: Sanitize all user inputs
- [ ] **HTTPS**: Use TLS encryption in production
- [ ] **CORS**: Configure appropriate CORS policies
- [ ] **Logging**: Implement comprehensive audit logging

## 🤝 Contributing

### **Adding Features**
1. Fork the repository
2. Create a feature branch
3. Add your MCP server implementation
4. Update documentation
5. Add tests
6. Submit a pull request

### **Code Style**
- Follow PEP 8 Python style guidelines
- Add comprehensive docstrings
- Include type hints where appropriate
- Write unit tests for new functionality

## 📚 Additional Resources

- **MCP Specification**: [Model Context Protocol Documentation](https://modelcontextprotocol.io/)
- **FastAPI Documentation**: [FastAPI Official Docs](https://fastapi.tiangolo.com/)
- **FastMCP Library**: [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)
- **Uvicorn Server**: [ASGI Server Documentation](https://www.uvicorn.org/)

## 🏷️ Version History

- **v1.0.0**: Initial implementation with greet and bye servers
- **v1.1.0**: Added comprehensive documentation and examples
- **v1.2.0**: Enhanced error handling and logging
- **Future**: Planning authentication, metrics, and deployment guides

---

This project demonstrates production-ready patterns for building scalable MCP server infrastructures with FastAPI. It serves as both a working example and a foundation for building more complex MCP-based applications.
