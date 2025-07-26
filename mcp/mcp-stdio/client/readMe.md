# MCP Client - Stdio Transport

A Model Context Protocol (MCP) client implementation that communicates with MCP servers using stdio (standard input/output) transport. This client demonstrates how to connect to, interact with, and manage MCP server processes.

## Application Overview

This MCP client provides a complete example of:
- **Server Process Management**: Automatically spawns and manages MCP server subprocesses
- **Protocol Communication**: Handles MCP message serialization and protocol compliance
- **Tool Execution**: Calls server-side tools/functions and handles responses
- **Resource Management**: Proper cleanup of processes, streams, and connections
- **Error Handling**: Robust error handling with guaranteed resource cleanup

### Key Features

- **Stdio Transport**: Uses standard input/output streams for server communication
- **Async/Await Pattern**: Non-blocking operations for better performance
- **Automatic Server Discovery**: Lists and discovers available tools from the server
- **Resource Safety**: Guaranteed cleanup using AsyncExitStack context management
- **Multiple Tool Calls**: Demonstrates calling the same tool multiple times

### 📝 Important Note: Stdio Mode vs Separate Server

**When using stdio transport (like this client), you DO NOT need to run the MCP server separately.**

        The stdio transport is ideal for local development and testing as it:
        - Automatically manages the server subprocess lifecycle
        - Uses standard input/output streams for communication
        - Doesn't require separate server startup or network configuration
        - Eliminates the need for manual server process management
        """Simply provide the server script path to the client, and it will handle everything:

```python
# The client automatically starts the server process
await client.connect_to_server("/path/to/your/mcp-server/main.py")
```

This is different from HTTP-based MCP servers where you need to:
1. Start the server manually: `python server.py`  
2. Then connect the client to the running server

**Stdio transport = Everything in one command! 🚀**

## Core Components

### MCPClient Class
The main class that encapsulates all MCP client functionality:

```python
class MCPClient:
    def __init__(self):
        self.session = None          # MCP session for protocol communication
        self.exit_stack = AsyncExitStack()  # Resource management
```

### Key Methods

1. **`connect_to_server()`**: Establishes connection to MCP server via stdio
2. **`call_tool()`**: Executes tools on the connected server
3. **`cleanup()`**: Properly closes all resources and terminates server process

## Installation & Usage

### Prerequisites
```bash
# Ensure you have Python 3.13+ and the MCP package installed
pip install "mcp[cli]"
```

### Running the Client
```bash
# Navigate to the client directory
cd mcp-clinet

# Activate virtual environment (if using one)
source .venv/bin/activate

# Run the client (this automatically starts the server!)
python main.py
# OR using uv
uv run main.py
```

**🔥 Pro Tip**: Notice how we only run the client? The stdio transport automatically starts the MCP server process in the background. No need for multiple terminals or manual server startup!

### Expected Output
```
Connected to server with tools: ['add']

Calling add tool with 5 + 7:
Result: 12

Calling add tool with 10 + 15:
Result: 25
```

---

## Python Async Programming Concepts

### 1. What is `asyncio` and When to Use It

**`asyncio`** is Python's built-in library for asynchronous programming. It allows you to write concurrent code that can handle multiple tasks without blocking.

#### When to Use `asyncio`:
- **I/O Operations**: File reading, network requests, database queries
- **Multiple Tasks**: When you need to do several things simultaneously
- **Server Applications**: Web servers, API clients, real-time applications
- **Long-running Operations**: Tasks that take time but don't use CPU continuously

#### How to Use `asyncio`:

**Basic Example:**
```python
import asyncio
import time

# Synchronous version (blocking)
def sync_task(name, delay):
    print(f"Starting {name}")
    time.sleep(delay)  # This blocks everything
    print(f"Finished {name}")

# Asynchronous version (non-blocking)
async def async_task(name, delay):
    print(f"Starting {name}")
    await asyncio.sleep(delay)  # This doesn't block other tasks
    print(f"Finished {name}")

# Running async tasks
async def main():
    # These run concurrently, not sequentially
    await asyncio.gather(
        async_task("Task 1", 2),
        async_task("Task 2", 1),
        async_task("Task 3", 3)
    )

# Execute the async main function
asyncio.run(main())
```

**Output:**
```
Starting Task 1
Starting Task 2  
Starting Task 3
Finished Task 2  # Finishes first (1 second)
Finished Task 1  # Finishes second (2 seconds)
Finished Task 3  # Finishes last (3 seconds)
```

### 2. What is `async` and Why It's Used

**`async`** is a keyword that defines an asynchronous function (coroutine). When combined with `await`, it allows functions to pause execution and let other tasks run.

#### Why Use `async`:
- **Non-blocking Operations**: Other code can run while waiting for I/O
- **Better Resource Utilization**: More efficient than threading for I/O-bound tasks
- **Scalability**: Can handle thousands of concurrent operations
- **Modern Python**: The preferred way for concurrent programming

#### When to Use `async`:
- Functions that perform I/O operations (network, file, database)
- Functions that call other async functions
- Functions that need to wait for external resources

#### How to Use `async`:

**Simple Example:**
```python
import asyncio
import aiohttp  # Async HTTP client

# Regular function - blocks everything
def fetch_url_sync(url):
    import requests
    response = requests.get(url)
    return response.status_code

# Async function - doesn't block other operations
async def fetch_url_async(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return response.status

# Using async functions
async def main():
    # These run concurrently
    results = await asyncio.gather(
        fetch_url_async("https://httpbin.org/delay/1"),
        fetch_url_async("https://httpbin.org/delay/2"),
        fetch_url_async("https://httpbin.org/delay/1")
    )
    print(f"Status codes: {results}")

asyncio.run(main())
```

**In Our MCP Client Context:**
```python
# Why our MCP functions are async:

async def connect_to_server(self, server_script_path: str):
    # This creates a subprocess - I/O operation
    stdio_transport = await self.exit_stack.enter_async_context(
        stdio_client(server_params)  # Doesn't block while starting server
    )
    
    # This initializes network-like communication - I/O operation  
    await self.session.initialize()  # Doesn't block during handshake

async def call_tool(self, tool_name: str, arguments: dict):
    # This sends a message and waits for response - I/O operation
    result = await self.session.call_tool(tool_name, arguments)
    return result  # Doesn't block while waiting for server response
```

### 3. What is `AsyncExitStack`

**`AsyncExitStack`** is an async context manager that manages multiple async resources and ensures they're properly cleaned up, even if errors occur.

#### Purpose:
- **Resource Management**: Automatically cleanup async resources
- **Exception Safety**: Ensures cleanup happens even if exceptions occur
- **Multiple Resources**: Manages several async context managers together
- **Deterministic Cleanup**: Resources are cleaned up in reverse order of creation

#### How It Works:
```python
from contextlib import AsyncExitStack

async def example_usage():
    async with AsyncExitStack() as stack:
        # Resources are registered with the stack
        file1 = await stack.enter_async_context(async_open_file("file1.txt"))
        connection = await stack.enter_async_context(async_connect_db())
        session = await stack.enter_async_context(async_create_session())
        
        # Do work with resources...
        
        # When leaving this block, all resources are automatically closed
        # in reverse order: session, connection, file1
```

#### In Our MCP Client:
```python
class MCPClient:
    def __init__(self):
        self.exit_stack = AsyncExitStack()  # Created once
    
    async def connect_to_server(self, server_script_path: str):
        # Register the stdio transport for automatic cleanup
        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(server_params)  # Server subprocess
        )
        
        # Register the session for automatic cleanup
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(self.stdio, self.write)  # MCP session
        )
    
    async def cleanup(self):
        # This closes ALL registered resources automatically:
        # 1. Closes MCP session
        # 2. Terminates server subprocess  
        # 3. Closes all file handles/streams
        await self.exit_stack.aclose()
```

### 4. What is `self.session`?

**`self.session`** is an instance of `ClientSession` from the MCP library that handles all protocol-level communication with the MCP server.

#### Purpose:
- **Protocol Management**: Handles MCP message format and protocol compliance
- **Communication Channel**: Sends requests and receives responses from server
- **State Management**: Maintains connection state and session information
- **Error Handling**: Manages protocol-level errors and retries

#### What It Does:
```python
# self.session responsibilities:

# 1. Protocol Initialization
await self.session.initialize()  # Handshake with server

# 2. Tool Discovery
response = await self.session.list_tools()  # Get available tools

# 3. Tool Execution  
result = await self.session.call_tool("add", {"a": 5, "b": 7})

# 4. Message Serialization
# Converts Python objects to MCP protocol messages
# Handles request/response matching
# Manages message IDs and protocol compliance
```

#### Session Lifecycle:
```python
class MCPClient:
    def __init__(self):
        self.session = None  # Initially no session
    
    async def connect_to_server(self, server_script_path: str):
        # Create session with stdio transport
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(self.stdio, self.write)
        )
        
        # Session is now ready for communication
        await self.session.initialize()
    
    async def call_tool(self, tool_name: str, arguments: dict):
        # Use session to communicate with server
        result = await self.session.call_tool(tool_name, arguments)
        return result
    
    async def cleanup(self):
        # Session is automatically closed by exit_stack
        # self.session becomes invalid after this
        await self.exit_stack.aclose()
```

#### Session vs Transport:
- **Transport** (`stdio_client`): Low-level communication (stdin/stdout streams)
- **Session** (`ClientSession`): High-level protocol (MCP messages, tool calls)

```
┌─────────────────┐    MCP Protocol    ┌─────────────────┐
│   MCP Client    │ ←─────────────────→ │   MCP Server    │
│                 │                     │                 │
│  ClientSession  │    stdio streams    │  FastMCP        │
│       ↓         │ ←─────────────────→ │       ↑         │  
│  stdio_client   │                     │  main.py        │
└─────────────────┘                     └─────────────────┘
```

## Error Handling

The client implements robust error handling:

```python
async def main():
    client = MCPClient()
    try:
        await client.connect_to_server("/path/to/server.py")
        result = await client.call_tool("add", {"a": 5, "b": 7})
        print(f"Result: {result.content}")
    except Exception as e:
        print(f"Error occurred: {e}")
    finally:
        # This ALWAYS runs, even if errors occur
        await client.cleanup()  # Ensures no resource leaks
```

## Architecture Summary

```
┌──────────────────────────────────────────────────────────────┐
│                        MCP Client                            │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────────┐  │
│  │ AsyncExitStack │  │ ClientSession │  │   stdio_client    │  │
│  │               │  │              │  │                   │  │
│  │ • Resource    │  │ • MCP Protocol│  │ • Server Process  │  │
│  │   Management  │  │ • Tool Calls  │  │ • stdin/stdout    │  │
│  │ • Auto Cleanup│  │ • Serialization│  │ • Communication  │  │
│  └─────────────┘  └──────────────┘  └─────────────────────┘  │
│                                                              │
└──────────────────────────────────────────────────────────────┘
                              │
                              │ stdio transport
                              ▼
┌──────────────────────────────────────────────────────────────┐
│                      MCP Server Process                      │
│                        (main.py)                            │
└──────────────────────────────────────────────────────────────┘
```

This architecture ensures reliable, efficient, and maintainable communication between the MCP client and server processes.