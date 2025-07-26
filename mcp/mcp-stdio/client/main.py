import asyncio
from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client



async def main():
    """
    Main function demonstrating MCP client usage.
    
    This function shows the complete lifecycle of an MCP client:
    1. Create a client instance
    2. Connect to the MCP server using stdio transport
    3. Call tools/functions on the server
    4. Handle results and display output
    5. Clean up resources properly
    
    The try/finally pattern ensures that even if an error occurs during
    tool execution, the cleanup method is called to prevent resource leaks.
    
    Example workflow:
    - Server is spawned as a subprocess
    - Client establishes MCP protocol communication
    - Tools are discovered and called
    - Server subprocess is terminated cleanly
    """
    # Create the MCP client instance
    client = MCPClient()
    
    try:
        # Connect to the MCP server
        # This will spawn the server process and establish communication
        await client.connect_to_server("<path>/mcp-server/main.py")
        
        # Call the 'add' tool with first set of numbers
        # Demonstrates basic tool calling with integer parameters
        print("\nCalling add tool with 5 + 7:")
        result = await client.call_tool("add", {"a": 5, "b": 7})
        print(f"Result: {result.content}")
        
        # Call the 'add' tool with different numbers
        # Shows that the server maintains state and can handle multiple calls
        print("\nCalling add tool with 10 + 15:")
        result = await client.call_tool("add", {"a": 10, "b": 15})
        print(f"Result: {result.content}")
        
    finally:
        # Always clean up resources, even if an exception occurred
        # This ensures the server subprocess is terminated and streams are closed
        await client.cleanup()


class MCPClient:
    """
    MCP (Model Context Protocol) Client for stdio transport communication.
    
    This client establishes a connection to an MCP server using standard input/output
    streams, allowing for seamless communication between the client and server processes.
    """
    
    def __init__(self):
        """
        Initialize the MCP client.
        
        Sets up the essential components:
        - session: Will hold the ClientSession for MCP communication
        - exit_stack: AsyncExitStack for proper resource management and cleanup
        """
        self.session = None
        self.exit_stack = AsyncExitStack()

    async def connect_to_server(self, server_script_path: str):
        """
        Connect to an MCP server using stdio transport.
        
        This method:
        1. Sets up stdio transport parameters (command, args, environment)
        2. Creates a subprocess running the MCP server
        3. Establishes bidirectional communication via stdin/stdout
        4. Initializes the MCP session and protocol handshake
        5. Lists available tools from the server
        
        Args:
            server_script_path: Absolute path to the MCP server Python script
        
        The stdio transport is ideal for local development and testing as it:
        - Automatically manages the server subprocess lifecycle
        - Uses standard input/output streams for communication
        - Doesn't require separate server startup or network configuration
        """
        # Set up Stdio transport parameters
        # This tells the client how to spawn and communicate with the server process
        server_params = StdioServerParameters(
            command="python",  # Command to run the server (could be "node" for JS servers)
            args=[server_script_path],  # Arguments passed to the command
            env=None  # Environment variables (None = inherit from parent process)
        )
        
        # Establish stdio transport
        # This creates a subprocess running the MCP server and sets up communication pipes
        # The exit_stack ensures proper cleanup of the subprocess when done
        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        # stdio_transport returns a tuple: (read_stream, write_stream)
        self.stdio, self.write = stdio_transport
        
        # Create session
        # ClientSession handles the MCP protocol layer on top of the transport
        # It manages message serialization, request/response matching, and protocol compliance
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(self.stdio, self.write)
        )
        
        # Initialize the MCP session
        # This performs the initial protocol handshake with the server
        # Exchanges capabilities, protocol version, and establishes the connection
        await self.session.initialize()
        
        # List available tools
        # Query the server for all available tools/functions it can execute
        # This is typically done after connection to discover server capabilities
        response = await self.session.list_tools()
        tools = response.tools
        print(f"Connected to server with tools: {[tool.name for tool in tools]}")

    async def call_tool(self, tool_name: str, arguments: dict):
        """
        Call a specific tool on the connected MCP server.
        
        This method sends a tool execution request to the server and waits for the result.
        The MCP protocol handles:
        - Message serialization (converting Python objects to MCP messages)
        - Request/response matching (ensuring responses match requests)
        - Error handling and validation
        
        Args:
            tool_name: Name of the tool to execute (must match server's available tools)
            arguments: Dictionary of arguments to pass to the tool
            
        Returns:
            The result object from the server, containing the tool's output
            
        Example:
            result = await client.call_tool("add", {"a": 5, "b": 7})
            print(result.content)  # Should print the sum: 12
        """
        result = await self.session.call_tool(tool_name, arguments)
        return result

    async def cleanup(self):
        """
        Clean up all resources and close connections.
        
        This method:
        1. Closes the MCP session (stops communication with server)
        2. Terminates the server subprocess
        3. Closes all file handles and streams
        4. Releases any other allocated resources
        
        The AsyncExitStack automatically handles the cleanup of all context managers
        that were registered during the connection process. This ensures no resource
        leaks and proper shutdown of the server process.
        
        Should always be called when done with the client, typically in a try/finally block.
        """
        await self.exit_stack.aclose()

if __name__ == "__main__":
    asyncio.run(main())
