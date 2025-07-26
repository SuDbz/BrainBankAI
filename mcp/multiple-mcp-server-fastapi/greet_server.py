
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("HaiServer",stateless_http=True)


@mcp.tool(description="Greet the user",name="echoHai")
def echoHai(userName: str) -> str:
    """Echo a greeting to the user."""
    return f"Hello, {userName}!"

  