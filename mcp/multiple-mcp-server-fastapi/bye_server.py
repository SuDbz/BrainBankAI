
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("ByeServer",stateless_http=True)


@mcp.tool(description="Say bye to the user",name="echoBye")
def echoBye(userName: str) -> str:
    """Echo a greeting to the user."""
    return f"Bye, {userName}!"