"""
Path: perception/mcp_client.py
Role: Model-Context-Protocol connector for using external tools.
"""
import httpx
from loguru import logger

# Mock implementation of MCP. In a real scenario, this would be a dedicated library.
class MockMCP_Session:
    def __init__(self, tools, api_keys):
        self.tools = {tool.split('/')[-1]: tool for tool in tools}
        self.api_keys = api_keys
        self.firecrawl_client = httpx.AsyncClient(
            base_url="https://api.firecrawl.dev",
            headers={"Authorization": f"Bearer {self.api_keys.get('FIRECRAWL_API_KEY')}"}
        )

    async def call_tool(self, tool_name: str, params: dict) -> str:
        logger.info(f"MCPClient calling tool '{tool_name}' with params: {params}")
        
        if tool_name == "browse" and "url" in params:
            try:
                response = await self.firecrawl_client.post("/v0/scrape", json={"url": params["url"]})
                response.raise_for_status()
                # Assuming Firecrawl returns Markdown directly
                return response.json().get("data", {}).get("markdown", "")
            except httpx.HTTPStatusError as e:
                logger.error(f"Firecrawl API error: {e.response.text}")
                return f"Error: Failed to browse URL {params['url']}."
            except Exception as e:
                logger.error(f"An unexpected error occurred with Firecrawl: {e}")
                return f"Error: Could not access URL {params['url']}."

        # Add more tools like 'shell', 'wolfram', etc. here
        elif tool_name == "shell":
            return "Shell tool is not implemented in this mock."
        elif tool_name == "wolfram":
            return "Wolfram tool is not implemented in this mock."
        else:
            logger.warning(f"Attempted to call unknown or unsupported tool: {tool_name}")
            return f"Error: Tool '{tool_name}' not found."

    async def close(self):
        await self.firecrawl_client.aclose()


def initialize(tools: list, api_keys: dict):
    return MockMCP_Session(tools, api_keys)

class MCPClient:
    def __init__(self, api_keys: dict):
        """
        Initializes the MCPClient with necessary API keys.
        Args:
            api_keys (dict): Dictionary containing API keys for tools (e.g., {'FIRECRAWL_API_KEY': '...'})
        """
        self.session = initialize(["mcp/browser", "mcp/shell", "mcp/wolfram"], api_keys)

    async def call(self, tool_name: str, params: dict) -> str:
        """
        Calls an external tool via the MCP session.
        Args:
            tool_name (str): The name of the tool to call (e.g., "browse").
            params (dict): The parameters for the tool call.
        Returns:
            str: The result from the tool.
        """
        return await self.session.call_tool(tool_name, params)
    
    async def close(self):
        """Closes the underlying session and client."""
        if hasattr(self.session, 'close'):
            await self.session.close()

