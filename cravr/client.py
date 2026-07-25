import asyncio
import os

from fastmcp import Client

DEFAULT_MCP_URL = os.getenv("CRAVR_MCP_URL", "http://127.0.0.1:8000/mcp")


async def main():
    async with Client(DEFAULT_MCP_URL) as client:
        result = await client.call_tool(
            name="order_history",
            arguments={"user_id": "USR-001"},
        )
    print(result)


if __name__ == "__main__":
    asyncio.run(main())