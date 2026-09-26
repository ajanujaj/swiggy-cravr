from __future__ import annotations

import asyncio
import os
from typing import Any

from fastmcp import Client

DEFAULT_MCP_URL = os.getenv("CRAVR_MCP_URL", "http://127.0.0.1:8000/mcp")

MOCK_MCP_URL = os.getenv("CRAVR_MOCK_MCP_URL", "http://127.0.0.1:8000/mcp")


def using_real_food_mcp() -> bool:
    return "mcp.swiggy.com" in DEFAULT_MCP_URL


def _bearer_token() -> str | None:
    try:
        from mcp_server.oauth import load_tokens
    except ImportError:
        return None

    tokens = load_tokens()
    return tokens.get("access_token") if tokens else None


async def call_tool(name: str, arguments: dict[str, Any], *, mcp_url: str | None = None) -> Any:
    url = mcp_url or DEFAULT_MCP_URL
    async with Client(url, auth=_bearer_token()) as client:
        return await client.call_tool(name=name, arguments=arguments)


def call_tool_sync(name: str, arguments: dict[str, Any], *, mcp_url: str | None = None) -> Any:
    return asyncio.run(call_tool(name, arguments, mcp_url=mcp_url))


async def main():
    result = await call_tool("order_history", {"user_id": "USR-001"})
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
