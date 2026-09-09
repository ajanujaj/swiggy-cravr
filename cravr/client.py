"""Shared MCP tool-calling helper. All agents call tools through
`call_tool`/`call_tool_sync` instead of building their own `fastmcp.Client`.

CRAVR_MCP_URL defaults to the local mock server; in production it should
point at Swiggy's Food MCP server (https://mcp.swiggy.com/food -- the only
one of Swiggy's three MCP servers Cravr targets). When a Swiggy OAuth
token exists (see mcp_server/oauth.py), it's attached as a Bearer token
automatically; otherwise calls stay unauthenticated, as they do today
against the mock server.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

from fastmcp import Client

DEFAULT_MCP_URL = os.getenv("CRAVR_MCP_URL", "http://127.0.0.1:8000/mcp")

# Separate from CRAVR_MCP_URL on purpose: user_preferences has no real Food
# MCP equivalent, so it must keep reaching the mock server even once
# CRAVR_MCP_URL is switched to Swiggy for everything else. Defaults to the
# same local mock URL; set explicitly on Render once cravr-mock-mcp is
# deployed (e.g. https://cravr-mock-mcp.onrender.com/mcp).
MOCK_MCP_URL = os.getenv("CRAVR_MOCK_MCP_URL", "http://127.0.0.1:8000/mcp")


def using_real_food_mcp() -> bool:
    """True once CRAVR_MCP_URL points at Swiggy's real Food MCP server --
    its tool names/schemas differ from the mock server, so agents branch
    on this to decide which tools to call. See cravr/food_mcp.py."""
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
    """Blocking wrapper for sync call sites (LangGraph node functions).
    Don't call from inside an already-running event loop -- use
    `call_tool` there instead."""
    return asyncio.run(call_tool(name, arguments, mcp_url=mcp_url))


async def main():
    result = await call_tool("order_history", {"user_id": "USR-001"})
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
