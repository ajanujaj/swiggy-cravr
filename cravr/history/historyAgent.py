"""History discovery agent: fetches a user's order history from the MCP tool."""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any

from fastmcp import Client

from ..debug_utils import debug
from ..state import CravrState


DEFAULT_MCP_URL = os.getenv("CRAVR_MCP_URL", "http://127.0.0.1:8000/mcp")


def fetch_order_history(user_id: str | None) -> dict[str, Any]:
    """Call the MCP order_history tool and normalize the response."""
    user_id = user_id or os.getenv("CRAVR_USER_ID") or "USR-001"
    target_url = DEFAULT_MCP_URL

    async def _call_tool() -> Any:
        async with Client(target_url) as client:
            return await client.call_tool(name="order_history", arguments={"user_id": user_id})

    try:
        result = asyncio.run(_call_tool())
    except Exception as exc:  # pragma: no cover - exercised through runtime failures
        debug("history_agent", f"MCP call FAILED ({type(exc).__name__}: {exc}) at {target_url}")
        return {"status": "error", "message": f"Unable to retrieve order history: {exc}"}

    if isinstance(result, dict):
        return result

    if hasattr(result, "content"):
        content = getattr(result, "content", None)
        if isinstance(content, list):
            for item in content:
                item_text = getattr(item, "text", None)
                if isinstance(item_text, str):
                    try:
                        return json.loads(item_text)
                    except json.JSONDecodeError:
                        return {"status": "success", "message": item_text}
        if isinstance(content, str):
            return {"status": "success", "message": content}

    return {"status": "error", "message": "Unexpected order history response format"}


def history_discovery_node(state: CravrState) -> dict:
    """Fetch order history for the current user and store it in state."""
    user_id = state.get("user_id") or state.get("context_profile", {}).get("user_id")
    history_profile = fetch_order_history(user_id)
    debug(
        "history_agent",
        f"user_id={user_id!r} status={history_profile.get('status')!r} "
        f"orders={len(history_profile.get('orders') or [])}",
    )
    return {"history_profile": history_profile, "user_id": user_id or "USR-001"}
