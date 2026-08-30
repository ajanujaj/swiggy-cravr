"""Context Discovery agent: turns a food-ordering request into structured context."""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any

from ..debug_utils import debug
from ..profile_utils import merge_profile
from ..state import CravrState

DEFAULT_MCP_URL = os.getenv("CRAVR_MCP_URL", "http://127.0.0.1:8000/mcp")


def fetch_user_location(user_id: str | None) -> dict[str, Any]:
    """Fetch saved delivery location without asking the user for it."""
    async def _call_tool() -> Any:
        from fastmcp import Client

        async with Client(DEFAULT_MCP_URL) as client:
            return await client.call_tool(name="user_location", arguments={"user_id": user_id})

    try:
        result = asyncio.run(_call_tool())
    except Exception as exc:  # pragma: no cover - runtime/network failure
        debug("context_agent", f"MCP user_location FAILED ({type(exc).__name__}: {exc})")
        return {"status": "error", "message": f"Unable to retrieve saved location: {exc}"}

    if isinstance(result, dict):
        return result
    content = getattr(result, "content", None)
    if isinstance(content, list):
        for item in content:
            if isinstance(getattr(item, "text", None), str):
                try:
                    return json.loads(item.text)
                except json.JSONDecodeError:
                    break
    return {"status": "error", "message": "Unexpected user_location response format"}


def context_discovery_node(state: CravrState) -> dict:
    """Second workflow node: enrich state with saved delivery context.

    Context is ambient. It must not call an LLM, fabricate live conditions,
    or create user-facing clarification questions when location data is
    unavailable.
    """
    user_id = state.get("user_id")
    location_result = fetch_user_location(user_id)
    extracted = {
        key: location_result[key]
        for key in ("location", "delivery_area")
        if location_result.get(key)
    }

    profile = merge_profile(state.get("context_profile"), extracted)
    profile["important_missing_information"] = []
    profile["clarification_questions"] = []

    debug(
        "context_agent",
        f"user_id={user_id!r} location_status={location_result.get('status')!r} "
        f"profile={profile}",
    )

    return {"context_profile": profile}
