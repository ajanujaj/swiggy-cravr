"""History discovery agent: fetches a user's order history from the MCP tool."""

from __future__ import annotations

import json
from typing import Any

from .. import food_mcp
from ..client import MOCK_MCP_URL, call_tool_sync, using_real_food_mcp
from ..debug_utils import debug
from ..profile_utils import merge_profile
from ..state import CravrState


def _call_mcp_tool(tool_name: str, user_id: str | None, *, mcp_url: str | None = None) -> dict[str, Any]:
    """Call one user-scoped MCP tool and normalize FastMCP's response."""
    try:
        result = call_tool_sync(tool_name, {"user_id": user_id}, mcp_url=mcp_url)
    except Exception as exc:  # pragma: no cover - exercised through runtime failures
        debug("preference_agent", f"MCP {tool_name} FAILED ({type(exc).__name__}: {exc})")
        return {"status": "error", "message": f"Unable to retrieve {tool_name}: {exc}"}

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

    return {"status": "error", "message": f"Unexpected {tool_name} response format"}


def fetch_order_history(address_id: str | None = None, *, user_id: str | None = None) -> dict[str, Any]:
    """Fetch a user's recent orders from MCP.

    `user_id` is keyword-only, used only by the mock `order_history` tool --
    Swiggy's real get_food_orders takes no user identifier at all (confirmed
    against its docs: session credentials are supplied automatically by the
    authenticated MCP session). `addressId` only says which saved address.

    `address_id` is usually None on the first call of a turn (this node runs
    before context_agent resolves it), so food_mcp.get_order_history() falls
    back to its own lookup; passed through when a caller already has it.
    """
    if using_real_food_mcp():
        return food_mcp.get_order_history(address_id=address_id)
    return _call_mcp_tool("order_history", user_id)

def fetch_user_preferences(user_id: str | None) -> dict[str, Any]:
    """Fetch a user's saved preference profile from MCP. No Swiggy Food MCP
    tool exposes this (checked all 20), so this always targets MOCK_MCP_URL
    explicitly regardless of what CRAVR_MCP_URL points at."""
    return _call_mcp_tool("user_preferences", user_id, mcp_url=MOCK_MCP_URL)


def history_discovery_node(state: CravrState) -> dict:
    """Fetch order history for the current user and store it in state."""
    user_id = state.get("user_id") or state.get("context_profile", {}).get("user_id")
    address_id = (state.get("context_profile") or {}).get("address_id")
    history_profile = fetch_order_history(address_id, user_id=user_id)
    debug(
        "history_agent",
        f"user_id={user_id!r} status={history_profile.get('status')!r} "
        f"orders={len(history_profile.get('orders') or [])}",
    )
    return {"history_profile": history_profile, "user_id": user_id }

def user_preference_node(state: CravrState) -> dict:
    """First workflow node: load saved profile facts and recent orders."""
    user_id = state.get("user_id") or state.get("context_profile", {}).get("user_id")
    context_profile = dict(state.get("context_profile") or {})
    address_id = context_profile.get("address_id")

    # Runs before context_discovery_node, so address_id is usually unresolved but get_food_orders
    # needs one. Resolve it once (real Food MCP only) and forward via context_profile so
    # context_discovery_node reuses it instead of calling get_addresses again.
    context_update: dict[str, Any] = {}
    if using_real_food_mcp() and not address_id:
        location_result = food_mcp.get_saved_address()
        address_id = location_result.get("address_id")
        if location_result.get("status") == "success":
            context_update = {
                key: location_result[key]
                for key in ("location", "delivery_area", "address_id")
                if location_result.get(key)
            }

    preference_profile = fetch_user_preferences(user_id)
    history_profile = fetch_order_history(address_id, user_id=user_id)
    debug(
        "preference_agent",
        f"user_id={user_id!r} status={preference_profile.get('status')!r} "
        f"preferences={bool(preference_profile.get('preferences'))} "
        f"history_status={history_profile.get('status')!r} "
        f"orders={len(history_profile.get('orders') or [])}",
    )
    result: dict[str, Any] = {
        "preference_profile": preference_profile,
        "history_profile": history_profile,
        "user_id": user_id,
    }
    if context_update:
        result["context_profile"] = merge_profile(context_profile, context_update)
    return result


