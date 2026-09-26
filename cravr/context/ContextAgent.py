from __future__ import annotations

import json
from typing import Any

from .. import food_mcp
from ..client import call_tool_sync, using_real_food_mcp
from ..debug_utils import debug
from ..profile_utils import merge_profile
from ..state import CravrState


def fetch_user_location(user_id: str | None) -> dict[str, Any]:
    if using_real_food_mcp():
        return food_mcp.get_saved_address()

    try:
        result = call_tool_sync("user_location", {"user_id": user_id})
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
    user_id = state.get("user_id")
    existing_profile = state.get("context_profile") or {}

    if existing_profile.get("address_id"):
        location_result = {"status": "success", **existing_profile}
    else:
        location_result = fetch_user_location(user_id)

    extracted = {
        key: location_result[key]
        for key in ("location", "delivery_area", "address_id")
        if location_result.get(key)
    }

    profile = merge_profile(existing_profile, extracted)
    profile["important_missing_information"] = []
    profile["clarification_questions"] = []

    debug(
        "context_agent",
        f"user_id={user_id!r} location_status={location_result.get('status')!r} "
        f"profile={profile}",
    )

    return {"context_profile": profile}
