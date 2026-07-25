"""Recommendation agent: uses the MCP recommendation tool and a human-in-the-loop feedback cycle."""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any

from fastmcp import Client
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from .llm import get_llm
from .state import CravrState


DEFAULT_MCP_URL = os.getenv("CRAVR_MCP_URL", "http://127.0.0.1:8000/mcp")


def _coerce_profile(profile: Any) -> dict:
    if isinstance(profile, dict):
        return profile
    return {}


def _extract_latest_user_input(messages: Any) -> str:
    if not messages:
        return ""

    for message in reversed(list(messages)):
        if isinstance(message, dict):
            role = message.get("role")
            content = message.get("content")
            if role == "user" and content:
                return str(content)
            if role == "human" and content:
                return str(content)
        if hasattr(message, "type") and getattr(message, "type") in {"human", "user"}:
            content = getattr(message, "content", "")
            if content:
                return str(content)
        if hasattr(message, "role") and getattr(message, "role") == "user":
            content = getattr(message, "content", "")
            if content:
                return str(content)

    last_message = messages[-1]
    content = getattr(last_message, "content", last_message)
    return str(content)


def _looks_like_acceptance(user_input: str) -> bool:
    normalized = (user_input or "").strip().lower()
    return any(token in normalized for token in ["yes", "yep", "sounds good", "good", "okay", "perfect", "love it", "go ahead", "that works", "choose it", "accept"])


def _looks_like_rejection(user_input: str) -> bool:
    normalized = (user_input or "").strip().lower()
    return any(token in normalized for token in ["no", "not that", "different", "another", "change", "new one", "something else", "skip", "reject"])


def _build_recommendation_text(recommendations: list[dict[str, Any]]) -> str:
    if not recommendations:
        return "I could not find a matching food suggestion right now. I can try a different cuisine or budget if you want."

    top = recommendations[0]
    item_name = top.get("item_name") or "a dish"
    restaurant_name = top.get("restaurant_name") or "the restaurant"
    cuisine = top.get("cuisine") or ""
    price = top.get("price_inr")
    rating = top.get("rating")
    tags = top.get("tags") or ""
    details = []
    if cuisine:
        details.append(cuisine)
    if price is not None:
        details.append(f"₹{price}")
    if rating is not None:
        details.append(f"rating {rating}")
    detail_text = ", ".join(details) if details else ""
    return (
        f"I’d suggest {item_name} from {restaurant_name}"
        f"{f' ({detail_text})' if detail_text else ''}."
        f" {f'It also fits tags like {tags}.' if tags else ''}"
        f" Would you like me to go with this suggestion or try a different option?"
    )


def fetch_recommendations(summary: str, intent_profile: dict[str, Any], context_profile: dict[str, Any], history_profile: dict[str, Any], user_id: str) -> list[dict[str, Any]]:
    """Call the MCP recommendation tool with the best available filters derived from the summary."""
    filters: dict[str, Any] = {}

    if summary:
        filters["summary"] = summary

    craving = intent_profile.get("craving") or intent_profile.get("cuisine")
    if craving:
        filters["tags"] = str(craving)

    cuisine = intent_profile.get("cuisine")
    if cuisine:
        filters["cuisine"] = cuisine

    dietary_preferences = intent_profile.get("dietary_preferences") or []
    if dietary_preferences:
        filters["dietary_type"] = dietary_preferences[0]

    budget = intent_profile.get("budget_inr")
    if budget is not None:
        filters["max_price_inr"] = int(budget)

    async def _call_tool() -> Any:
        async with Client(DEFAULT_MCP_URL) as client:
            return await client.call_tool(name="recommendation", arguments=filters)

    try:
        result = asyncio.run(_call_tool())
    except Exception:
        return []

    if isinstance(result, dict) and result.get("status") == "success":
        return result.get("recommendations") or []

    if hasattr(result, "content"):
        content = getattr(result, "content", None)
        if isinstance(content, list):
            for item in content:
                item_text = getattr(item, "text", None)
                if isinstance(item_text, str):
                    try:
                        payload = json.loads(item_text)
                        if isinstance(payload, dict) and payload.get("status") == "success":
                            return payload.get("recommendations") or []
                    except json.JSONDecodeError:
                        continue

    return []


def recommendation_agent_node(state: CravrState) -> dict:
    """Recommend food suggestions via MCP and keep looping until the user is satisfied."""
    intent_profile = _coerce_profile(state.get("intent_profile"))
    context_profile = _coerce_profile(state.get("context_profile"))
    history_profile = _coerce_profile(state.get("history_profile"))
    summary = str(state.get("summary") or "")
    user_id = state.get("user_id") or "USR-001"

    recommendation_state = dict(state.get("recommendation_state") or {})
    awaiting_feedback = bool(recommendation_state.get("awaiting_feedback"))
    last_recommendations = recommendation_state.get("last_recommendations") or []
    satisfied = bool(recommendation_state.get("satisfied"))

    user_input = _extract_latest_user_input(state.get("messages") or [])
    if awaiting_feedback and not satisfied:
        if _looks_like_acceptance(user_input):
            return {
                "recommendation_state": {"awaiting_feedback": False, "satisfied": True, "last_recommendations": last_recommendations},
                "messages": [AIMessage(content="Perfect — I’ll proceed with that suggestion.")],
            }
        if _looks_like_rejection(user_input):
            return {
                "recommendation_state": {"awaiting_feedback": False, "satisfied": False, "last_recommendations": last_recommendations},
                "messages": [AIMessage(content="No problem — I can suggest a different option. Tell me what you want to change, such as cuisine, budget, or dietary preference.")],
            }

    recommendations = fetch_recommendations(summary, intent_profile, context_profile, history_profile, user_id)
    if not recommendations:
        return {
            "recommendation_state": {"awaiting_feedback": False, "satisfied": False, "last_recommendations": []},
            "messages": [AIMessage(content="I’m not finding a strong match right now. Tell me your cuisine, budget, or dietary preference and I’ll refine it.")],
        }

    next_state = {
        "awaiting_feedback": True,
        "satisfied": False,
        "last_recommendations": recommendations,
    }
    message_text = _build_recommendation_text(recommendations)
    return {
        "recommendation_state": next_state,
        "messages": [AIMessage(content=message_text)],
    }
