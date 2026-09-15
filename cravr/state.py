"""State contract shared by every LangGraph node."""

from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages


class CravrState(TypedDict):
    """Conversation state; `add_messages` preserves the chat history."""

    messages: Annotated[list, add_messages]
    intent_profile: dict
    context_profile: dict  # ambient MCP facts (e.g. saved address), never user-facing requirements
    preference_profile: dict  # persistent dietary/budget/allergy prefs from the user profile MCP endpoint
    history_profile: dict
    user_id: str | None  # optional; no default identity is seeded -- Swiggy's OAuth token is the real one
    recommendation_state: dict
    summary: str
    # Frozen hand-off from decision_agent -> recommendation_agent:
    # {"filters": {...MCP args...}, "history": {...hints...}, "summary": "..."}.
    # recommendation_agent reads only this, never intent/history_profile directly.
    recommendation_brief: dict
    # decision_agent_node's verdict -- always ready_to_recommend=True/"recommend"
    # today (it never asks a clarifying question of its own), kept for visibility
    # in CRAVR_DEBUG output and as the hook for reintroducing a real verdict later.
    ready_to_recommend: bool
    decision: str
    # cart_agent_node's view of the current Swiggy cart -- {"items", "pricing",
    # "restaurant", "coupon_applied", "coupon_discount", "cart_id", ...}, the
    # same shape food_mcp.get_cart returns. Empty/absent until the user's
    # first cart-related message.
    cart_state: dict
