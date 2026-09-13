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
    # Clarification round-trips in the current gap-filling streak; reset by
    # intent_discovery_node at the start of each fresh turn.
    clarification_rounds: int
    # Field decision_agent_node just asked about (e.g. "craving_or_cuisine");
    # intent_discovery_node uses this to apply the answer deterministically
    # instead of trusting extraction to guess which question was answered.
    pending_clarification_field: str
    # Free-text question decision_agent_node wants asked with tappable options
    # when it doesn't map to one of intent_discovery_node's fixed fields.
    # decision_agent_node never calls interrupt() itself -- only
    # intent_discovery_node does (one interrupt site, one round budget) -- so
    # it hands the question off here instead.
    pending_decision_question: str
    asked_clarification_fields: list  # fields already asked this streak; never re-asked
    # decision_agent_node's verdict; graph.py's conditional edge routes on
    # these. Must stay declared here -- LangGraph only threads state through
    # schema-declared channels, and routing on an undeclared key fails silently.
    ready_to_recommend: bool
    decision: str
