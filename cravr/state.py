"""State contract shared by every LangGraph node."""

from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages


class CravrState(TypedDict):
    """Conversation state; `add_messages` preserves the chat history."""

    messages: Annotated[list, add_messages]
    intent_profile: dict
    # context_profile: dict | None
    history_profile: dict
    user_id: str
    recommendation_state: dict
    summary: str
    # The single, frozen hand-off from decision_agent to recommendation_agent,
    # built once at the moment the decision becomes "recommend":
    #   {"filters": {...MCP args...}, "history": {...hints...}, "summary": "..."}
    # recommendation_agent reads ONLY this -- it never reaches back into
    # intent_profile/history_profile, so there is exactly one place where a
    # request is turned into menu filters.
    recommendation_brief: dict
    # How many clarification round-trips decision_agent_node has asked for in
    # the current gap-filling streak; reset to 0 by intent_discovery_node at
    # the start of every *fresh* user turn (not on a clarify-and-retry loop).
    clarification_rounds: int
    # The profile field decision_agent_node just asked about, e.g.
    # "craving_or_cuisine". intent_discovery_node consumes this to apply the
    # user's answer deterministically instead of relying on the (small,
    # unreliable) extraction model to infer which question a bare reply like
    # "Indian" or "1000" was answering. Cleared once consumed.
    pending_clarification_field: str
    # Fields already asked about in this streak. A field is never asked twice:
    # if the answer couldn't be captured, we move on rather than badgering the
    # user with the same question until MAX_CLARIFICATION_ROUNDS runs out.
    asked_clarification_fields: list
    # decision_agent_node's verdict, read by graph.py's conditional edge to
    # route to recommendation_agent / back to intent_discovery / END. These
    # must be declared here -- LangGraph only reliably threads state through
    # its schema-declared channels, and routing on an undeclared key is silent
    # until it isn't.
    ready_to_recommend: bool
    decision: str
