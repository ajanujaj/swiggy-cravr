"""State contract shared by every LangGraph node."""

from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages


class CravrState(TypedDict):
    """Conversation state; `add_messages` preserves the chat history."""

    messages: Annotated[list, add_messages]
    intent_profile: dict
    context_profile: dict
    history_profile: dict
    user_id: str
    recommendation_state: dict
    summary: str
