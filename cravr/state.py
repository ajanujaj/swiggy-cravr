from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages


class CravrState(TypedDict):
    messages: Annotated[list, add_messages]
    intent_profile: dict
    context_profile: dict
    preference_profile: dict
    history_profile: dict
    user_id: str | None
    recommendation_state: dict
    summary: str
    recommendation_brief: dict
    ready_to_recommend: bool
    decision: str
    cart_state: dict
