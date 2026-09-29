from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from .cart import cart_agent_node, is_cart_request
from .context import context_discovery_node
from .group import group_agent_node, is_group_request
from .decision import decision_agent_node
from .userPreference import user_preference_node
from .intent import intent_discovery_node
from .profile_utils import latest_user_text
from .recommend import is_pure_feedback, recommendation_agent_node
from .state import CravrState


FEEDBACK = "feedback"
NEW_REQUEST = "new_request"
CART = "cart"
GROUP = "group"


def _route_entry(state: CravrState) -> str:
    text = latest_user_text(state.get("messages") or [])
    if is_group_request(text, state.get("group_state")):
        return GROUP
    if is_cart_request(text):
        return CART
    if is_pure_feedback(text, state.get("recommendation_state")):
        return FEEDBACK
    return NEW_REQUEST


def build_graph():
    workflow = StateGraph(CravrState)

    workflow.add_node("user_preference", user_preference_node)
    workflow.add_node("context_discovery", context_discovery_node)
    workflow.add_node("intent_discovery", intent_discovery_node)
    workflow.add_node("decision_agent", decision_agent_node)
    workflow.add_node("recommendation_agent", recommendation_agent_node)
    workflow.add_node("cart_agent", cart_agent_node)
    workflow.add_node("group_agent", group_agent_node)

    workflow.add_conditional_edges(
        START,
        _route_entry,
        {FEEDBACK: "recommendation_agent", NEW_REQUEST: "user_preference", CART: "cart_agent",
         GROUP: "group_agent"},
    )
    workflow.add_edge("user_preference", "context_discovery")
    workflow.add_edge("context_discovery", "intent_discovery")
    workflow.add_edge("intent_discovery", "decision_agent")
    workflow.add_edge("decision_agent", "recommendation_agent")
    workflow.add_edge("recommendation_agent", END)
    workflow.add_edge("cart_agent", END)
    workflow.add_edge("group_agent", END)

    return workflow.compile(checkpointer=MemorySaver())


app = build_graph()
