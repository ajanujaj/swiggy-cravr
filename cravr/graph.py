"""Compiled LangGraph workflow for the Cravr chat agent."""

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from .cart import cart_agent_node, is_cart_request
from .context import context_discovery_node
from .decision import decision_agent_node
from .userPreference import user_preference_node
from .intent import intent_discovery_node
from .profile_utils import latest_user_text
from .recommend import is_pure_feedback, recommendation_agent_node
from .state import CravrState


# Doubles as the conditional-edge label in the generated Mermaid diagram, so it's
# named for *why* the branch was taken, not the node it lands on.
FEEDBACK = "feedback"
NEW_REQUEST = "new_request"
CART = "cart"


def _route_entry(state: CravrState) -> str:
    """Send replies about the current suggestions straight to the recommender --
    once dishes are on the table, "3" or "yes" is a reaction, not a new request,
    and running it through intent/history/decision wastes 3 calls on a bare "3".
    Anything that changes the request ("make it non-veg") still takes the full path.

    Cart/coupon messages ("add it to cart", "apply SAVE20") are checked first --
    they can arrive right after a dish is picked (a `satisfied=True` terminal
    state that `is_pure_feedback` no longer treats as feedback) or at any other
    point in the conversation, so this can't just be another feedback branch.
    """
    text = latest_user_text(state.get("messages") or [])
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

    workflow.add_conditional_edges(
        START,
        _route_entry,
        {FEEDBACK: "recommendation_agent", NEW_REQUEST: "user_preference", CART: "cart_agent"},
    )
    # New requests load the durable profile/order history, then delivery context, then intent.
    workflow.add_edge("user_preference", "context_discovery")
    workflow.add_edge("context_discovery", "intent_discovery")
    workflow.add_edge("intent_discovery", "decision_agent")
    # decision_agent always recommends now (see decisionAgent.py) -- one edge, not a branch.
    workflow.add_edge("decision_agent", "recommendation_agent")
    workflow.add_edge("recommendation_agent", END)
    workflow.add_edge("cart_agent", END)

    # A checkpointer is required so intent_discovery_node can pause mid-graph (via `interrupt()`)
    # to ask a clarifying question, then resume with just the answer instead of restarting.
    return workflow.compile(checkpointer=MemorySaver())


app = build_graph()
