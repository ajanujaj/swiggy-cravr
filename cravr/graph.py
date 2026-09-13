"""Compiled LangGraph workflow for the Cravr chat agent."""

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from .context import context_discovery_node
from .decision import CLARIFY_AND_RETRY, decision_agent_node
from .userPreference import user_preference_node
from .intent import intent_discovery_node
from .profile_utils import latest_user_text
from .recommend import is_pure_feedback, recommendation_agent_node
from .state import CravrState


# These double as conditional-edge labels in the generated Mermaid diagram, so they're named
# for *why* the branch was taken, not the node it lands on (an unnamed key draws an unlabelled edge).
FEEDBACK = "feedback"
NEW_REQUEST = "new_request"
RECOMMEND = "recommend"
NEEDS_MORE_INFO = "needs_more_info"


def _route_entry(state: CravrState) -> str:
    """Send replies about the current suggestions straight to the recommender --
    once dishes are on the table, "3" or "yes" is a reaction, not a new request,
    and running it through intent/history/decision wastes 3 calls on a bare "3".
    Anything that changes the request ("make it non-veg") still takes the full path.
    """
    if is_pure_feedback(latest_user_text(state.get("messages") or []), state.get("recommendation_state")):
        return FEEDBACK
    return NEW_REQUEST


def _route_after_decision(state: CravrState) -> str:
    if state.get("ready_to_recommend"):
        return RECOMMEND
    if state.get("decision") == CLARIFY_AND_RETRY:
        return CLARIFY_AND_RETRY
    return NEEDS_MORE_INFO


def build_graph():
    workflow = StateGraph(CravrState)

    workflow.add_node("user_preference", user_preference_node)
    workflow.add_node("context_discovery", context_discovery_node)
    workflow.add_node("intent_discovery", intent_discovery_node)
    workflow.add_node("decision_agent", decision_agent_node)
    workflow.add_node("recommendation_agent", recommendation_agent_node)

    workflow.add_conditional_edges(
        START,
        _route_entry,
        {FEEDBACK: "recommendation_agent", NEW_REQUEST: "user_preference"},
    )
    # New requests load the durable profile/order history, then delivery context, then intent.
    workflow.add_edge("user_preference", "context_discovery")
    workflow.add_edge("context_discovery", "intent_discovery")
    workflow.add_edge("intent_discovery", "decision_agent")
    workflow.add_conditional_edges(
        "decision_agent",
        _route_after_decision,
        {
            RECOMMEND: "recommendation_agent",
            CLARIFY_AND_RETRY: "intent_discovery",
            NEEDS_MORE_INFO: END,
        },
    )
    workflow.add_edge("recommendation_agent", END)

    # A checkpointer is required so intent_discovery_node can pause mid-graph (via `interrupt()`)
    # to ask a clarifying question, then resume with just the answer instead of restarting.
    return workflow.compile(checkpointer=MemorySaver())


app = build_graph()
