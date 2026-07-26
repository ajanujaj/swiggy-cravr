"""Compiled LangGraph workflow for the Cravr chat agent."""

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from .intentAgent import intent_discovery_node
from .ContextAgent import context_discovery_node
from .historyAgent import history_discovery_node
from .decisionAgent import CLARIFY_AND_RETRY, decision_agent_node
from .profile_utils import latest_user_text
from .RecommendAgent import is_pure_feedback, recommendation_agent_node
# from ..cravrbacklog.orchestrator import orchestrator_node
from .state import CravrState


def _route_entry(state: CravrState) -> str:
    """Send replies about the current suggestions straight to the recommender.

    Once dishes are on the table, "3" or "yes" is a reaction, not a new
    request. Running it through intent/history/decision costs an extraction
    call, an MCP round-trip and a decision call for nothing -- and worse,
    hands the extraction model a bare "3" to interpret.

    Anything that changes the request ("make it non-veg", "under 200") still
    takes the full path so the profile is actually updated.
    """
    if is_pure_feedback(latest_user_text(state.get("messages") or []), state.get("recommendation_state")):
        return "recommendation_agent"
    return "intent_discovery"


def _route_after_decision(state: CravrState) -> str:
    if state.get("ready_to_recommend"):
        return "recommendation_agent"
    if state.get("decision") == CLARIFY_AND_RETRY:
        # A clarification question was just answered (via interrupt/resume) —
        # loop back through intent/context discovery so the answer is folded
        # into the profiles by those nodes before deciding again.
        return "intent_discovery"
    return END


def build_graph():
    workflow = StateGraph(CravrState)

    workflow.add_node("intent_discovery", intent_discovery_node)
    # workflow.add_node("context_discovery", context_discovery_node)
    workflow.add_node("history_discovery", history_discovery_node)
    workflow.add_node("decision_agent", decision_agent_node)
    workflow.add_node("recommendation_agent", recommendation_agent_node)

    workflow.add_conditional_edges(
        START,
        _route_entry,
        {"intent_discovery": "intent_discovery", "recommendation_agent": "recommendation_agent"},
    )
    workflow.add_edge("intent_discovery", "history_discovery")
    # workflow.add_edge("context_discovery", "history_discovery")
    workflow.add_edge("history_discovery", "decision_agent")
    workflow.add_conditional_edges(
        "decision_agent",
        _route_after_decision,
        {"recommendation_agent": "recommendation_agent", "intent_discovery": "intent_discovery", END: END},
    )
    workflow.add_edge("recommendation_agent", END)
    # A checkpointer is required so decision_agent_node can pause mid-graph
    # (via `interrupt()`) to ask the user a clarifying question, then resume
    # with just their answer instead of restarting the whole pipeline.
    return workflow.compile(checkpointer=MemorySaver())


app = build_graph()
