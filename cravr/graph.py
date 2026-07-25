"""Compiled LangGraph workflow for the Cravr chat agent."""

from langgraph.graph import END, START, StateGraph

from .intentAgent import intent_discovery_node
from .ContextAgent import context_discovery_node
from .historyAgent import history_discovery_node
from .decisionAgent import decision_agent_node
from .RecommendAgent import recommendation_agent_node
# from ..cravrbacklog.orchestrator import orchestrator_node
from .state import CravrState


def build_graph():
    workflow = StateGraph(CravrState)

    workflow.add_node("intent_discovery", intent_discovery_node)
    workflow.add_node("context_discovery", context_discovery_node)
    workflow.add_node("history_discovery", history_discovery_node)
    workflow.add_node("decision_agent", decision_agent_node)
    workflow.add_node("recommendation_agent", recommendation_agent_node)

    workflow.add_edge(START, "intent_discovery")
    workflow.add_edge("intent_discovery", "context_discovery")
    workflow.add_edge("context_discovery", "history_discovery")
    workflow.add_edge("history_discovery", "decision_agent")
    workflow.add_conditional_edges(
        "decision_agent",
        lambda state: "recommendation_agent" if state.get("ready_to_recommend") else END,
        {"recommendation_agent": "recommendation_agent", END: END},
    )
    workflow.add_edge("recommendation_agent", END)
    return workflow.compile()


app = build_graph()
