"""Decision agent: freezes the recommendation brief and always recommends.
Never calls `interrupt()` -- only intent_discovery_node does (see
`DecisionAgentNeverAsksTests`), so there's exactly one interrupt site in the
whole graph.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage

from ..debug_utils import debug
from ..profile_utils import latest_user_text
from ..recommend import build_recommendation_brief
from ..state import CravrState


def _coerce_profile(profile: Any) -> dict:
    if isinstance(profile, dict):
        return profile
    return {}


def _extract_latest_user_input(messages: Any) -> str:
    return latest_user_text(messages)


def decision_agent_node(state: CravrState) -> dict:
    """Always recommends -- no clarifying question of its own. Still the
    only place that freezes `recommendation_brief`, so recommendation_agent
    has something to read."""
    intent_profile = _coerce_profile(state.get("intent_profile"))
    context_profile = _coerce_profile(state.get("context_profile"))
    history_profile = _coerce_profile(state.get("history_profile"))
    preference_profile = _coerce_profile(state.get("preference_profile"))
    user_input = _extract_latest_user_input(state.get("messages") or [])

    debug("decision_agent", f"user_input={user_input!r}")
    debug("decision_agent", f"intent_profile={intent_profile}")
    debug("decision_agent", f"context_profile={context_profile}")
    debug("decision_agent", f"preference_profile={preference_profile}")

    brief = build_recommendation_brief(intent_profile, history_profile, preference_profile)
    debug("decision_agent", f"brief filters={brief['filters']} history={brief['history']}")
    debug("decision_agent", f"brief summary={brief['summary']!r}")

    return {
        "ready_to_recommend": True,
        "decision": "recommend",
        "messages": [AIMessage(content="Ready to recommend.", name="decision_agent")],
        "summary": brief["summary"],
        "recommendation_brief": brief,
    }
