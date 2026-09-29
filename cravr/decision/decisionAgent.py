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
    if intent_profile.get("opening_picks"):
        # Dishes-first opening: the recommender shows these cards instead of searching.
        from ..recommend.opening import new_opening_token

        brief["opening_picks"] = intent_profile["opening_picks"]
        brief["opening_token"] = new_opening_token()
    debug("decision_agent", f"brief filters={brief['filters']} history={brief['history']}")
    debug("decision_agent", f"brief summary={brief['summary']!r}")

    return {
        "ready_to_recommend": True,
        "decision": "recommend",
        "messages": [AIMessage(content="Ready to recommend.", name="decision_agent")],
        "summary": brief["summary"],
        "recommendation_brief": brief,
    }
