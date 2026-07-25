"""Context Discovery agent: turns a food-ordering request into structured context."""

from __future__ import annotations

import json

from langchain_core.messages import SystemMessage

from .llm import get_llm
from .state import CravrState


CONTEXT_DISCOVERY_PROMPT = """
You are Cravr's Context discovery agent. Analyze the user's food-ordering message.
You are not a conversational assistant: do not recommend food and do not ask a
question. Extract only facts explicitly stated or safely implied by the message.

Return exactly one valid JSON object with these keys:
{
  "location": string or null,
  "weather": string or null
}

Use null or [] when unknown. Never fabricate values. Add only information that
would materially improve a food recommendation to `missing_information`; prefer
craving/cuisine, dietary needs, and budget when they are absent.
""".strip()


def context_discovery_node(state: CravrState) -> dict:
    """Call the local model and save a resilient structured context profile."""
    response = get_llm().invoke([SystemMessage(content=CONTEXT_DISCOVERY_PROMPT), *state["messages"]])
    raw_profile = str(response.content)
    try:
        profile = json.loads(raw_profile)
        if not isinstance(profile, dict):
            raise ValueError("Context response was not a JSON object")
    except (json.JSONDecodeError, ValueError):
        profile = {
            "raw_context": raw_profile,
            "missing_information": ["craving_or_cuisine"],
        }
    missing = profile.get("missing_information") or []

    def _is_important(item: str) -> bool:
        important_keywords = [
            "location",
            "weather",
        ]
        it = (item or "").lower()
        return any(k in it for k in important_keywords)

    def _question_for(item: str) -> str:
        it = (item or "").lower()
        if "location" in it or "city" in it:
            return "What's the delivery location (city/area)?"
        if "delivery" in it or "dine_in" in it:
            return "Do you prefer delivery or dining in?"
        return "Can you clarify this: %s" % item

    important_missing = [m for m in missing if _is_important(m)]
    clarification_questions = [_question_for(m) for m in important_missing]

    # Attach derived fields so downstream orchestrator can act immediately.
    profile.setdefault("important_missing_information", important_missing)
    profile.setdefault("clarification_questions", clarification_questions)

    return {"context_profile": profile}
