"""Intent Discovery agent: turns a food-ordering request into structured context."""

from __future__ import annotations

import json

from langchain_core.messages import SystemMessage

from .llm import get_llm
from .state import CravrState


INTENT_DISCOVERY_PROMPT = """
You are Cravr's Intent Discovery agent. Analyze the user's food-ordering message.
You are not a conversational assistant: do not recommend food and do not ask a
question. Extract only facts explicitly stated or safely implied by the message.

Return exactly one valid JSON object with these keys:
{
  "mood": string or null,
  "hunger_level": "low" | "medium" | "high" | null,
  "occasion": string or null,
  "meal_time": string or null,
  "craving": string or null,
  "cuisine": string or null,
  "dietary_preferences": [string],
  "allergies_or_avoidances": [string],
  "budget_inr": number or null,
  "minimum_rating": number or null,
  "delivery_preference": "delivery" | "dine_in" | null,
  "location": string or null,
  "other_constraints": [string],
  "missing_information": [string]
}

Use null or [] when unknown. Never fabricate values. Add only information that
would materially improve a food recommendation to `missing_information`; prefer
craving/cuisine, dietary needs, and budget when they are absent.
""".strip()


def intent_discovery_node(state: CravrState) -> dict:
    """Call the local model and save a resilient structured intent profile."""
    response = get_llm().invoke([SystemMessage(content=INTENT_DISCOVERY_PROMPT), *state["messages"]])
    raw_profile = str(response.content)
    try:
        profile = json.loads(raw_profile)
        if not isinstance(profile, dict):
            raise ValueError("Intent response was not a JSON object")
    except (json.JSONDecodeError, ValueError):
        profile = {
            "raw_intent": raw_profile,
            "missing_information": ["craving_or_cuisine"],
        }
    # Post-process missing information to surface important items and
    # provide short clarification questions the orchestrator can use.
    missing = profile.get("missing_information") or []

    def _is_important(item: str) -> bool:
        important_keywords = [
            "craving",
            "cuisine",
            "vegetarian",
            "diet",
            "budget"
        ]
        it = (item or "").lower()
        return any(k in it for k in important_keywords)

    def _question_for(item: str) -> str:
        it = (item or "").lower()
        if "craving" in it or "cuisine" in it:
            return "Do you have a specific craving or cuisine in mind?"
        if "vegetarian" in it or "diet" in it:
            return "Should the meal be strictly vegetarian or is ovo/veg+okay?"
        if "budget" in it or "inr" in it:
            return "What's your budget for this order in INR?"
        if "location" in it or "city" in it:
            return "What's the delivery location (city/area)?"
        if "rain" in it or "weather" in it:
            return "Is the weather (e.g., raining) affecting delivery where you are?"
        if "delivery" in it or "dine_in" in it:
            return "Do you prefer delivery or dining in?"
        return "Can you clarify this: %s" % item

    important_missing = [m for m in missing if _is_important(m)]
    clarification_questions = [_question_for(m) for m in important_missing]

    # Attach derived fields so downstream orchestrator can act immediately.
    profile.setdefault("important_missing_information", important_missing)
    profile.setdefault("clarification_questions", clarification_questions)

    return {"intent_profile": profile}
