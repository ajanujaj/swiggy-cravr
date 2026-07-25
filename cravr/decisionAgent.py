"""Decision agent: uses the LLM to decide whether the request is ready for recommendations."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from .llm import get_llm
from .state import CravrState


DECISION_PROMPT = """
You are Cravr's decision agent. Decide whether the assistant is ready to recommend a food option to the user.

You must inspect the user's latest message plus these profiles:
- intent_profile
- context_profile
- history_profile
- summary

Return exactly one valid JSON object with this shape:
{
  "ready_to_recommend": true | false,
  "decision": "recommend" | "clarify",
  "clarification_message": string | null
  "Summary": string
}

Rules:
- Set ready_to_recommend to true only when the assistant has enough information to make a useful recommendation.
- If important details are missing, set decision to "clarify" and provide a concise clarification_message.
- Use history_profile as supporting context when it is meaningful, but do not require it if the other profiles already provide enough information.
- Do not recommend if the request is vague or the profiles do not contain enough evidence.
- Return null for clarification_message when ready_to_recommend is true.
- Output only JSON. Do not include markdown fences or explanatory text.
""".strip()


def _coerce_profile(profile: Any) -> dict:
    if isinstance(profile, dict):
        return profile
    return {}


def _extract_latest_user_input(messages: Any) -> str:
    if not messages:
        return ""

    for message in reversed(list(messages)):
        if isinstance(message, dict):
            role = message.get("role")
            content = message.get("content")
            if role == "user" and content:
                return str(content)
            if role == "human" and content:
                return str(content)
        if hasattr(message, "type") and getattr(message, "type") in {"human", "user"}:
            content = getattr(message, "content", "")
            if content:
                return str(content)
        if hasattr(message, "role") and getattr(message, "role") == "user":
            content = getattr(message, "content", "")
            if content:
                return str(content)

    last_message = messages[-1]
    content = getattr(last_message, "content", last_message)
    return str(content)


def _extract_json_object(raw_content: str) -> dict:
    text = (raw_content or "").strip()
    if not text:
        raise ValueError("LLM returned no content")

    if text.startswith("```"):
        text = text.replace("```json", "").replace("```", "").strip()

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start : end + 1]

    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("Decision response was not a JSON object")
    return parsed


def _build_summary(intent_profile: dict, context_profile: dict, history_profile: dict, user_input: str) -> str:
    """Create a single concise summary that captures the user's intent, context, and order history."""
    parts: list[str] = []

    if user_input:
        parts.append(f"User asked: {user_input}")

    intent_bits = []
    if intent_profile.get("craving"):
        intent_bits.append(f"craving={intent_profile['craving']}")
    if intent_profile.get("cuisine"):
        intent_bits.append(f"cuisine={intent_profile['cuisine']}")
    dietary = intent_profile.get("dietary_preferences") or []
    if dietary:
        intent_bits.append(f"dietary_preferences={','.join(map(str, dietary))}")
    budget = intent_profile.get("budget_inr")
    if budget is not None:
        intent_bits.append(f"budget_inr={budget}")
    if intent_bits:
        parts.append("intent:" + "; ".join(intent_bits))

    context_bits = []
    if context_profile.get("location"):
        context_bits.append(f"location={context_profile['location']}")
    if context_profile.get("weather"):
        context_bits.append(f"weather={context_profile['weather']}")
    if context_bits:
        parts.append("context:" + "; ".join(context_bits))

    history_bits = []
    orders = history_profile.get("orders") or []
    if orders:
        history_bits.append(f"recent_orders={len(orders)}")
        recent = orders[0]
        if recent.get("restaurant_name"):
            history_bits.append(f"last_restaurant={recent['restaurant_name']}")
        if recent.get("item_name"):
            history_bits.append(f"last_item={recent['item_name']}")
    else:
        history_bits.append("recent_orders=0")
    if history_bits:
        parts.append("history:" + "; ".join(history_bits))

    return " | ".join(parts) if parts else "No additional context available."


def _is_vague_request(user_input: str, intent_profile: dict, context_profile: dict) -> bool:
    """Return True when the request is too general to recommend confidently."""
    normalized = (user_input or "").strip().lower()
    if not normalized:
        return True

    explicit_preference_markers = [
        "spicy",
        "vegetarian",
        "vegan",
        "non-veg",
        "chinese",
        "indian",
        "italian",
        "mexican",
        "thai",
        "pizza",
        "burger",
        "biryani",
        "fried rice",
        "dessert",
        "budget",
        "under",
        "below",
        "max",
        "craving",
        "cuisine",
        "protein",
        "low carb",
        "high protein",
        "breakfast",
        "lunch",
        "dinner",
        "snack",
        "tea",
        "coffee",
        "sweet",
        "salty",
        "soup",
        "salad",
        "rice",
        "noodles",
    ]
    if any(marker in normalized for marker in explicit_preference_markers):
        return False

    generic_terms = [
        "something",
        "anything",
        "eat",
        "food",
        "meal",
        "restaurant",
        "order",
        "recommend",
        "light food",
        "healthy food",
        "quick food",
        "light",
        "healthy",
        "quick",
    ]
    if not any(term in normalized for term in generic_terms):
        return False

    missing_fields = []
    for profile in (intent_profile, context_profile):
        if not isinstance(profile, dict):
            continue
        missing_fields.extend(profile.get("missing_information") or [])
        missing_fields.extend(profile.get("important_missing_information") or [])

    missing_text = " ".join(str(item or "") for item in missing_fields).lower()
    return any(term in missing_text for term in ["craving", "cuisine", "diet", "budget", "location"])


def decision_agent_node(state: CravrState) -> dict:
    """Ask the LLM to decide whether the bot can recommend now or should ask a clarification."""
    intent_profile = _coerce_profile(state.get("intent_profile"))
    context_profile = _coerce_profile(state.get("context_profile"))
    history_profile = _coerce_profile(state.get("history_profile"))

    user_input = _extract_latest_user_input(state.get("messages") or [])
    summary = _build_summary(intent_profile, context_profile, history_profile, user_input)
    payload = {
        "user_input": user_input,
        "intent_profile": intent_profile,
        "context_profile": context_profile,
        "history_profile": history_profile,
        "summary": summary,
    }

    if _is_vague_request(user_input, intent_profile, context_profile):
        return {
            "ready_to_recommend": False,
            "decision": "clarify",
            "messages": [AIMessage(content="I can help with that, but I need a bit more detail such as your craving, cuisine, dietary preference, or budget.")],
            "summary": summary,
        }

    response = get_llm().invoke([
        SystemMessage(content=DECISION_PROMPT),
        HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
    ])

    raw_content = str(getattr(response, "content", ""))
    try:
        result = _extract_json_object(raw_content)
    except (json.JSONDecodeError, ValueError):
        result = {
            "ready_to_recommend": False,
            "decision": "clarify",
            "clarification_message": "I need a bit more detail such as your craving, cuisine, dietary preference, or budget before I can recommend something confidently.",
        }

    ready_to_recommend = bool(result.get("ready_to_recommend"))
    decision = result.get("decision") if result.get("decision") in {"recommend", "clarify"} else "clarify"
    clarification_message = result.get("clarification_message") or (
        "I need a bit more detail such as your craving, cuisine, dietary preference, or budget before I can recommend something confidently."
    )

    return {
        "ready_to_recommend": ready_to_recommend,
        "decision": decision,
        "messages": [AIMessage(content=clarification_message if not ready_to_recommend else "Ready to recommend.")],
        "summary": summary,
    }
