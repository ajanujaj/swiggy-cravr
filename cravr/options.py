from __future__ import annotations

import json
import threading
from typing import Any, Callable

from langchain_core.messages import HumanMessage, SystemMessage

from .debug_utils import debug
from .llm import get_llm
from .schemas import SuggestedOptions

OTHER_VALUE = "__other__"
OTHER_LABEL = "Something else (type your own)"

_OPTIONS_PROMPT = """
You are Cravr's option-suggestion helper. Cravr is a food-ordering assistant
for someone who isn't sure what to eat, so it should ask for as little
typing as possible. Given a question Cravr is about to ask and whatever is
already known about the request, propose up to 5 short (1-4 word), concrete,
mutually exclusive answer choices a busy person could tap instead of typing
a sentence -- aim for 5 whenever the context genuinely supports that many
distinct, plausible answers; offer fewer only when it doesn't (never pad
with a weak or redundant choice just to hit 5). Tailor them tightly to the
context given -- meal time, meal already chosen, hunger level,
cuisine/craving hints -- so a breakfast question and a late-night question
about the same field get visibly different, time-appropriate choices rather
than the same generic list. Never include a catch-all choice like "other" or
"something else" -- that is added separately. Never invent facts about the
user; base choices only on the question and the given context.

The user orders delivery on Swiggy in India. Suggest dishes and categories
that are commonly delivered in Indian cities (the given delivery area, if
any) -- not niche items few restaurants deliver, such as charcuterie boards
or tapas, unless the user asked for them. When `recent_orders` or
`favourite_restaurants` are given, make at least one choice come straight
from them (a dish they have ordered before), and keep the rest in a similar
spirit and price range.
""".strip()

# Options shown with an interrupt must be the SAME options when the graph
# resumes: LangGraph re-runs the node from the top on resume, so asking the
# LLM again produced a different list (a typed "3" could then resolve to a
# different choice than the one on screen, and each resume cost an extra
# LLM call). Keyed by thread id + caller's key; callers clear their entries
# once the node finishes.
_OPTION_CACHE: dict[tuple, Any] = {}
_CACHE_LOCK = threading.Lock()


def _current_thread_id() -> str | None:
    try:
        from langgraph.config import get_config

        return (get_config().get("configurable") or {}).get("thread_id")
    except Exception:
        return None  # called outside a graph run (tests, scripts) -- no caching


def stable(key: tuple, build: Callable[[], Any]) -> Any:
    """Return the value built for `key` in this thread, building it only once."""
    thread_id = _current_thread_id()
    if thread_id is None:
        return build()
    full_key = (thread_id, *key)
    with _CACHE_LOCK:
        if full_key in _OPTION_CACHE:
            debug("options", f"reusing options shown earlier for {key!r}")
            return _OPTION_CACHE[full_key]
    value = build()
    with _CACHE_LOCK:
        _OPTION_CACHE.setdefault(full_key, value)
        return _OPTION_CACHE[full_key]


def clear_stable(prefix: tuple) -> None:
    """Forget this thread's cached entries whose key starts with `prefix`."""
    thread_id = _current_thread_id()
    if thread_id is None:
        return
    full_prefix = (thread_id, *prefix)
    with _CACHE_LOCK:
        for key in [k for k in _OPTION_CACHE if k[: len(full_prefix)] == full_prefix]:
            del _OPTION_CACHE[key]


def suggest_options(
    question: str,
    context: dict | None = None,
    fallback: list[str] | None = None,
    include_other: bool = True,
    max_options: int = 5,
) -> list[dict[str, str]]:
    labels: list[str] = []
    try:
        structured_llm = get_llm().with_structured_output(SuggestedOptions)
        payload = {"question": question, "known_so_far": context or {}}
        result = structured_llm.invoke([
            SystemMessage(content=_OPTIONS_PROMPT),
            HumanMessage(content=json.dumps(payload, ensure_ascii=False, default=str)),
        ])
        labels = [
            label.strip()
            for label in (getattr(result, "options", None) or [])
            if isinstance(label, str) and label.strip()
        ]
        debug("options", f"LLM suggested options for {question!r}: {labels}")
    except Exception as exc:
        debug("options", f"option suggestion failed ({exc}) -- using fallback")

    if not labels:
        labels = list(fallback or [])

    seen: set[str] = set()
    deduped: list[str] = []
    for label in labels:
        key = label.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(label)

    options = [{"value": label, "label": label} for label in deduped[:max_options]]
    if include_other:
        options.append({"value": OTHER_VALUE, "label": OTHER_LABEL})
    return options


def resolve_option_choice(answer: str, options: list[dict]) -> str:
    if not options:
        return answer
    normalized = (answer or "").strip().lower().rstrip(".!")
    if not normalized:
        return answer

    if normalized.isdigit():
        position = int(normalized)
        if 1 <= position <= len(options):
            return options[position - 1].get("value", answer)
        return answer

    for option in options:
        value = str(option.get("value", "")).strip().lower()
        label = str(option.get("label", "")).strip().lower()
        if normalized in {value, label}:
            return option.get("value", answer)

    return answer
