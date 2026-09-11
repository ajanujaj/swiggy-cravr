"""Bridge between the Chainlit chat session and the compiled Cravr
LangGraph app -- owns every call into `cravr.graph.app` (start a thread,
send a message, resume an interrupt) and turns each result into chat
message(s) via `components.py`. No other UI module calls the graph
directly.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import chainlit as cl
from langgraph.types import Command

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from cravr.debug_utils import debug  # noqa: E402
from cravr.graph import app as cravr_app  # noqa: E402

import components  # noqa: E402
import state  # noqa: E402

_ainvoke = cl.make_async(cravr_app.invoke)

_BACKEND_ERROR_HINT = (
    "\n\nMake sure the mock MCP server (`python mcp_server/run_mcp.py`) is "
    "running and GROQ_API_KEY / groq_apikey is set."
)


async def _report_backend_error(exc: Exception) -> None:
    await cl.Message(
        content=f"Something went wrong talking to the backend: {exc}{_BACKEND_ERROR_HINT}",
        author="Cravr",
    ).send()


async def process_result(result: dict[str, Any] | None) -> None:
    """Turn one `app.invoke(...)` result into chat message(s), mirroring
    `cravr/run.py`'s resume/print loop but stopping at the first interrupt."""
    if result is None:
        return

    if result.get("__interrupt__"):
        pending = result["__interrupt__"][0]
        payload = pending.value
        if isinstance(payload, dict):
            question = payload.get("question")
            source = payload.get("agent") or "assistant"
            options = payload.get("options") or []
        else:
            question, source, options = payload, "assistant", []

        debug("ui", f"interrupt from {source!r}: {question!r} options={[o.get('value') for o in options]}")
        state.set_awaiting_interrupt(True)
        await components.send_option_message(question, options)
        return

    state.set_awaiting_interrupt(False)

    messages = result.get("messages") or []
    if not messages:
        return
    last = messages[-1]
    role = getattr(last, "type", None) or getattr(last, "role", None)
    if role in ("human", "user"):
        return

    content = getattr(last, "content", str(last))
    source = getattr(last, "name", None) or "assistant"

    dishes = None
    options = None
    if source == "recommendation_agent":
        rec_state = result.get("recommendation_state") or {}
        dishes = rec_state.get("last_recommendations") or None
        options = rec_state.get("options") or None

    debug("ui", f"reply from {source!r}: {content!r}" + (f" ({len(dishes)} dish(es))" if dishes else ""))

    if dishes:
        await components.send_dish_cards(content, dishes, options)
    else:
        await components.send_assistant_message(content, options)


async def send(value: str, display_label: str | None = None) -> None:
    """Send one turn to the graph: either a resume value or a fresh
    message. `display_label` is unused here -- `chainlit_app.on_pick`
    already echoes it before calling this."""
    try:
        if state.is_awaiting_interrupt():
            debug("ui", f"resuming interrupt with {value!r}")
            result = await _ainvoke(Command(resume=value), config=state.get_config())
        else:
            debug("ui", f"sending new message {value!r}")
            result = await _ainvoke(
                {"messages": [{"role": "user", "content": value}]},
                config=state.get_config(),
            )
    except Exception as exc:  # noqa: BLE001
        debug("ui", f"invoke failed: {type(exc).__name__}: {exc}")
        await _report_backend_error(exc)
        return
    await process_result(result)


async def start_conversation() -> None:
    debug("ui", f"starting conversation thread={state.get_config()['configurable']['thread_id']!r}")
    try:
        result = await _ainvoke(
            {
                # No hardcoded user_id -- Swiggy's OAuth token already scopes
                # everything to one real account (address_id is the anchor
                # now, not an internal USR-xxx identity). Nodes that still
                # take a user_id (the mock-only fallback branches) tolerate
                # None fine; they're unreachable once using_real_food_mcp().
                "messages": [],
                "intent_profile": {},
                "context_profile": {},
                "history_profile": {},
            },
            config=state.get_config(),
        )
    except Exception as exc:  # noqa: BLE001
        debug("ui", f"invoke failed: {type(exc).__name__}: {exc}")
        await _report_backend_error(exc)
        return
    await process_result(result)
