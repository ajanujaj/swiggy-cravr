"""Bridge between the Chainlit chat session and the compiled Cravr LangGraph app.

Owns every call into `cravr.graph.app` -- starting a thread, sending a
free-text message, and resuming an `interrupt()` -- and turns each result
directly into chat message(s) via `components.py`. No other UI module
calls into the graph directly.

Ported from the Streamlit version of this file 1:1 in *responsibility*:
the exact same invoke/interrupt/resume contract `cravr/run.py` (the CLI)
already uses. What changes is *mechanics* -- Streamlit reran this whole
script on every click and read `st.session_state.history` back in a
separate render pass; Chainlit is message-stream-native, so this module
sends messages itself, via `components.py`, the moment each result is
ready -- no rerun, no separate render step.

`cravr.graph.app.invoke(...)` is still the exact same *synchronous* call
the CLI and Streamlit both used, unchanged -- wrapped in `cl.make_async` so
it runs on a worker thread instead of blocking Chainlit's asyncio event
loop (and every other connected user's session) for however long an LLM
call takes.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import chainlit as cl
from langgraph.types import Command

# UI/ lives one level below the repo root, next to the `cravr` package.
# Adding the repo root to sys.path here (rather than relying on however
# Chainlit resolves the launched app file's directory) means `import cravr`
# resolves the same way whether this is run via `chainlit run`, mounted
# into `server.py` via `mount_chainlit`, or anything else.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from cravr.debug_utils import debug  # noqa: E402
from cravr.graph import app as cravr_app  # noqa: E402

import components  # noqa: E402
import state  # noqa: E402

# Wrapped once, at import time -- cl.make_async just builds a thread-pool
# wrapper around the function reference, no per-call cost to redo that.
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
    """Turn one `app.invoke(...)` result into chat message(s).

    Mirrors `cravr/run.py`'s `_resume_until_complete` / `_print_assistant_reply`
    exactly, just sending Chainlit messages instead of printing to stdout,
    and stopping at the first interrupt instead of blocking on `input()` --
    the next interrupt (if any) is handled the same way on the *next* user
    action, one graph invocation at a time.
    """
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
        # Intent collection can end on the user's own last clarification --
        # not an assistant turn, so nothing to render (same guard as run.py).
        return

    content = getattr(last, "content", str(last))
    source = getattr(last, "name", None) or "assistant"

    dishes = None
    options = None
    if source == "recommendation_agent":
        rec_state = result.get("recommendation_state") or {}
        dishes = rec_state.get("last_recommendations") or None
        # Tappable dish-pick / yes-widen buttons, built by RecommendAgent
        # itself (see `_dish_pick_options`/`ONLY_MATCH_OPTIONS`) so they can
        # never drift out of sync with the numbered text in `content`.
        options = rec_state.get("options") or None

    debug("ui", f"reply from {source!r}: {content!r}" + (f" ({len(dishes)} dish(es))" if dishes else ""))

    if dishes:
        await components.send_dish_cards(content, dishes, options)
    else:
        await components.send_assistant_message(content, options)


async def send(value: str, display_label: str | None = None) -> None:
    """Send one turn to the graph: either a resume value or a fresh message.

    `display_label` is accepted for symmetry with the old Streamlit bridge
    (a dish's name vs. the bare position number it resolves to) but isn't
    used here -- `chainlit_app.on_pick` already echoes the picked label as
    its own user-facing message before calling this, since Chainlit (unlike
    Streamlit's full-history rerender) doesn't otherwise show what a button
    click "said" at all.
    """
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
    except Exception as exc:  # noqa: BLE001 -- surface backend errors in the chat, not a stack trace
        debug("ui", f"invoke failed: {type(exc).__name__}: {exc}")
        await _report_backend_error(exc)
        return
    await process_result(result)


async def start_conversation() -> None:
    debug("ui", f"starting conversation thread={state.get_config()['configurable']['thread_id']!r}")
    try:
        result = await _ainvoke(
            {
                "messages": [],
                "intent_profile": {},
                "context_profile": {},
                "history_profile": {},
                "user_id": state.DEFAULT_USER_ID,
            },
            config=state.get_config(),
        )
    except Exception as exc:  # noqa: BLE001
        debug("ui", f"invoke failed: {type(exc).__name__}: {exc}")
        await _report_backend_error(exc)
        return
    await process_result(result)
