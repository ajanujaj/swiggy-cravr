"""Bridge between Streamlit session state and the compiled Cravr LangGraph app.

Owns every call into `cravr.graph.app` -- starting a thread, sending a
free-text message, and resuming an `interrupt()` -- and translates each
result into `st.session_state.history` entries. `components.py` only reads
what this module writes; no other UI module calls into the graph directly.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import streamlit as st
from langgraph.types import Command

# UI/ lives one level below the repo root, next to the `cravr` package.
# Adding the repo root to sys.path here (rather than relying on however
# Streamlit resolves the launched script's directory) means `import cravr`
# resolves the same way whether Streamlit is started from the repo root,
# from inside UI/, or anywhere else.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from cravr.debug_utils import debug  # noqa: E402
from cravr.graph import app as cravr_app  # noqa: E402
from cravr.options import OTHER_VALUE  # noqa: E402

DEFAULT_USER_ID = "USR-002"


def process_result(result: dict[str, Any] | None) -> None:
    """Turn one `app.invoke(...)` result into a chat-history entry.

    Mirrors `cravr/run.py`'s `_resume_until_complete` / `_print_assistant_reply`
    exactly, just writing into `st.session_state.history` instead of stdout,
    and stopping at the first interrupt instead of blocking on `input()` --
    the next interrupt (if any) is handled the same way on the *next* user
    action, one Streamlit rerun at a time.
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
        st.session_state.awaiting_interrupt = True
        st.session_state.history.append(
            {"role": "assistant", "content": question, "source": source, "options": options}
        )
        return

    st.session_state.awaiting_interrupt = False

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
    if source == "recommendation_agent":
        rec_state = result.get("recommendation_state") or {}
        dishes = rec_state.get("last_recommendations") or None

    debug("ui", f"reply from {source!r}: {content!r}" + (f" ({len(dishes)} dish(es))" if dishes else ""))
    st.session_state.history.append(
        {"role": "assistant", "content": content, "source": source, "dishes": dishes}
    )


def send(value: str, display_label: str | None = None) -> None:
    """Send one turn to the graph: either a resume value or a fresh message."""
    st.session_state.history.append({"role": "user", "content": display_label or value})
    st.session_state.error = None
    try:
        if st.session_state.awaiting_interrupt:
            debug("ui", f"resuming interrupt with {value!r}")
            result = cravr_app.invoke(Command(resume=value), config=st.session_state.config)
        else:
            debug("ui", f"sending new message {value!r}")
            result = cravr_app.invoke(
                {"messages": [{"role": "user", "content": value}]},
                config=st.session_state.config,
            )
    except Exception as exc:  # noqa: BLE001 -- surface backend errors in the UI, not a stack trace
        debug("ui", f"invoke failed: {type(exc).__name__}: {exc}")
        st.session_state.error = str(exc)
        return
    process_result(result)


def start_conversation() -> None:
    debug("ui", f"starting conversation thread={st.session_state.config['configurable']['thread_id']!r}")
    st.session_state.error = None
    try:
        result = cravr_app.invoke(
            {
                "messages": [],
                "intent_profile": {},
                "context_profile": {},
                "history_profile": {},
                "user_id": DEFAULT_USER_ID,
            },
            config=st.session_state.config,
        )
    except Exception as exc:  # noqa: BLE001
        debug("ui", f"invoke failed: {type(exc).__name__}: {exc}")
        st.session_state.error = str(exc)
        return
    process_result(result)
