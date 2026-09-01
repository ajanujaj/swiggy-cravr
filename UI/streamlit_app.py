"""Streamlit chat UI entry point for Cravr.

Run alongside the existing backend exactly the way `cravr/run.py` (the CLI)
is run:

    ollama pull qwen2.5:7b-instruct        # once
    python mcp_server/run_mcp.py           # terminal 1
    streamlit run UI/streamlit_app.py      # terminal 2

This file only wires the pieces together -- `state` (session-state setup),
`graph_bridge` (the only module that talks to `cravr.graph.app`), `styles`
(CSS), and `components` (chat bubbles, option chips, dish cards). None of
them re-implement any conversation logic: they reuse the exact
invoke/interrupt/resume contract `cravr/run.py` already uses.
"""

from __future__ import annotations

import sys
from pathlib import Path

# So `import state`, `import components`, etc. resolve to the sibling
# modules in this folder regardless of the working directory Streamlit was
# launched from.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))
_REPO_ROOT = _THIS_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import streamlit as st

from cravr.debug_utils import debug_enabled

import components
import scroll
import styles
from graph_bridge import DEFAULT_USER_ID, send, start_conversation
from state import init_state, reset_state

st.set_page_config(page_title="Cravr", page_icon="🍽️", layout="centered")


def main() -> None:
    init_state()
    styles.inject()

    st.title("🍽️ Cravr")
    st.caption("Not sure what to eat? Tell me a little, or just tap an option.")

    if not st.session_state.started:
        st.session_state.started = True
        # Unconditional (not gated on CRAVR_DEBUG) -- if this line doesn't
        # show up in the terminal where `streamlit run` was launched, the
        # problem is that stderr isn't reaching that terminal at all, not
        # that CRAVR_DEBUG is unset.
        print("[Cravr UI] new session started -- set CRAVR_DEBUG=1 before "
              "`streamlit run` to also see per-turn agent traces here.", file=sys.stderr)
        if not debug_enabled():
            print("[Cravr UI] CRAVR_DEBUG is not set, so agent traces are off.", file=sys.stderr)
        with st.spinner("Starting..."):
            start_conversation()

    components.render_history()
    scroll.to_bottom()

    if st.session_state.error:
        st.error(
            "Something went wrong talking to the backend: "
            f"{st.session_state.error}\n\n"
            "Make sure the MCP server (`python mcp_server/run_mcp.py`) and Ollama are both running."
        )

    prompt = st.chat_input("Type a message...")
    if prompt:
        send(prompt)
        st.rerun()

    with st.sidebar:
        components.render_sidebar(DEFAULT_USER_ID, reset_state)


if __name__ == "__main__":
    main()
