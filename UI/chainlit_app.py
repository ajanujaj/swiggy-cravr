"""Chainlit chat UI entry point for Cravr -- the `target` file `server.py`
mounts via `chainlit.utils.mount_chainlit`. Only wires session lifecycle
events to `graph_bridge`; no conversation logic lives here.

Run standalone for local development:
    chainlit run UI/chainlit_app.py -w
"""

from __future__ import annotations

import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))
_REPO_ROOT = _THIS_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import chainlit as cl

from cravr.debug_utils import debug_enabled

import graph_bridge
import state


@cl.on_chat_start
async def on_chat_start() -> None:
    state.init_state()
    print(
        "[Cravr UI] new session started -- set CRAVR_DEBUG=1 before starting "
        "the server to also see per-turn agent traces here.",
        file=sys.stderr,
    )
    if not debug_enabled():
        print("[Cravr UI] CRAVR_DEBUG is not set, so agent traces are off.", file=sys.stderr)
    await graph_bridge.start_conversation()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    await state.clear_live_options()
    await graph_bridge.send(message.content)


@cl.action_callback("pick")
async def on_pick(action: cl.Action) -> None:
    """A clarification chip or a dish-pick button -- echoes the picked
    label as a user-facing bubble, then sends the value like typed text."""
    await state.clear_live_options()
    value = action.payload.get("value")
    label = action.payload.get("label") or value
    await cl.Message(content=str(label), type="user_message").send()
    await graph_bridge.send(value, display_label=label)
