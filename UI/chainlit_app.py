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

import components
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
    if state.get_multiselect():
        state.set_multiselect(None)
    await state.clear_live_options()
    await graph_bridge.send(message.content)


@cl.action_callback("pick")
async def on_pick(action: cl.Action) -> None:
    value = action.payload.get("value")
    label = action.payload.get("label") or value
    await state.clear_live_options(selected=value)
    await cl.Message(content=str(label), type="user_message").send()
    await graph_bridge.send(value, display_label=label)


@cl.action_callback("multiselect_submit")
async def on_multiselect_submit(action: cl.Action) -> None:
    if not state.get_multiselect():
        return
    values = [str(v) for v in action.payload.get("values") or []]
    await state.clear_live_options(selected=values)
    value, label = components.multiselect_answer(values)
    await cl.Message(content=label, type="user_message").send()
    await graph_bridge.send(value, display_label=label)
