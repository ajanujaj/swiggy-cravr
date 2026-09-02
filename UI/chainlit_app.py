"""Chainlit chat UI entry point for Cravr.

This is the `target` file `server.py` mounts via `chainlit.utils.mount_chainlit`.
Chainlit owns the chat page itself -- bubbles, the input box, scrolling,
layout, "start a new chat" -- so this file only wires session lifecycle
events to `graph_bridge`, the one module that talks to `cravr.graph.app`.
None of it re-implements any conversation logic: same invoke/interrupt/
resume contract `cravr/run.py` (the CLI) already uses.

Every clickable option (a clarification chip or a dish-pick button) funnels
through the single `"pick"` action name registered below, carrying its real
value + display label in the action's payload -- one callback, not one per
question, so a brand-new kind of choice never needs new wiring.

Run standalone for local development:
    chainlit run UI/chainlit_app.py -w
(`-w` auto-reloads on save.) In production this file is never run directly
-- `server.py` mounts it alongside the Swiggy OAuth callback, see there.
"""

from __future__ import annotations

import sys
from pathlib import Path

# So `import components`, `import state`, etc. resolve to the sibling
# modules in this folder regardless of the working directory Chainlit (or
# `mount_chainlit`) was launched from.
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
    # Unconditional (not gated on CRAVR_DEBUG) -- if this line doesn't show
    # up in the terminal the server process is running in, the problem is
    # that stderr isn't reaching that terminal at all, not that CRAVR_DEBUG
    # is unset.
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
    """A typed free-text message -- Chainlit has already shown it as a user
    bubble by the time this fires, so unlike a button click (see `on_pick`)
    there's nothing extra to echo here."""
    await state.clear_live_options()
    await graph_bridge.send(message.content)


@cl.action_callback("pick")
async def on_pick(action: cl.Action) -> None:
    """A clarification chip or a dish-pick button. Echoes the picked label
    as a user-facing bubble first -- Chainlit doesn't do this on its own
    for an action click the way it does for typed text -- then sends the
    underlying value through the exact same path a typed answer takes.
    """
    await state.clear_live_options()
    value = action.payload.get("value")
    label = action.payload.get("label") or value
    await cl.Message(content=str(label), type="user_message").send()
    await graph_bridge.send(value, display_label=label)
