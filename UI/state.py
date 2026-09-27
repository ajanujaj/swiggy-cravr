from __future__ import annotations

import uuid
from typing import Any

import chainlit as cl

_THREAD_CONFIG_KEY = "cravr_config"
_AWAITING_INTERRUPT_KEY = "cravr_awaiting_interrupt"
_LIVE_ELEMENTS_KEY = "cravr_live_elements"
_MULTISELECT_KEY = "cravr_multiselect"


def init_state() -> None:
    cl.user_session.set(_THREAD_CONFIG_KEY, {"configurable": {"thread_id": str(uuid.uuid4())}})
    cl.user_session.set(_AWAITING_INTERRUPT_KEY, False)
    cl.user_session.set(_LIVE_ELEMENTS_KEY, [])
    cl.user_session.set(_MULTISELECT_KEY, None)


def get_config() -> dict[str, Any]:
    return cl.user_session.get(_THREAD_CONFIG_KEY)


def is_awaiting_interrupt() -> bool:
    return bool(cl.user_session.get(_AWAITING_INTERRUPT_KEY))


def set_awaiting_interrupt(value: bool) -> None:
    cl.user_session.set(_AWAITING_INTERRUPT_KEY, value)


def get_live_elements() -> list[cl.CustomElement]:
    return cl.user_session.get(_LIVE_ELEMENTS_KEY) or []


def set_live_elements(elements: list[cl.CustomElement]) -> None:
    cl.user_session.set(_LIVE_ELEMENTS_KEY, elements)


async def clear_live_options(selected: Any = None) -> None:
    import components

    for element in get_live_elements():
        try:
            await components.lock_element(element, selected)
        except Exception:  # noqa: BLE001 -- best-effort cleanup
            pass
    set_live_elements([])


def get_multiselect() -> dict[str, Any] | None:
    return cl.user_session.get(_MULTISELECT_KEY)


def set_multiselect(value: dict[str, Any] | None) -> None:
    cl.user_session.set(_MULTISELECT_KEY, value)
