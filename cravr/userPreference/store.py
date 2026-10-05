"""Cravr's own preference store.

Swiggy's Food MCP has no tool for saved preferences, so Cravr keeps them
itself, one record per user. Fields follow the mock server's
`user_preferences` shape (`dietary_preference`, `default_budget_inr`,
`allergies`, `preferred_cuisines`, `preferred_flavors`) plus the learned
taste profile (see taste.py), so `_apply_saved_preferences` reads both.

Where it's stored is decided by cravr/storage.py: Postgres when
`DATABASE_URL` is set (survives Render redeploys), otherwise a JSON file.

Primary key: (namespace="preferences", key=<owner>). The owner is the
Swiggy `address_id` -- the id the whole flow runs on (search, cart, orders,
checkout all take it). Real Swiggy calls carry no user id, so `user_id`
(the mock server's id) is only the fallback, then `default`.
Use `profile_owner(state)` to get it from graph state.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .. import storage
from ..debug_utils import debug

_DEFAULT_PATH = Path(__file__).resolve().parents[2] / "mcp_server" / "data" / "user_preferences.json"
_NAMESPACE = "preferences"
DEFAULT_USER_KEY = "default"


def _path() -> Path:
    return Path(os.getenv("CRAVR_PREFS_PATH") or _DEFAULT_PATH)


def _user_key(owner: str | None) -> str:
    return str(owner).strip() if owner and str(owner).strip() else DEFAULT_USER_KEY


def profile_owner(state: dict[str, Any] | None) -> str | None:
    """Whose profile this is: the delivery `address_id`, else the mock `user_id`."""
    state = state or {}
    return (state.get("context_profile") or {}).get("address_id") or state.get("user_id")


def load_preferences(owner: str | None) -> dict[str, Any]:
    """Saved preferences for this owner (address_id), in the same response shape the
    mock `user_preferences` tool returns."""
    try:
        prefs = storage.get(_NAMESPACE, _user_key(owner), _path()) or {}
    except Exception as exc:  # a storage outage must never block ordering
        debug("preference_store", f"load failed ({type(exc).__name__}: {exc})")
        prefs = {}
    if not prefs:
        return {"status": "no_results", "preferences": {}}
    return {"status": "success", "preferences": dict(prefs)}


def save_preferences(owner: str | None, **fields: Any) -> dict[str, Any]:
    """Merge `fields` into this user's saved preferences. A value of None
    removes that field. Returns the updated preferences."""
    key = _user_key(owner)

    def change(current: dict[str, Any] | None) -> dict[str, Any]:
        prefs = dict(current or {})
        for name, value in fields.items():
            if value is None:
                prefs.pop(name, None)
            else:
                prefs[name] = value
        return prefs

    prefs = storage.update(_NAMESPACE, key, _path(), change) or {}
    debug("preference_store", f"saved preferences for {key!r} ({storage.backend_name()}): {prefs}")
    return prefs


def clear_preferences(owner: str | None) -> None:
    """Forget everything learned for this user (e.g. after a bad inference)."""
    storage.update(_NAMESPACE, _user_key(owner), _path(), lambda _current: {})
