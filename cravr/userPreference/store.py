"""Cravr's own preference store.

Swiggy's Food MCP has no tool for saved preferences, so Cravr keeps them
itself: a small JSON file, one entry per user. Fields follow the mock
server's `user_preferences` shape (`dietary_preference`,
`default_budget_inr`, `allergies`, `preferred_cuisines`,
`preferred_flavors`) so `_apply_saved_preferences` reads both the same way.

Real Swiggy calls carry no user id (the OAuth token identifies the
account), so without one the entry is keyed `default` -- right for a
single-account deployment. Swap this module for a database once Cravr
serves several accounts.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

from ..debug_utils import debug

_DEFAULT_PATH = Path(__file__).resolve().parents[2] / "mcp_server" / "data" / "user_preferences.json"
_LOCK = threading.Lock()
DEFAULT_USER_KEY = "default"


def _path() -> Path:
    return Path(os.getenv("CRAVR_PREFS_PATH") or _DEFAULT_PATH)


def _user_key(user_id: str | None) -> str:
    return str(user_id).strip() if user_id and str(user_id).strip() else DEFAULT_USER_KEY


def _load_all() -> dict[str, dict[str, Any]]:
    path = _path()
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8") or "{}")
    except (OSError, json.JSONDecodeError) as exc:
        debug("preference_store", f"could not read {path} ({exc}) -- treating as empty")
        return {}
    return data if isinstance(data, dict) else {}


def load_preferences(user_id: str | None) -> dict[str, Any]:
    """Saved preferences for this user, in the same response shape the
    mock `user_preferences` tool returns."""
    with _LOCK:
        prefs = _load_all().get(_user_key(user_id)) or {}
    if not prefs:
        return {"status": "no_results", "preferences": {}}
    return {"status": "success", "preferences": dict(prefs)}


def save_preferences(user_id: str | None, **fields: Any) -> dict[str, Any]:
    """Merge `fields` into this user's saved preferences. A value of None
    removes that field. Returns the updated preferences."""
    key = _user_key(user_id)
    with _LOCK:
        data = _load_all()
        prefs = dict(data.get(key) or {})
        for name, value in fields.items():
            if value is None:
                prefs.pop(name, None)
            else:
                prefs[name] = value
        data[key] = prefs
        path = _path()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)  # atomic, so a crash never leaves half a file
    debug("preference_store", f"saved preferences for {key!r}: {prefs}")
    return prefs
