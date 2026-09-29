"""Group orders: one poll per group, stored as JSON (like the preference store)."""

from __future__ import annotations

import json
import os
import secrets
import threading
import time
from pathlib import Path
from typing import Any

_DEFAULT_PATH = Path(__file__).resolve().parents[2] / "mcp_server" / "data" / "groups.json"
_LOCK = threading.Lock()
_MAX_AGE_SECONDS = 2 * 24 * 3600  # polls older than two days are dropped on write


def _path() -> Path:
    return Path(os.getenv("CRAVR_GROUPS_PATH") or _DEFAULT_PATH)


def _load() -> dict[str, dict[str, Any]]:
    path = _path()
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8") or "{}")
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _save(data: dict[str, dict[str, Any]]) -> None:
    now = time.time()
    data = {k: v for k, v in data.items() if now - float(v.get("created_at") or now) < _MAX_AGE_SECONDS}
    path = _path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def create_group(size: int, label: str = "") -> dict[str, Any]:
    group = {
        "id": secrets.token_urlsafe(6),
        "size": max(2, min(int(size or 2), 20)),
        "label": label,
        "created_at": time.time(),
        "responses": [],
    }
    with _LOCK:
        data = _load()
        data[group["id"]] = group
        _save(data)
    return group


def get_group(group_id: str) -> dict[str, Any] | None:
    with _LOCK:
        return _load().get(str(group_id))


def add_response(group_id: str, response: dict[str, Any]) -> dict[str, Any] | None:
    """Save one person's answers; the same name answering again replaces their answer."""
    with _LOCK:
        data = _load()
        group = data.get(str(group_id))
        if not group:
            return None
        name = str(response.get("name") or "").strip().lower()
        responses = [r for r in group.get("responses") or [] if not name or str(r.get("name") or "").strip().lower() != name]
        responses.append({**response, "answered_at": time.time()})
        group["responses"] = responses
        _save(data)
        return group
