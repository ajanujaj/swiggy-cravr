"""Group orders: one poll per group (storage: see cravr/storage.py)."""

from __future__ import annotations

import os
import secrets
import time
from pathlib import Path
from typing import Any

from .. import storage

_DEFAULT_PATH = Path(__file__).resolve().parents[2] / "mcp_server" / "data" / "groups.json"
_NAMESPACE = "groups"
_MAX_AGE_SECONDS = 2 * 24 * 3600  # a poll is only good for two days


def _path() -> Path:
    return Path(os.getenv("CRAVR_GROUPS_PATH") or _DEFAULT_PATH)


def _fresh(group: dict[str, Any] | None) -> bool:
    return bool(group) and time.time() - float(group.get("created_at") or 0) < _MAX_AGE_SECONDS


def create_group(size: int, label: str = "") -> dict[str, Any]:
    group = {
        "id": secrets.token_urlsafe(6),
        "size": max(2, min(int(size or 2), 20)),
        "label": label,
        "created_at": time.time(),
        "responses": [],
    }
    storage.prune_json(_path(), _fresh)
    storage.update(_NAMESPACE, group["id"], _path(), lambda _current: group)
    return group


def get_group(group_id: str) -> dict[str, Any] | None:
    group = storage.get(_NAMESPACE, str(group_id), _path())
    return group if _fresh(group) else None


def add_response(group_id: str, response: dict[str, Any]) -> dict[str, Any] | None:
    """Save one person's answers; the same name answering again replaces their answer."""
    name = str(response.get("name") or "").strip().lower()

    def change(group: dict[str, Any] | None) -> dict[str, Any] | None:
        if not _fresh(group):
            return None
        responses = [r for r in group.get("responses") or []
                     if not name or str(r.get("name") or "").strip().lower() != name]
        responses.append({**response, "answered_at": time.time()})
        return {**group, "responses": responses}

    return storage.update(_NAMESPACE, str(group_id), _path(), change)
