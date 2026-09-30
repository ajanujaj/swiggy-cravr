"""Small key-value storage for Cravr's own data (taste profiles, group polls).

- `DATABASE_URL=postgres://...`  -> Postgres (e.g. Render Postgres). Survives
  redeploys and restarts -- use this in production.
- `DATABASE_URL=sqlite:///path.db` -> a local SQLite file.
- unset -> one JSON file per namespace (fine locally; on Render's free tier
  the disk is wiped on every redeploy).

Values are JSON objects, stored under (namespace, key).
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from pathlib import Path
from typing import Any, Callable

from .debug_utils import debug

_LOCK = threading.Lock()
_TABLE_READY: set[str] = set()


def _url() -> str:
    return (os.getenv("DATABASE_URL") or "").strip()


def backend_name() -> str:
    url = _url()
    if url.startswith(("postgres://", "postgresql://")):
        return "postgres"
    if url.startswith("sqlite:///"):
        return "sqlite"
    return "json"


# ---------------------------------------------------------------- SQL backends

def _connect():
    url = _url()
    if backend_name() == "postgres":
        import psycopg  # only needed when a Postgres URL is configured

        return psycopg.connect(url.replace("postgres://", "postgresql://", 1), autocommit=True), "%s"
    return sqlite3.connect(url[len("sqlite:///"):]), "?"


def _ensure_table(conn, kind: str) -> None:
    if _url() in _TABLE_READY:
        return
    value_type = "JSONB" if kind == "postgres" else "TEXT"
    conn.execute(
        f"CREATE TABLE IF NOT EXISTS cravr_kv (namespace TEXT NOT NULL, key TEXT NOT NULL, "
        f"value {value_type} NOT NULL, updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, "
        f"PRIMARY KEY (namespace, key))"
    )
    _TABLE_READY.add(_url())


def _sql_get(namespace: str, key: str) -> dict[str, Any] | None:
    conn, p = _connect()
    try:
        _ensure_table(conn, backend_name())
        row = conn.execute(f"SELECT value FROM cravr_kv WHERE namespace = {p} AND key = {p}", (namespace, key)).fetchone()
    finally:
        conn.close()
    if not row:
        return None
    value = row[0]
    return json.loads(value) if isinstance(value, str) else value


def _sql_put(namespace: str, key: str, value: dict[str, Any]) -> None:
    conn, p = _connect()
    try:
        _ensure_table(conn, backend_name())
        value_param = f"{p}::jsonb" if backend_name() == "postgres" else p  # text -> jsonb needs an explicit cast
        conn.execute(
            f"INSERT INTO cravr_kv (namespace, key, value) VALUES ({p}, {p}, {value_param}) "
            f"ON CONFLICT (namespace, key) DO UPDATE SET value = excluded.value, updated_at = CURRENT_TIMESTAMP",
            (namespace, key, json.dumps(value, ensure_ascii=False)),
        )
        if backend_name() == "sqlite":
            conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------- JSON backend

def _json_load(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8") or "{}")
    except (OSError, json.JSONDecodeError) as exc:
        debug("storage", f"could not read {path} ({exc}) -- treating as empty")
        return {}
    return data if isinstance(data, dict) else {}


def _json_save(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)  # atomic


# ---------------------------------------------------------------- public API

def get(namespace: str, key: str, json_path: Path) -> dict[str, Any] | None:
    with _LOCK:
        if backend_name() == "json":
            value = _json_load(json_path).get(key)
            return value if isinstance(value, dict) else None
        return _sql_get(namespace, key)


def update(namespace: str, key: str, json_path: Path,
           change: Callable[[dict[str, Any] | None], dict[str, Any] | None]) -> dict[str, Any] | None:
    """Read-modify-write one record under the lock. `change` returns the new value
    (or None to leave it untouched)."""
    with _LOCK:
        if backend_name() == "json":
            data = _json_load(json_path)
            new = change(data.get(key) if isinstance(data.get(key), dict) else None)
            if new is not None:
                data[key] = new
                _json_save(json_path, data)
            return new
        new = change(_sql_get(namespace, key))
        if new is not None:
            _sql_put(namespace, key, new)
        return new


def prune_json(json_path: Path, keep: Callable[[dict[str, Any]], bool]) -> None:
    """Drop stale records from a JSON-backed namespace (SQL rows are tiny; left alone)."""
    if backend_name() != "json":
        return
    with _LOCK:
        data = _json_load(json_path)
        kept = {k: v for k, v in data.items() if isinstance(v, dict) and keep(v)}
        if len(kept) != len(data):
            _json_save(json_path, kept)
