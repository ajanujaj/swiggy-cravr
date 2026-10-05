from __future__ import annotations

import base64
import hashlib
import json
import os
import secrets
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlencode, urlparse

import httpx
from starlette.requests import Request
from starlette.responses import HTMLResponse, RedirectResponse

TOKENS_PATH = Path(__file__).resolve().parent / "data" / "swiggy_tokens.json"
CLIENT_PATH = Path(__file__).resolve().parent / "data" / "swiggy_client.json"

_PENDING: dict[str, dict[str, Any]] = {}
_PENDING_TTL_SECONDS = 600


def _env(name: str, default: str | None = None) -> str | None:
    return os.environ.get(name, default) or None


def redirect_uri() -> str:
    return _env("SWIGGY_REDIRECT_URI") or "https://cravr-app.onrender.com/oauth/callback"


def _mcp_url() -> str:
    return _env("SWIGGY_MCP_URL") or "https://mcp.swiggy.com"


def _new_pkce_pair() -> tuple[str, str]:
    verifier = base64.urlsafe_b64encode(secrets.token_bytes(40)).rstrip(b"=").decode()
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    return verifier, challenge


def _prune_expired_pending() -> None:
    cutoff = time.time() - _PENDING_TTL_SECONDS
    for state, entry in list(_PENDING.items()):
        if entry["created_at"] < cutoff:
            _PENDING.pop(state, None)


def load_tokens() -> dict[str, Any] | None:
    if not TOKENS_PATH.exists():
        return None
    return json.loads(TOKENS_PATH.read_text())


def _save_tokens(payload: dict[str, Any]) -> None:
    TOKENS_PATH.parent.mkdir(parents=True, exist_ok=True)
    TOKENS_PATH.write_text(json.dumps({**payload, "obtained_at": time.time()}, indent=2))


def _load_client() -> dict[str, Any] | None:
    if not CLIENT_PATH.exists():
        return None
    return json.loads(CLIENT_PATH.read_text())


def _save_client(payload: dict[str, Any]) -> None:
    CLIENT_PATH.parent.mkdir(parents=True, exist_ok=True)
    CLIENT_PATH.write_text(json.dumps(payload, indent=2))


def _well_known_candidates(base_url: str, well_known_name: str) -> list[str]:
    parsed = urlparse(base_url)
    origin = f"{parsed.scheme}://{parsed.netloc}"
    path = parsed.path.rstrip("/")
    candidates = [f"{origin}/.well-known/{well_known_name}{path}"] if path else []
    candidates.append(f"{origin}/.well-known/{well_known_name}")
    return candidates


async def _fetch_first(http_client: httpx.AsyncClient, urls: list[str]) -> dict[str, Any] | None:
    for url in urls:
        try:
            response = await http_client.get(url, headers={"Accept": "application/json"})
        except httpx.HTTPError:
            continue
        if response.status_code == 200:
            try:
                return response.json()
            except ValueError:
                continue
    return None


async def _discover_metadata(http_client: httpx.AsyncClient, mcp_url: str) -> dict[str, Any]:
    resource_meta = await _fetch_first(http_client, _well_known_candidates(mcp_url, "oauth-protected-resource"))
    authz_server_url = mcp_url
    if resource_meta and resource_meta.get("authorization_servers"):
        authz_server_url = resource_meta["authorization_servers"][0]

    metadata = await _fetch_first(http_client, _well_known_candidates(authz_server_url, "oauth-authorization-server"))
    if not metadata:
        metadata = await _fetch_first(http_client, _well_known_candidates(authz_server_url, "openid-configuration"))
    if not metadata:
        raise RuntimeError(f"Could not discover OAuth metadata from {mcp_url} / {authz_server_url}.")
    return metadata


async def _bootstrap(http_client: httpx.AsyncClient, mcp_url: str) -> dict[str, Any]:
    cached = _load_client()
    if cached and cached.get("client_id") and cached.get("metadata"):
        return cached

    metadata = await _discover_metadata(http_client, mcp_url)
    registration_endpoint = metadata.get("registration_endpoint")
    if not registration_endpoint:
        raise RuntimeError("Swiggy's OAuth metadata has no registration_endpoint.")

    response = await http_client.post(
        registration_endpoint,
        json={
            "client_name": "Cravr",
            "redirect_uris": [redirect_uri()],
            "grant_types": ["authorization_code", "refresh_token"],
            "response_types": ["code"],
            "token_endpoint_auth_method": "none",
        },
        headers={"Accept": "application/json"},
    )
    if response.status_code >= 400:
        raise RuntimeError(f"Client registration rejected ({response.status_code}): {response.text}")

    registration = response.json()
    if not registration.get("client_id"):
        raise RuntimeError(f"Registration response had no client_id: {registration}")

    client = {**registration, "metadata": metadata}
    _save_client(client)
    return client


async def login(request: Request):
    _prune_expired_pending()
    mcp_url = _mcp_url()
    next_path = request.query_params.get("next") or "/"

    async with httpx.AsyncClient(timeout=15) as http_client:
        try:
            client = await _bootstrap(http_client, mcp_url)
        except RuntimeError as exc:
            return HTMLResponse(f"<p>{exc}</p>", status_code=502)

    authorize_endpoint = client["metadata"].get("authorization_endpoint")
    if not authorize_endpoint:
        return HTMLResponse("<p>Swiggy's OAuth metadata had no authorization_endpoint.</p>", status_code=502)

    state = secrets.token_urlsafe(24)
    verifier, challenge = _new_pkce_pair()
    _PENDING[state] = {"verifier": verifier, "created_at": time.time(), "next": next_path}

    query = urlencode(
        {
            "response_type": "code",
            "client_id": client["client_id"],
            "redirect_uri": redirect_uri(),
            "state": state,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            "scope": _env("SWIGGY_SCOPES") or "mcp:tools mcp:resources mcp:prompts",
        }
    )
    return RedirectResponse(f"{authorize_endpoint}?{query}")


async def callback(request: Request):
    error = request.query_params.get("error")
    if error:
        return HTMLResponse(f"<p>Swiggy declined authorization: {error}</p>", status_code=400)

    code = request.query_params.get("code")
    state = request.query_params.get("state")
    if not code or not state:
        return HTMLResponse("<p>Missing `code` or `state` on the callback request.</p>", status_code=400)

    _prune_expired_pending()
    pending = _PENDING.pop(state, None)
    if not pending:
        return HTMLResponse(
            "<p>This login link expired or was already used -- start again at /oauth/login.</p>",
            status_code=400,
        )

    async with httpx.AsyncClient(timeout=15) as http_client:
        try:
            client = await _bootstrap(http_client, _mcp_url())
        except RuntimeError as exc:
            return HTMLResponse(f"<p>{exc}</p>", status_code=502)

        token_endpoint = client["metadata"].get("token_endpoint")
        if not token_endpoint:
            return HTMLResponse("<p>Swiggy's OAuth metadata had no token_endpoint.</p>", status_code=502)

        data = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": redirect_uri(),
            "client_id": client["client_id"],
            "code_verifier": pending["verifier"],
        }
        if client.get("client_secret"):
            data["client_secret"] = client["client_secret"]

        response = await http_client.post(token_endpoint, data=data)

    if response.status_code >= 400:
        return HTMLResponse(
            f"<p>Swiggy's token endpoint rejected the exchange ({response.status_code}): {response.text}</p>",
            status_code=502,
        )

    _save_tokens(response.json())
    return RedirectResponse(pending.get("next") or "/")
