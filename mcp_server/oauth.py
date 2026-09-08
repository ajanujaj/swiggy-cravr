"""Swiggy OAuth 2.1 + PKCE: the `/oauth/callback` route Swiggy redirects the
user back to once they approve, the `/oauth/login` link that starts that
approval, and small on-disk stores for the registered client and the token.

This is the *client* side of OAuth -- Cravr authenticating itself against
Swiggy's authorization server so it can call Swiggy's real Food/Instamart/
Dineout MCP tools once Builders Club grants production access. It is not
about Cravr protecting its own MCP server (that would be FastMCP's
`OAuthProxy`, a different tool for the reverse situation, and not what this
project needs).

The one URI Swiggy actually needs to whitelist is:

    {redirect_uri()} -- see SWIGGY_REDIRECT_URI below

Swiggy's Builders Club team confirmed directly that whitelisting the
redirect URI is the *only* setup needed on their side -- no manually issued
client_id, secret, or endpoint URLs. That's Dynamic Client Registration
(RFC 7591): Cravr discovers Swiggy's OAuth endpoints from its own metadata
(RFC 8414 / RFC 9728) and registers itself as a client on the fly, the
first time `/oauth/login` runs; the redirect URI is the actual security
boundary, which is exactly why it's the one thing that has to be
pre-approved. See `_bootstrap()` for the discovery + registration flow.

That means the only thing this file still needs from you is where to
discover from: SWIGGY_MCP_URL, the base URL of Swiggy's MCP server. The
`/oauth/callback` route itself is live the moment this is deployed
regardless -- it's `/oauth/login` that needs SWIGGY_MCP_URL set before it
can do anything.

`login()` and `callback()` are plain Starlette request-in, response-out
functions -- not tied to any particular server framework. `server.py`
(repo root) registers them directly on its FastAPI app, alongside the
mounted Chainlit chat UI.
"""

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

# Which PKCE verifier a given `state` belongs to, kept in memory only. A
# server restart mid-approval just forces the user to click "Connect to
# Swiggy" again -- an acceptable trade-off for a value that's dead the
# instant it's consumed anyway, and far simpler than persisting a secret
# that only needs to live for the ~seconds between redirect and callback.
_PENDING: dict[str, dict[str, Any]] = {}
_PENDING_TTL_SECONDS = 600


def _env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name, default)
    return value or None


def redirect_uri() -> str:
    """The exact URI to register with Swiggy as the OAuth redirect_uri.

    Defaults to this project's Render URL; override with SWIGGY_REDIRECT_URI
    if that domain ever changes (custom domain, a staging deploy, ...) --
    whatever this returns must match, character for character, what's
    whitelisted with Swiggy and what the authorize call and the client
    registration below both send.
    """
    return _env("SWIGGY_REDIRECT_URI") or "https://cravr-app.onrender.com/oauth/callback"


def _mcp_url() -> str | None:
    """Base URL of Swiggy's MCP server -- where OAuth discovery starts.
    Almost certainly the same URL you'll eventually point CRAVR_MCP_URL at
    once real tool calls replace the mock server; kept as its own env var
    for now since that cutover hasn't happened yet and this needs to be
    testable independently of it.
    """
    return _env("SWIGGY_MCP_URL")


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
    """Whatever `callback()` last stored -- access/refresh token, expiry,
    scope, exactly as Swiggy's token endpoint returned it, plus an
    `obtained_at` timestamp. `cravr/client.py` (or whatever eventually talks
    to Swiggy's real MCP servers) reads this to authenticate its calls."""
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
    """RFC 8414 (authorization server metadata) and RFC 9728 (protected
    resource metadata) both insert the well-known segment *before* the
    resource's own path, and both require an origin-root fallback too --
    so for https://mcp.swiggy.com/food, the metadata could be at either
    https://mcp.swiggy.com/.well-known/<name>/food (the path-aware form)
    or plain https://mcp.swiggy.com/.well-known/<name>. Try the more
    specific one first.
    """
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
    """Follow the MCP Authorization spec's discovery chain: the MCP
    resource's own protected-resource metadata names the authorization
    server(s) that issue tokens for it; that server's own metadata then
    gives the actual authorize/token/registration endpoints. Falls back to
    treating `mcp_url` itself as the authorization server if no protected-
    resource document is found (some gateways skip that indirection and
    are their own authorization server), and to OpenID Connect discovery
    if the OAuth-specific well-known name comes back empty -- same
    metadata shape, different conventional name, and a common fallback in
    practice.
    """
    resource_meta = await _fetch_first(http_client, _well_known_candidates(mcp_url, "oauth-protected-resource"))
    authz_server_url = mcp_url
    if resource_meta and resource_meta.get("authorization_servers"):
        authz_server_url = resource_meta["authorization_servers"][0]

    metadata = await _fetch_first(http_client, _well_known_candidates(authz_server_url, "oauth-authorization-server"))
    if not metadata:
        metadata = await _fetch_first(http_client, _well_known_candidates(authz_server_url, "openid-configuration"))
    if not metadata:
        raise RuntimeError(
            f"Could not discover OAuth metadata starting from {mcp_url} "
            f"(tried oauth-protected-resource there, then "
            f"oauth-authorization-server / openid-configuration on {authz_server_url})."
        )
    return metadata


async def _bootstrap(http_client: httpx.AsyncClient, mcp_url: str) -> dict[str, Any]:
    """Discover Swiggy's OAuth endpoints and register Cravr as a client if
    that hasn't happened yet (Dynamic Client Registration, RFC 7591 --
    this is what makes "just whitelist the redirect URI" true), caching
    the combined result so this only touches the network once. Called
    from both `login()` (which needs it to build the authorize URL) and
    `callback()` (normally just reads what `login()` already cached
    seconds earlier; redoes discovery only if that cache is somehow
    missing -- e.g. the very first callback right after a fresh deploy).
    """
    cached = _load_client()
    if cached and cached.get("client_id") and cached.get("metadata"):
        return cached

    metadata = await _discover_metadata(http_client, mcp_url)
    registration_endpoint = metadata.get("registration_endpoint")
    if not registration_endpoint:
        raise RuntimeError(
            "Swiggy's OAuth metadata has no registration_endpoint -- Dynamic "
            "Client Registration doesn't look supported here after all, so "
            "this will need a manually issued SWIGGY_CLIENT_ID."
        )

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
    """GET /oauth/login -- discovers Swiggy's OAuth endpoints, registers
    Cravr as a client if it hasn't already, and sends the browser to
    Swiggy's consent screen. See `_bootstrap()`.
    """
    _prune_expired_pending()
    mcp_url = _mcp_url()
    if not mcp_url:
        return HTMLResponse(
            "<p>SWIGGY_MCP_URL isn't set yet -- it's the one thing this "
            "still needs: the base URL of Swiggy's MCP server, used to "
            "discover its OAuth endpoints. The callback route is already "
            "live and whitelisted either way.</p>",
            status_code=503,
        )

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
    _PENDING[state] = {"verifier": verifier, "created_at": time.time()}

    query = urlencode(
        {
            "response_type": "code",
            "client_id": client["client_id"],
            "redirect_uri": redirect_uri(),
            "state": state,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            "scope": _env("SWIGGY_SCOPES", "") or "",
        }
    )
    return RedirectResponse(f"{authorize_endpoint}?{query}")


async def callback(request: Request):
    """GET /oauth/callback -- this is the URI Swiggy has whitelisted.
    Verifies `state` against a pending PKCE login, exchanges the
    authorization `code` for tokens, and stores them via `_save_tokens`.
    """
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

    mcp_url = _mcp_url()
    if not mcp_url:
        return HTMLResponse("<p>SWIGGY_MCP_URL isn't set -- see /oauth/login.</p>", status_code=503)

    async with httpx.AsyncClient(timeout=15) as http_client:
        try:
            client = await _bootstrap(http_client, mcp_url)
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
            f"<p>Swiggy's token endpoint rejected the exchange "
            f"({response.status_code}): {response.text}</p>",
            status_code=502,
        )

    _save_tokens(response.json())
    return HTMLResponse("<p>Swiggy is connected. You can close this tab.</p>")
