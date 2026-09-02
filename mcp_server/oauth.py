"""Swiggy OAuth 2.1 + PKCE: the `/oauth/callback` route Swiggy redirects the
user back to once they approve, the `/oauth/login` link that starts that
approval, and a small on-disk token store.

This is the *client* side of OAuth -- Cravr authenticating itself against
Swiggy's authorization server so it can call Swiggy's real Food/Instamart/
Dineout MCP tools once Builders Club grants production access. It is not
about Cravr protecting its own MCP server (that would be FastMCP's
`OAuthProxy`, a different tool for the reverse situation, and not what this
project needs).

The one URI Swiggy actually needs to whitelist is:

    {redirect_uri()} -- see SWIGGY_REDIRECT_URI below

That route is live the moment this server is deployed, independent of
whether real Swiggy credentials exist yet -- `/oauth/login` and the token
exchange in `callback()` degrade to a clear "not configured" response until
SWIGGY_CLIENT_ID / SWIGGY_AUTHORIZE_URL / SWIGGY_TOKEN_URL are set, so
whitelisting doesn't have to wait on the rest of the credentials.

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
from urllib.parse import urlencode

import httpx
from starlette.requests import Request
from starlette.responses import HTMLResponse, RedirectResponse

TOKENS_PATH = Path(__file__).resolve().parent / "data" / "swiggy_tokens.json"

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
    submitted to Swiggy and what Swiggy's authorize call is sent.
    """
    return _env("SWIGGY_REDIRECT_URI") or "https://cravr-swiggy-mcp.onrender.com/oauth/callback"


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


async def login(request: Request):
    """GET /oauth/login -- sends the browser to Swiggy's consent screen.

    Not needed for whitelisting itself (only `/oauth/callback` is), but
    wires the loop shut so the whole connect-to-Swiggy flow is testable the
    moment real credentials exist, instead of only after someone builds a
    second thing to kick it off.
    """
    _prune_expired_pending()
    authorize_url = _env("SWIGGY_AUTHORIZE_URL")
    client_id = _env("SWIGGY_CLIENT_ID")
    if not authorize_url or not client_id:
        return HTMLResponse(
            "<p>Swiggy OAuth isn't configured yet -- set SWIGGY_AUTHORIZE_URL "
            "and SWIGGY_CLIENT_ID once Builders Club grants production "
            "credentials. The callback endpoint is already live and can be "
            "whitelisted in the meantime.</p>",
            status_code=503,
        )

    state = secrets.token_urlsafe(24)
    verifier, challenge = _new_pkce_pair()
    _PENDING[state] = {"verifier": verifier, "created_at": time.time()}

    query = urlencode(
        {
            "response_type": "code",
            "client_id": client_id,
            "redirect_uri": redirect_uri(),
            "state": state,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            "scope": _env("SWIGGY_SCOPES", "") or "",
        }
    )
    return RedirectResponse(f"{authorize_url}?{query}")


async def callback(request: Request):
    """GET /oauth/callback -- this is the URI to submit to Swiggy for
    whitelisting. Verifies `state` against a pending PKCE login, exchanges
    the authorization `code` for tokens, and stores them via `_save_tokens`.
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

    token_url = _env("SWIGGY_TOKEN_URL")
    client_id = _env("SWIGGY_CLIENT_ID")
    if not token_url or not client_id:
        return HTMLResponse(
            "<p>Got Swiggy's authorization code, but SWIGGY_TOKEN_URL / "
            "SWIGGY_CLIENT_ID aren't configured yet, so it can't be "
            "exchanged for a token.</p>",
            status_code=503,
        )

    data = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": redirect_uri(),
        "client_id": client_id,
        "code_verifier": pending["verifier"],
    }
    client_secret = _env("SWIGGY_CLIENT_SECRET")
    if client_secret:
        data["client_secret"] = client_secret

    async with httpx.AsyncClient(timeout=15) as http_client:
        response = await http_client.post(token_url, data=data)

    if response.status_code >= 400:
        return HTMLResponse(
            f"<p>Swiggy's token endpoint rejected the exchange "
            f"({response.status_code}): {response.text}</p>",
            status_code=502,
        )

    _save_tokens(response.json())
    return HTMLResponse("<p>Swiggy is connected. You can close this tab.</p>")

