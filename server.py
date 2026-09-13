"""Deploy entrypoint: one process serving the Cravr chat UI (Chainlit, mounted
at "/") and the Swiggy OAuth callback (/oauth/login, /oauth/callback) on one
port -- Swiggy needs one stable public URL for its redirect.

The mock MCP tool server (mcp_server/run_mcp.py) is a separate process,
reached over HTTP via CRAVR_MCP_URL, not part of this app.

Run locally (explicit port -- the mock server also defaults to 8000):
    uvicorn server:app --reload --port 8001

Deploy:
    uvicorn server:app --host 0.0.0.0 --port $PORT
"""

from __future__ import annotations

import os

from chainlit.utils import mount_chainlit
from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse

from mcp_server import oauth

app = FastAPI(title="Cravr")

app.get("/oauth/login")(oauth.login)
app.get("/oauth/callback")(oauth.callback)


@app.middleware("http")
async def require_swiggy_login(request: Request, call_next):
    """Bounce the root page to /oauth/login when no Swiggy token is on disk
    yet (e.g. right after a fresh deploy, whose filesystem starts empty) --
    oauth.callback sends the user straight back to `next` once login
    succeeds, so visiting this app's URL is the only manual step left."""
    if request.method == "GET" and request.url.path == "/" and not oauth.load_tokens():
        return RedirectResponse(f"/oauth/login?next={request.url.path}")
    return await call_next(request)


mount_chainlit(app=app, target="UI/chainlit_app.py", path="/")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
