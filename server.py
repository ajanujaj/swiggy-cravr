"""Deploy entrypoint: one process serving the Cravr chat UI (Chainlit,
mounted at "/") and the Swiggy OAuth callback (`/oauth/login`,
`/oauth/callback`) side by side, on one port.

Why one process: Swiggy needs one whitelisted, stable, public URL for its
OAuth redirect; the chat UI needs a public URL too, and there's no reason
to pay for that twice. Chainlit supports mounting itself inside a plain
FastAPI app instead of always being the whole server
(`chainlit.utils.mount_chainlit`), and `mcp_server/oauth.py`'s `login` /
`callback` were already written as plain Starlette request-in,
response-out functions -- never actually tied to FastMCP, just registered
through it before -- so both fit on the same FastAPI app with no changes
to either one. One Render service, one URL, one thing to whitelist.

The mock-data MCP tool server (`mcp_server/run_mcp.py`) is deliberately
NOT part of this app. Nothing outside Cravr's own backend ever calls it --
only `cravr/client.py`, over HTTP via `CRAVR_MCP_URL` -- so it stays its
own separate, easily-removable process/service: once real Swiggy MCP
access exists, stop deploying that one and point `CRAVR_MCP_URL` at
Swiggy's real endpoint. Nothing here needs to change for that swap.

Run locally (repo root) -- note the explicit port. The mock MCP tool
server (`mcp_server/run_mcp.py`) and uvicorn's own CLI default both land on
8000, so left unset here they silently collide: whichever process starts
second either fails to bind, or ends up fielding the other's traffic
(the mock server's real symptom the first time this happened: MCP calls
that should hit `mcp_server/run_mcp.py` landed on *this* app instead, which
has no `/mcp` route of its own, so every one came back 422):
    uvicorn server:app --reload --port 8001
Then open http://localhost:8001 -- not 8000, that's the mock server.

Deploy (Render or any host that runs a plain ASGI app -- each service gets
its own dedicated port there, so this collision only happens locally):
    uvicorn server:app --host 0.0.0.0 --port $PORT
"""

from __future__ import annotations

import os

from chainlit.utils import mount_chainlit
from fastapi import FastAPI

from mcp_server import oauth

app = FastAPI(title="Cravr")

# Registered directly on this app (not through the Chainlit mount below) --
# see docs.chainlit.io/integrations/fastapi: routes defined on the parent
# app keep working normally alongside a mounted Chainlit app.
app.get("/oauth/login")(oauth.login)
app.get("/oauth/callback")(oauth.callback)

# `target` is resolved relative to the working directory the process is
# started from -- the run/deploy commands above both assume the repo root.
mount_chainlit(app=app, target="UI/chainlit_app.py", path="/")


if __name__ == "__main__":
    # `python server.py` as a no-reload alternative to the uvicorn CLI --
    # same 8001 default as the documented `uvicorn ... --port 8001`, so
    # running it this way can't accidentally fall back to 8000 and collide
    # with mcp_server/run_mcp.py again. Render overrides this via $PORT.
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
