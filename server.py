"""Deploy entrypoint: one process serving the Cravr chat UI (Chainlit,
mounted at "/") and the Swiggy OAuth callback (/oauth/login,
/oauth/callback) on one port -- Swiggy needs one stable public URL for
its redirect, so there's no reason to run two services for that.

The mock MCP tool server (mcp_server/run_mcp.py) is NOT part of this app --
it's a separate process, reached over HTTP via CRAVR_MCP_URL.

Run locally (note the explicit port -- the mock server also defaults to
8000, so this must not collide with it):
    uvicorn server:app --reload --port 8001

Deploy:
    uvicorn server:app --host 0.0.0.0 --port $PORT
"""

from __future__ import annotations

import os

from chainlit.utils import mount_chainlit
from fastapi import FastAPI

from mcp_server import oauth

app = FastAPI(title="Cravr")

app.get("/oauth/login")(oauth.login)
app.get("/oauth/callback")(oauth.callback)

mount_chainlit(app=app, target="UI/chainlit_app.py", path="/")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
