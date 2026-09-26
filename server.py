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
    if request.method == "GET" and request.url.path == "/" and not oauth.load_tokens():
        return RedirectResponse(f"/oauth/login?next={request.url.path}")
    return await call_next(request)


mount_chainlit(app=app, target="UI/chainlit_app.py", path="/")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
