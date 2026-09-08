# Cravr

The active `cravr/` package is a minimal LangChain + LangGraph chat workflow. It stores conversation messages with LangGraph's `add_messages` reducer, calls Groq's hosted API in the chatbot node, and ends after one response.

## Run

Use Python 3.11 or 3.12 for the chat UI. Chainlit's own dependencies (via
`flask_socketio` -> `gevent`) don't yet have prebuilt wheels for very new
Python releases like 3.14 -- Chainlit's maintainers have closed that as
"not planned" for now, so `pip install` will fail trying to compile
`gevent` from source on 3.14 rather than it being anything wrong with this
project. If you're on Anaconda: `conda create -n cravr python=3.11 -y &&
conda activate cravr` before the steps below (and again in every new
terminal you use for this project).

```bash
pip install -r requirements.txt
# set GROQ_API_KEY (or groq_apikey) in .env -- free key at https://console.groq.com/keys
python -m cravr.run
python -m unittest discover -s tests -v
CRAVR_DEBUG=1 python cravr/run.py  ## debug model
```

Also start the mock MCP server (`python mcp_server/run_mcp.py`) before running the CLI, since `historyAgent`/`RecommendAgent` call it over HTTP.

## Run the chat UI

A [Chainlit](https://docs.chainlit.io) chat UI is available as an alternative to the CLI -- same backend, same graph, just chat bubbles, tappable option buttons, and dish cards instead of a terminal prompt. It's a Python-only UI: no HTML/CSS/JS to write or touch.

```bash
pip install -r requirements.txt
# set GROQ_API_KEY (or groq_apikey) in .env -- free key at https://console.groq.com/keys
./run_mock_mcp.sh   # terminal 1 -- leave running (mock tool data), port 8000
./run_ui.sh          # terminal 2 -- the chat UI + Swiggy OAuth callback, port 8001
```

Then open **http://localhost:8001** -- not 8000, that's the mock MCP
server, and it has no page to show you.

Use these two scripts rather than typing the raw `python`/`uvicorn`
commands by hand: both processes want port 8000 by default (uvicorn's own
CLI default is 8000, same as the mock server), so a hand-typed command
that forgets `--port 8001` silently fights the mock server for the same
port instead of erroring clearly -- whichever starts second either fails
to bind, or ends up fielding the *other's* traffic (symptom: the chat UI
loads, but every message logs `POST /mcp ... 422 Unprocessable Entity`,
because MCP calls meant for `mcp_server/run_mcp.py` landed on `server.py`
instead). `run_ui.sh` hardcodes `--port 8001` so that mistake isn't
possible. If you'd rather run the commands directly, they're just:

```bash
python mcp_server/run_mcp.py               # terminal 1, port 8000
uvicorn server:app --reload --port 8001    # terminal 2 -- the --port is not optional
```

`server.py` is the entry point, not `UI/chainlit_app.py` directly -- it mounts the Chainlit UI alongside the Swiggy OAuth callback routes on one process (see "Deploying" below for why). You can still run the UI on its own, without the OAuth routes, for quick local iteration (same port-collision reasoning applies -- keep it off 8000 too):

```bash
chainlit run UI/chainlit_app.py -w --port 8001   # -w reloads on save -- same port note as above applies
```

To see the same `[debug:...]` traces the CLI prints, set `CRAVR_DEBUG=1` in the
**same terminal you started the server from** (not the mock MCP server's
terminal, and not the browser) -- logs only ever go to that terminal's
stderr, never to the page itself:

```bash
CRAVR_DEBUG=1 uvicorn server:app --reload --port 8001
```

Every new chat session also prints a one-line `[Cravr UI] new session
started ...` marker on startup regardless of `CRAVR_DEBUG` -- if you don't see
even that in the terminal, the terminal itself isn't the one the server is
attached to. Chainlit's own "new chat" button (built into its UI) starts a
fresh session -- and therefore a fresh conversation thread -- the same way
reloading used to.

Environment variables:
- `CRAVR_MODEL` — override the Groq model id (must support tool calling, e.g. `openai/gpt-oss-120b` [default], `qwen/qwen3.6-27b`). `llama-3.3-70b-versatile` was retired by Groq on 2026-08-16. See https://console.groq.com/docs/models.
- `GROQ_API_KEY` (or `groq_apikey`) — your Groq API key, read from `.env`. Required.
- `CRAVR_DEBUG=1` — print each agent's raw LLM output, merged profile, and routing decisions to stderr.
- `CRAVR_MCP_URL` — override the mock MCP server URL (defaults to `http://127.0.0.1:8000/mcp`).

## Architecture: three independent pieces

- **`cravr/`** — the LangGraph agent (intent discovery, decision, recommendation) and the Groq LLM client. Framework-agnostic: doesn't know or care whether it's driven by the CLI, the Chainlit UI, or a test.
- **`UI/` + `server.py`** — the chat UI (Chainlit) and the Swiggy OAuth callback, deployed together as one service (see below).
- **`mcp_server/`** — a standalone mock-data MCP tool server, standing in for Swiggy's real Food/Instamart/Dineout MCP tools while those aren't wired up yet. Nothing else in the repo imports it directly; `cravr/client.py` only ever reaches it over HTTP, through the `CRAVR_MCP_URL` env var. That's deliberate: once real Swiggy MCP access exists, removing this piece is "stop deploying it, point `CRAVR_MCP_URL` at Swiggy's real endpoint" -- no other code changes.

## Deploying

Both services need Python 3.12, same as local dev (see "Run" above) -- Render's own default (3.14.3 as of writing) hits the exact same Chainlit/gevent build failure documented there. A `.python-version` file at the repo root pins this for Render automatically; nothing else to configure for it.

Two Render services (see `render.yaml` -- deploy both via Render's "New +" -> "Blueprint" pointing at this repo, or create them by hand with the settings below):

**`cravr-app`** — the chat UI and the Swiggy OAuth callback, combined into one process by `server.py`. This is the service whose URL matters to Swiggy.

```
build command: pip install -r requirements.txt
start command: uvicorn server:app --host 0.0.0.0 --port $PORT
```

**`cravr-mock-mcp`** — the standalone mock tool server. Never needs to be public-facing in principle (only `cravr-app`'s own backend code calls it), but Render doesn't offer a private-only free web service, so it ends up with a public URL too -- that's fine, it only ever serves read-only mock menu data.

```
build command: pip install -r requirements.txt
start command: python mcp_server/run_mcp.py
```

Once both are deployed, set `cravr-app`'s `CRAVR_MCP_URL` env var to `cravr-mock-mcp`'s real URL + `/mcp` (e.g. `https://cravr-mock-mcp.onrender.com/mcp`) -- it defaults to `http://127.0.0.1:8000/mcp`, which only works when both processes share a machine. Skipping this makes the chat UI load fine but fail every tool call (recommendations, order history, preferences).

### Getting a URL for Swiggy to whitelist

Swiggy's Builders Club whitelists an OAuth **redirect URI**, not an IP or a
generic domain -- so what Swiggy needs from you is one exact, stable,
HTTPS URL: `<cravr-app's real URL>/oauth/callback`. `server.py` serves that
route directly (see `mcp_server/oauth.py`), right alongside the chat UI --
one deployed service, no separate app needed just for OAuth.

Render assigns each service's live `*.onrender.com` URL from the name you
give it at creation time, and that name has to be globally unique across
every Render account -- `cravr-app` may already be taken by someone else,
in which case Render appends a random suffix and your real URL won't match
that guess. **Create the `cravr-app` service first, copy the URL Render
actually shows you, and submit that exact URL** (with `/oauth/callback`
appended) to Swiggy for whitelisting -- not a guessed one. If your URL
isn't exactly `https://cravr-app.onrender.com`, also set `cravr-app`'s
`SWIGGY_REDIRECT_URI` env var to match, so `/oauth/login` sends Swiggy the
same URI you registered.

`/oauth/callback` is live the moment `cravr-app` is deployed, independent
of whether `/oauth/login` can do anything yet -- it's why it was safe to
submit for whitelisting right away.

Builders Club confirmed that whitelisting the redirect URI above is the
**entire** setup on their side -- no client_id, secret, authorize URL, or
token URL gets issued manually. That's Dynamic Client Registration (RFC
7591): `/oauth/login` discovers Swiggy's OAuth endpoints itself from
standard `.well-known` metadata (RFC 8414 / RFC 9728) and registers Cravr
as a client on the fly the first time it runs, caching the result in
`mcp_server/data/swiggy_client.json` (also `.gitignore`d). See
`mcp_server/oauth.py` for the discovery + registration + PKCE flow.

Swiggy's docs (https://mcp.swiggy.com/builders/docs/start/authenticate/)
confirm the base MCP URL is `https://mcp.swiggy.com`, discovery lives at
its `.well-known/oauth-authorization-server` / `.well-known/oauth-protected-resource`
paths, and registration is Dynamic Client Registration (RFC 7591) via
`POST /auth/register` -- `mcp_server/oauth.py` hardcodes that base URL as
its default, so no env var is required to make this work. Two are
available only as overrides, both optional:

- `SWIGGY_MCP_URL` — only if Swiggy's base URL ever changes, or you want
  to point at a sandbox. Defaults to `https://mcp.swiggy.com`.
- `SWIGGY_REDIRECT_URI` — only needed if your Render URL isn't
  `https://cravr-app.onrender.com/oauth/callback`; must exactly match
  what's whitelisted.
- `SWIGGY_SCOPES` — space-separated. Defaults to Swiggy's three documented
  scopes: `mcp:tools mcp:resources mcp:prompts`.

Visiting `/oauth/login` discovers Swiggy's endpoints, registers Cravr as a
client (first time only, cached in `mcp_server/data/swiggy_client.json`),
and starts the PKCE flow; Swiggy redirects back to `/oauth/callback`, which
exchanges the code for a token and writes it to
`mcp_server/data/swiggy_tokens.json` (both already `.gitignore`d -- never
commit them). Swiggy's docs note access tokens last 5 days and refresh
tokens aren't issued yet in v1 -- expiry just means going through
`/oauth/login` again, nothing to build for that yet. That token isn't read
by anything else yet -- getting the callback deployed and whitelisted now
just means the eventual switch to Swiggy's real MCP tools won't be
blocked on infrastructure later.

Both Render free-tier services spin down when idle and cold-start on the
next request (tens of seconds) -- so the first message after either has
been quiet for a while will be noticeably slower than the rest. Not a bug,
just the free tier; a paid instance removes it if it becomes annoying.
