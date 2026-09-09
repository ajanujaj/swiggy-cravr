"""Chainlit chat UI for Cravr -- a thin presentation layer over `cravr.graph.app`.

  state.py          -- cl.user_session setup/reset
  graph_bridge.py   -- the only module that talks to cravr.graph.app
  components.py     -- chat message builders: option buttons, dish cards
  chainlit_app.py   -- entry point, wires the above together

Mounted into `server.py` (repo root) alongside the Swiggy OAuth callback.
"""
