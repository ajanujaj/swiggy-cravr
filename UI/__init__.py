"""Chainlit chat UI for Cravr -- a thin presentation layer over `cravr.graph.app`.

Split by concern rather than one file:
  state.py          -- cl.user_session setup/reset
  graph_bridge.py   -- the only module that talks to cravr.graph.app
  components.py     -- chat message builders: option buttons, dish cards
  chainlit_app.py   -- entry point (Chainlit lifecycle hooks), wires the above together

Mounted into `server.py` (repo root) alongside the Swiggy OAuth callback --
see that file for why, and for how to run this UI on its own.
"""
