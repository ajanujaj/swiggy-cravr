"""Streamlit chat UI for Cravr -- a thin presentation layer over `cravr.graph.app`.

Split by concern rather than one file:
  state.py         -- st.session_state setup/reset
  graph_bridge.py   -- the only module that talks to cravr.graph.app
  styles.py         -- CSS
  components.py     -- chat bubbles, option chips, dish cards
  streamlit_app.py  -- entry point, wires the above together
"""
