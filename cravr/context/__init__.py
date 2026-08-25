"""Context discovery: delivery-side facts (location, weather).

Currently wired out of `graph.py` — the node is kept importable so reinstating
it is a one-line change rather than a resurrection.
"""

from .ContextAgent import context_discovery_node

__all__ = ["context_discovery_node"]
