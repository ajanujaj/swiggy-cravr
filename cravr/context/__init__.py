"""Context discovery: delivery-side facts (location, delivery area) enriched
after user_preference, before intent discovery. See graph.py."""

from .ContextAgent import context_discovery_node

__all__ = ["context_discovery_node"]
