"""Recommendation agent: MCP menu search plus the feedback loop.

`build_recommendation_brief` is exported because `decision` calls it at the
moment it commits to recommending — the brief is the frozen hand-off between
the two packages, so it belongs on this package's public surface rather than
being reached for inside the module.
"""

from .RecommendAgent import (
    build_recommendation_brief,
    fetch_recommendations,
    is_pure_feedback,
    recommendation_agent_node,
)

__all__ = [
    "build_recommendation_brief",
    "fetch_recommendations",
    "is_pure_feedback",
    "recommendation_agent_node",
]
