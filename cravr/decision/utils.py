# `decision` value telling graph.py to route back to intent_discovery (not END) after a
# clarification round, so the answer is folded in by the normal graph nodes.
CLARIFY_AND_RETRY = "clarify_and_retry"

# Upper bound on clarification round-trips (tracked in graph state) before giving up gracefully.
MAX_CLARIFICATION_ROUNDS = 3

# Fallback question used once the budget is spent. Must never be the question just given up on --
# re-emitting it invites another answer as a fresh turn, restarting the whole streak (users
# answering "vegetarian" six times in a row). Doubles as the pseudo-field name for that
# question in the "never ask the same thing twice" bookkeeping.
GENERIC_CLARIFICATION_FIELD = "__generic__"
