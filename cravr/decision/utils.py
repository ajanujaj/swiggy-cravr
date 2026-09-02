# `decision` value that tells graph.py to route back to intent_discovery
# (rather than END) after a clarification round, so the human's answer gets
# folded into the profiles by the normal, reliably-committed graph nodes.
CLARIFY_AND_RETRY = "clarify_and_retry"

# Upper bound on how many clarification round-trips (tracked in graph state,
# not a local loop) we'll ask for before giving up gracefully.
MAX_CLARIFICATION_ROUNDS = 3

# Used when the clarification budget is spent. It must NOT be the question we
# just gave up on: re-emitting that question invited another answer, which
# arrived as a fresh turn and restarted the whole streak -- the loop that had
# users answering "vegetarian" six times in a row.
# Pseudo-field for the catch-all question, so it participates in the
# "never ask the same thing twice" bookkeeping like any named gap.
GENERIC_CLARIFICATION_FIELD = "__generic__" 
