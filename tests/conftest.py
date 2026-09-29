import os

# The dishes-first opening searches Swiggy/the mock server. Tests written for the
# question flow run without it; opening tests turn it back on explicitly.
os.environ.setdefault("CRAVR_DISHES_FIRST", "0")
