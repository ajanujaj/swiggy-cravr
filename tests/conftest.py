import os

# The dishes-first opening searches Swiggy/the mock server. Tests written for the
# question flow run without it; opening tests turn it back on explicitly.
os.environ.setdefault("CRAVR_DISHES_FIRST", "0")

import tempfile

# Picks and rejections are learned into the preference store -- keep tests out of the real file.
os.environ.setdefault("CRAVR_PREFS_PATH", os.path.join(tempfile.mkdtemp(prefix="cravr-prefs-"), "prefs.json"))
os.environ.setdefault("CRAVR_GROUPS_PATH", os.path.join(tempfile.mkdtemp(prefix="cravr-groups-"), "groups.json"))
os.environ.setdefault("CRAVR_WEATHER", "0")
