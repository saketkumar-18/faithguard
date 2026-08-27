"""Re-add the TOKENROUTER_API_KEY secret env var to the Render service.

Reads the key from the TOKENROUTER_API_KEY shell env var at runtime.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.render_api import api

key = os.environ.get("TOKENROUTER_API_KEY", "")
if not key:
    print("ERROR: TOKENROUTER_API_KEY not set in environment")
    sys.exit(1)

r = api(
    "/services/srv-da86o1p42hec73c18p60/env-vars",
    method="POST",
    body={"key": "TOKENROUTER_API_KEY", "value": key},
)
print("RESPONSE:", str(r)[:300])
