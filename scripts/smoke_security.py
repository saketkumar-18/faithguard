"""Live smoke test of the production security layer.

Reads the API key from a file (never from argv/env in this script's source)
and exercises: open /health, 401 without key, 401 wrong key, 200 correct key
(x-api-key + Bearer), 429 rate limit, open /metrics.
"""
import os
import sys
import time

import requests

BASE = "http://127.0.0.1:8766"
KEYFILE = os.path.join(os.environ["LOCALAPPDATA"], "Temp", "fgkey.txt")
KEY = open(KEYFILE).read().strip()

results = []


def check(name, expected, got):
    ok = expected == got
    results.append((name, expected, got, ok))
    print(f"{'PASS' if ok else 'FAIL'}  {name}: expected {expected}, got {got}")
    return ok


# wait for server
for _ in range(30):
    try:
        r = requests.get(f"{BASE}/health", timeout=5)
        if r.status_code == 200:
            break
    except requests.RequestException:
        pass
    time.sleep(2)

# 1. /health open
check("/health open (no key)", 200, requests.get(f"{BASE}/health", timeout=10).status_code)

# 2. /detect no key -> 401
check("/detect no key -> 401", 401, requests.post(
    f"{BASE}/detect", json={"answer": "x", "passages": ["y"]}, timeout=10).status_code)

# 3. /detect wrong key -> 401
check("/detect wrong key -> 401", 401, requests.post(
    f"{BASE}/detect", json={"answer": "x", "passages": ["y"]},
    headers={"x-api-key": "wrong-key"}, timeout=10).status_code)

# 4. /detect correct key (x-api-key) -> 200, and a contradicted answer is flagged
r = requests.post(
    f"{BASE}/detect",
    json={"answer": "The Eiffel Tower is in Berlin.",
          "passages": ["The Eiffel Tower is in Paris."]},
    headers={"x-api-key": KEY}, timeout=60)
check("/detect correct key (x-api-key) -> 200", 200, r.status_code)
if r.status_code == 200:
    body = r.json()
    print(f"      hallucinated={body['hallucinated']} p={body['probability']} method={body['method']}")
    check("contradicted answer is flagged (safety floor)", True, body["hallucinated"])

# 5. Bearer token -> 200
r = requests.post(
    f"{BASE}/detect",
    json={"answer": "Paris is the capital of France.",
          "passages": ["Paris is the capital of France."]},
    headers={"Authorization": f"Bearer {KEY}"}, timeout=60)
check("/detect Bearer token -> 200", 200, r.status_code)

# 6. rate limit: limit=5/window; we've used 2 successful + failures don't count
#    against the key... actually failures DO count (they carry the key).
#    Send until we hit 429 (max 6 more).
hit_429 = False
for i in range(6):
    r = requests.post(
        f"{BASE}/detect",
        json={"answer": "Paris is the capital of France.",
              "passages": ["Paris is the capital of France."]},
        headers={"x-api-key": KEY}, timeout=60)
    if r.status_code == 429:
        hit_429 = True
        print(f"      429 after {i+1} additional requests (Retry-After: {r.headers.get('Retry-After')})")
        break
check("rate limit triggers 429", True, hit_429)

# 7. /metrics open
check("/metrics open -> 200", 200, requests.get(f"{BASE}/metrics", timeout=10).status_code)

print()
n_pass = sum(1 for r in results if r[3])
print(f"{n_pass}/{len(results)} checks passed")
sys.exit(0 if n_pass == len(results) else 1)
