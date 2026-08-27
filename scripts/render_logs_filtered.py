"""Fetch Render runtime logs since the failed deploy and filter out health spam."""
import os
import subprocess
import sys

env = dict(os.environ)
out = subprocess.run(
    [
        os.path.expanduser("~/bin/render.exe"),
        "logs", "-r", "srv-da86o1p42hec73c18p60",
        "--start", "2026-08-27T19:13:00Z",
        "--limit", "400",
    ],
    capture_output=True, text=True, env=env, timeout=120,
)
lines = out.stdout.splitlines()
keep = []
for ln in lines:
    if "GET /health" in ln or ('" 200' in ln and "GET /" in ln):
        continue
    keep.append(ln)
print("\n".join(keep[-60:]))
print("---TOTAL LINES:", len(lines), "KEPT:", len(keep))
