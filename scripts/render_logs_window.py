"""Fetch Render logs around a specific time window."""
import os
import subprocess
import sys

start = sys.argv[1] if len(sys.argv) > 1 else "2026-08-27T20:04:00Z"
out = subprocess.run(
    [
        os.path.expanduser("~/bin/render.exe"),
        "logs", "-r", "srv-da86o1p42hec73c18p60",
        "--start", start,
        "--limit", "200",
    ],
    capture_output=True, text=True, timeout=120,
)
lines = out.stdout.splitlines()
keep = [ln for ln in lines if "GET /health" not in ln]
print("\n".join(keep[-80:]))
print("---TOTAL:", len(lines), "KEPT:", len(keep))
