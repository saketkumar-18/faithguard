"""Granular memory measurement to find what to lazy-load / cut."""
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def rss_mb():
    pid = os.getpid()
    out = subprocess.run(
        ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
        capture_output=True, text=True,
    ).stdout.strip()
    m = re.search(r'"([\d,]+) K"\s*$', out)
    return int(m.group(1).replace(",", "")) // 1024 if m else -1


print("baseline python:", rss_mb(), "MB", flush=True)
import numpy  # noqa
print("+ numpy:", rss_mb(), "MB", flush=True)
import fastapi  # noqa
print("+ fastapi:", rss_mb(), "MB", flush=True)
import sklearn  # noqa
print("+ sklearn:", rss_mb(), "MB", flush=True)
import onnxruntime  # noqa
print("+ onnxruntime:", rss_mb(), "MB", flush=True)
