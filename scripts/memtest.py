"""Measure RSS at each stage of model loading (Windows tasklist)."""
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def rss_mb():
    import re
    pid = os.getpid()
    out = subprocess.run(
        ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
        capture_output=True, text=True,
    ).stdout.strip()
    # last quoted field looks like "12,416 K"
    m = re.search(r'"([\d,]+) K"\s*$', out)
    if m:
        return int(m.group(1).replace(",", "")) // 1024
    return -1


print("baseline python:", rss_mb(), "MB", flush=True)
import numpy, sklearn, fastapi  # noqa: E401,F401
print("+ numpy/sklearn/fastapi:", rss_mb(), "MB", flush=True)
from fastembed import TextEmbedding
m = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
list(m.embed(["test query"]))
print("+ fastembed bge-small loaded+warmed:", rss_mb(), "MB", flush=True)
from faithguard.detection.nli import NLIScorer
nli = NLIScorer("cross-encoder/nli-deberta-v3-small")
nli.score_claims([{"text": "a claim", "hedged": False}], ["a passage"])
print("+ NLI ONNX loaded+warmed:", rss_mb(), "MB", flush=True)
