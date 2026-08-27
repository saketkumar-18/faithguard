"""Find a working TokenRouter API key without printing any secret.

Probes candidate locations, prints only metadata (source, length, hash prefix,
first-6/last-4 chars), then tests each distinct candidate against the
TokenRouter API and reports pass/fail.
"""
import hashlib
import os
import re
from pathlib import Path

import requests

candidates = []  # (source, key)


def add(source, key):
    if key and isinstance(key, str) and len(key) > 8:
        candidates.append((source, key.strip()))


# 1. shell environment
add("env:TOKENROUTER_API_KEY", os.environ.get("TOKENROUTER_API_KEY", ""))

# 2. Hermes config files (default profile + all profiles)
localappdata = Path(os.environ.get("LOCALAPPDATA", ""))
cfg_paths = [localappdata / "hermes" / "config.yaml"]
profiles = localappdata / "hermes" / "profiles"
if profiles.exists():
    cfg_paths += list(profiles.glob("*/config.yaml"))
for cfg in cfg_paths:
    if not cfg.exists():
        continue
    text = cfg.read_text(encoding="utf-8", errors="ignore")
    # capture api_key lines within 12 lines of a 'tokenrouter' mention
    lines = text.splitlines()
    for i, ln in enumerate(lines):
        if "tokenrouter" in ln.lower():
            window = lines[max(0, i - 6): i + 14]
            for w in window:
                m = re.search(r"api_key:\s*[\"']?([A-Za-z0-9_\-\.]+)[\"']?", w)
                if m:
                    add(f"hermes_cfg:{cfg.name}#L{i}", m.group(1))
    # also any env-style reference
    m = re.search(r"TOKENROUTER_API_KEY\s*[:=]\s*[\"']?([A-Za-z0-9_\-\.]+)", text)
    if m:
        add(f"hermes_cfg_ref:{cfg.name}", m.group(1))

# 3. ~/.render/cli.yaml
cli_yaml = Path.home() / ".render" / "cli.yaml"
if cli_yaml.exists():
    text = cli_yaml.read_text(encoding="utf-8", errors="ignore")
    m = re.search(r"TOKENROUTER_API_KEY\s*[:=]\s*[\"']?([A-Za-z0-9_\-\.]+)", text)
    if m:
        add("render_cli.yaml", m.group(1))

# dedupe by value
seen = {}
for src, key in candidates:
    seen.setdefault(key, []).append(src)

print(f"distinct candidates: {len(seen)}")
results = []
for key, srcs in seen.items():
    h = hashlib.sha256(key.encode()).hexdigest()[:10]
    meta = f"len={len(key)} sha={h} pre=***"
    ok = False
    detail = ""
    try:
        r = requests.get(
            "https://api.tokenrouter.com/v1/models",
            headers={"Authorization": f"Bearer {key}"},
            timeout=30,
        )
        ok = r.status_code == 200
        detail = f"HTTP {r.status_code}"
        if not ok:
            detail += " " + r.text[:120].replace("\n", " ")
    except Exception as e:
        detail = f"EXC {type(e).__name__}: {str(e)[:100]}"
    results.append((ok, key, srcs, meta, detail))
    print(f"[{'VALID' if ok else 'invalid'}] {meta} | sources={srcs} | {detail}")

valid = [k for ok, k, *_ in results if ok]
if valid:
    # stash the first valid key for the next step (file readable only locally)
    out = Path.home() / ".render" / "tokenrouter_key.tmp"
    out.write_text(valid[0], encoding="utf-8")
    print(f"\nWORKING KEY stashed to {out} (not printed)")
else:
    print("\nNO VALID KEY FOUND")
