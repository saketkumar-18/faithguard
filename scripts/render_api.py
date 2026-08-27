"""Render API helper — reads API key from ~/.render/cli.yaml at runtime."""
import json
import re
import sys
import urllib.request
from pathlib import Path

def get_key():
    cfg = Path.home() / ".render" / "cli.yaml"
    text = cfg.read_text(encoding="utf-8")
    m = re.search(r"key:\s*(\S+)", text)
    return m.group(1)

def api(path, method="GET", body=None):
    key = get_key()
    url = f"https://api.render.com/v1{path}"
    req = urllib.request.Request(url, method=method)
    req.add_header("Authorization", f"Bearer {key}")
    req.add_header("Accept", "application/json")
    if body is not None:
        req.add_header("Content-Type", "application/json")
        req.data = json.dumps(body).encode()
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        return {"_http_error": e.code, "_body": e.read().decode()[:2000]}

if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "owners":
        for o in api("/owners"):
            ow = o.get("owner", {})
            print(ow.get("email"), "|", ow.get("name"), "|", ow.get("type"), "|", ow.get("id"))
    elif cmd == "services":
        for s in api("/services?limit=50"):
            sv = s.get("service", {})
            print(sv.get("id"), "|", sv.get("name"), "|", sv.get("type"), "|", sv.get("serviceDetails", {}).get("url", ""))
    elif cmd == "workspace":
        cfg = (Path.home() / ".render" / "cli.yaml").read_text()
        m = re.search(r"workspace:\s*(\S+)", cfg)
        print(m.group(1) if m else "none")
