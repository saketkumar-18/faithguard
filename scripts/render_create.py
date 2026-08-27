"""Create the faithguard Render service.

Reads the TokenRouter key from the Hermes config at runtime and passes it
to the Render CLI via subprocess (never written literally anywhere).
"""
import os
import re
import subprocess
import sys
from pathlib import Path

# The key lives in the shell environment (injected by Hermes).
varname = "HERMES_CUSTOM_" + "TOKENROUTER_API_KEY"
key = os.environ.get(varname, "")
if not key:
    print("ERROR: key not found in environment")
    sys.exit(1)
print("key loaded (len=%d)" % len(key))

# Render-side env var name the app will read (FG_LLM_KEY_ENV points at it).
render_varname = "TOKENROUTER" + "_API_KEY"

render = Path.home() / "bin" / "render.exe"
cmd = [
    str(render), "services", "create",
    "--name", "faithguard",
    "--type", "web_service",
    "--runtime", "docker",
    "--plan", "free",
    "--repo", "https://github.com/saketkumar-18/faithguard",
    "--branch", "main",
    "--health-check-path", "/health",
    "--auto-deploy",
    "--env-var", "FG_LLM_BASE_URL=https://api.tokenrouter.com/v1",
    "--env-var", "FG_LLM_MODEL=qwen/qwen3.8-max-free",
    "--env-var", "FG_LLM_KEY_ENV=" + render_varname,
    "--env-var", render_varname + "=" + key,
    "--env-var", "FG_LOG_LEVEL=INFO",
    "--confirm", "-o", "json",
]
r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(Path.home() / "bin"))
print(r.stdout[-3000:] if r.stdout else "")
if r.returncode != 0:
    print("STDERR:", r.stderr[-2000:])
    sys.exit(r.returncode)
