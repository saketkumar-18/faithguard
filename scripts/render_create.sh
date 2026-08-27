#!/usr/bin/env bash
# Create the faithguard service on Render, injecting the TokenRouter key
# read from the Hermes config at runtime (never written literally anywhere).
set -e
cd ~/bin

CFG="$LOCALAPPDATA/hermes/config.yaml"
KEY=*** -E "TOKENROUTER_API_KEY:\s*['\"]?([A-Za-z0-9_.-]+)" "$CFG" | head -1 | sed -E "s/.*:\s*['\"]?//")

if [ -z "$KEY" ]; then
  echo "ERROR: could not read key from $CFG"
  exit 1
fi
echo "key loaded (len=${#KEY})"

./render.exe services create \
  --name faithguard \
  --type web_service \
  --runtime docker \
  --plan free \
  --repo https://github.com/saketkumar-18/faithguard \
  --branch main \
  --health-check-path /health \
  --auto-deploy \
  --env-var "FG_LLM_BASE_URL=https://api.tokenrouter.com/v1" \
  --env-var "FG_LLM_MODEL=qwen/qwen3.8-max-free" \
  --env-var "FG_LLM_KEY_ENV=TOKENR…_KEY" \
  --env-var "TOKENR…KEY" \
  --env-var "FG_LOG_LEVEL=INFO" \
  --confirm -o json
