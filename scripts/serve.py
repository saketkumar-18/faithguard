#!/usr/bin/env python
"""Run the FaithGuard API server.

Local dev:   python scripts/serve.py --reload
Production:  python scripts/serve.py --host 0.0.0.0 --port 8000

Production hardening is driven by env vars (see README):
  FG_API_KEY        enable API-key auth (strongly recommended when public)
  FG_RATE_LIMIT     requests per window (e.g. 60)
  FG_RATE_WINDOW_S  window seconds (default 60)
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--reload", action="store_true", help="dev auto-reload")
    ap.add_argument("--workers", type=int, default=1,
                    help="uvicorn workers (keep 1: models are in-process)")
    args = ap.parse_args()

    level = os.environ.get("FG_LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    log = logging.getLogger("faithguard.serve")

    if not os.environ.get("FG_API_KEY"):
        log.warning("FG_API_KEY not set — the API will run WITHOUT authentication. "
                    "Fine for local dev; set it before exposing publicly.")

    import uvicorn

    uvicorn.run(
        "faithguard.api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers if not args.reload else 1,
        log_level=level.lower(),
        timeout_graceful_shutdown=30,
        access_log=True,
    )


if __name__ == "__main__":
    main()
