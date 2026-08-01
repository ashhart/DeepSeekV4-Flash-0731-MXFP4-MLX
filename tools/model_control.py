#!/usr/bin/env python3
"""Guarded authenticated oMLX model load/unload control for the Studio."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
import urllib.parse
import urllib.request

import studio_guard


MODEL = "DeepSeek-V4-Flash-0731-MXFP4-MLX"
BASE = "http://127.0.0.1:8000/admin/api/models"


def _cookie() -> str:
    from omlx.admin.auth import SESSION_COOKIE_NAME, create_session_token, init_auth

    settings = json.loads((Path.home() / ".omlx" / "settings.json").read_text())
    init_auth(settings["auth"]["secret_key"])
    return f"{SESSION_COOKIE_NAME}={create_session_token()}"


def _post(action: str) -> dict:
    model = urllib.parse.quote(MODEL, safe="")
    request = urllib.request.Request(
        f"{BASE}/{model}/{action}",
        data=b"",
        headers={"Cookie": _cookie()},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.load(response)


def load() -> None:
    studio_guard.assert_safe(required_free_gib=180.0)
    started = time.monotonic()
    result = _post("load")
    studio_guard.assert_safe(required_free_gib=24.0, expected_loaded={MODEL})
    print(
        json.dumps(
            {
                "action": "load",
                "status": result.get("status"),
                "model": MODEL,
                "elapsed_s": round(time.monotonic() - started, 2),
            }
        )
    )


def unload() -> None:
    studio_guard.assert_safe(required_free_gib=24.0, expected_loaded={MODEL})
    started = time.monotonic()
    result = _post("unload")
    deadline = time.monotonic() + 120
    while True:
        report, failures = studio_guard.preflight(required_free_gib=180.0)
        if not failures:
            break
        if time.monotonic() >= deadline:
            print(json.dumps({"last_reclaim_report": report}, indent=2))
            raise RuntimeError("model unload did not return to safe headroom")
        time.sleep(1)
    print(json.dumps({"post_unload": report}, indent=2))
    print(
        json.dumps(
            {
                "action": "unload",
                "status": result.get("status"),
                "model": MODEL,
                "elapsed_s": round(time.monotonic() - started, 2),
            }
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("load", "unload", "status"))
    args = parser.parse_args()
    if args.action == "load":
        load()
    elif args.action == "unload":
        unload()
    else:
        report, failures = studio_guard.preflight(required_free_gib=32.0)
        print(json.dumps(report, indent=2))
        if failures:
            for failure in failures:
                print(failure, file=sys.stderr)
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
