#!/usr/bin/env python3
"""Measure prefill through the oMLX HTTP API.

Sends a long prompt with max_tokens=1 so the reported time is dominated by
prefill, and subtracts any model-load time the server reports.

The API key comes from OMLX_API_KEY, falling back to ~/.omlx/settings.json.
Never hardcode it -- it is a credential.
"""

from __future__ import annotations

import json
import os
import time
import urllib.request
from pathlib import Path

MODEL = os.environ.get("OMLX_MODEL", "DeepSeek-V4-Flash-0731-MXFP4-MLX")
URL = os.environ.get("OMLX_URL", "http://127.0.0.1:8000/v1/chat/completions")


def api_key() -> str:
    key = os.environ.get("OMLX_API_KEY")
    if key:
        return key
    settings = Path.home() / ".omlx" / "settings.json"
    if settings.exists():
        key = json.loads(settings.read_text()).get("auth", {}).get("api_key")
        if key:
            return key
    raise SystemExit("set OMLX_API_KEY, or run where ~/.omlx/settings.json exists")


def call(key: str, prompt: str, max_tokens: int = 1):
    body = json.dumps(
        {
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0,
        }
    ).encode()
    req = urllib.request.Request(
        URL,
        data=body,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    t0 = time.time()
    resp = json.load(urllib.request.urlopen(req, timeout=1800))
    return time.time() - t0, resp["usage"]


UNIT = (
    "def transform_record(record, index):\n"
    "    total = record.get('value', 0) * index\n"
    "    return {'id': record['id'], 'total': total}\n\n"
)

if __name__ == "__main__":
    key = api_key()
    print(f"{'prompt tok':>11} {'wall s':>8} {'load s':>8} {'prefill tok/s':>14}")
    print("-" * 45)

    call(key, "hi", 1)  # warm, so model-load time is out of the measured rows

    for reps in (200, 400, 800):
        dt, usage = call(key, UNIT * reps + "\nReply with exactly: OK", 1)
        load = usage.get("model_load_duration") or 0.0
        n = usage["prompt_tokens"]
        print(f"{n:>11} {dt:>8.2f} {load:>8.2f} {n / max(dt - load, 1e-6):>14.0f}")
