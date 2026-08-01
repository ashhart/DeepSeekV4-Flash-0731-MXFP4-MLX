#!/usr/bin/env python3
"""Does oMLX's prompt cache actually engage on a repeated long prompt?

For agent traffic that resends a growing context every turn, a working prompt
cache matters far more than any kernel: a 25K prompt is ~58s of prefill, and a
cache hit removes almost all of it.

`hot_cache_max_size` defaults to "0", which the config comments mark as
*disabled* -- and the oMLX UI does not expose the field, so with "Hot Cache
Only" on (RAM only, no SSD) nothing is cached and every turn re-prefills.

Sends the same long prompt twice and reports cached_tokens and wall time.
"""

from __future__ import annotations

import json
import pathlib
import sys
import time
import urllib.request

MODEL = "DeepSeek-V4-Flash-0731-MXFP4-MLX"
URL = "http://127.0.0.1:8000/v1/chat/completions"


def key() -> str:
    settings = pathlib.Path.home() / ".omlx" / "settings.json"
    return json.loads(settings.read_text())["auth"]["api_key"]


def call(k, prompt, max_tokens=8):
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
        headers={"Authorization": f"Bearer {k}", "Content-Type": "application/json"},
    )
    t0 = time.time()
    resp = json.load(urllib.request.urlopen(req, timeout=1800))
    return time.time() - t0, resp["usage"]


if __name__ == "__main__":
    src = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/bigctx.txt")
    prompt = src.read_text()[:90000] + "\n\nReply with exactly: OK"
    k = key()

    print(f"{'run':>5} {'prompt':>8} {'cached':>8} {'wall':>8}")
    print("-" * 34)
    for i in (1, 2):
        dt, u = call(k, prompt)
        cached = u.get("prompt_tokens_details", {}).get("cached_tokens", 0)
        print(f"{i:>5} {u['prompt_tokens']:>8} {cached:>8} {dt:>7.1f}s")
    print("\nrun 2 should show cached_tokens ~= prompt_tokens and a much")
    print("shorter wall time. If cached is 0, the prompt cache is not engaging.")
