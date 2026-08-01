#!/usr/bin/env python3
"""End-to-end timing of a realistic agent turn: long cached prompt + generation.

Splits prefill from decode by timing max_tokens=1 against max_tokens=N on the
same (already cached) prompt, so the two costs can be read separately. That
matters because for a 25K-token turn the two are wildly different in scale and a
single tok/s figure hides which one you are actually paying.
"""

from __future__ import annotations

import json
import pathlib
import sys
import time
import urllib.request

MODEL = "DeepSeek-V4-Flash-0731-MXFP4-MLX"
URL = "http://127.0.0.1:8000/v1/chat/completions"
KEY = json.loads(
    (pathlib.Path.home() / ".omlx" / "settings.json").read_text()
)["auth"]["api_key"]


def call(prompt: str, max_tokens: int):
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
        headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"},
    )
    t0 = time.time()
    resp = json.load(urllib.request.urlopen(req, timeout=1800))
    return time.time() - t0, resp["usage"], resp["choices"][0]["message"]


if __name__ == "__main__":
    src = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/bigctx.txt")
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    prompt = src.read_text()[:85000] + (
        "\n\nExplain in about 150 words what the rollback bug was."
    )

    call(prompt, 4)  # make sure the prefix is cached
    t1, u1, _ = call(prompt, 1)
    t2, u2, msg = call(prompt, n)

    cached = u2["prompt_tokens_details"]["cached_tokens"]
    decode = (u2["completion_tokens"] - 1) / max(t2 - t1, 1e-6)
    print(f"prompt tokens : {u2['prompt_tokens']}")
    print(f"cached        : {cached}  ({cached / u2['prompt_tokens'] * 100:.0f}%)")
    print(f"prefill       : {t1:.2f}s")
    print(f"{n} tokens     : {t2:.2f}s")
    print(f"decode        : {decode:.1f} tok/s")
    print("---")
    print((msg.get("content") or "")[:300])
