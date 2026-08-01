#!/usr/bin/env python3
"""Measure decode rate by SLOPE, so prefill cannot contaminate it.

Timing `max_tokens=1` and subtracting it from `max_tokens=N` is unreliable:
prefill time varies run to run (cache state, scheduling), and since it is the
dominant term at long context, small variation in it swamps the decode estimate.
The same prompt measured that way gave 34.4 and 77.4 tok/s on consecutive runs
while total turn time was ~6.9s both times.

Instead generate N1 and N2 tokens from the identical prompt and take

    decode = (N2 - N1) / (t2 - t1)

Any fixed per-request cost -- prefill, scheduling, HTTP -- cancels. Repeats each
point and uses the median to blunt scheduler jitter.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import time
import urllib.request

MODEL = "DeepSeek-V4-Flash-0731-MXFP4-MLX"
URL = "http://127.0.0.1:8000/v1/chat/completions"
KEY = json.loads(
    (pathlib.Path.home() / ".omlx" / "settings.json").read_text()
)["auth"]["api_key"]


def call(prompt: str, max_tokens: int) -> tuple[float, dict]:
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
    return time.time() - t0, resp["usage"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--context-file", default=None)
    ap.add_argument("--context-chars", type=int, default=85000)
    ap.add_argument("--n1", type=int, default=60)
    ap.add_argument("--n2", type=int, default=260)
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()

    prompt = ""
    if args.context_file:
        prompt = pathlib.Path(args.context_file).read_text()[: args.context_chars]
        prompt += "\n\n"
    prompt += "Explain, in detail, how a KV cache works in transformer inference."

    call(prompt, 8)  # prime the prompt cache so both points share a warm prefix

    t1s, t2s = [], []
    for _ in range(args.reps):
        t1, u1 = call(prompt, args.n1)
        t2, u2 = call(prompt, args.n2)
        t1s.append(t1)
        t2s.append(t2)

    m1, m2 = statistics.median(t1s), statistics.median(t2s)
    got1, got2 = u1["completion_tokens"], u2["completion_tokens"]
    rate = (got2 - got1) / max(m2 - m1, 1e-9)

    cached = u2["prompt_tokens_details"]["cached_tokens"]
    print(f"prompt {u2['prompt_tokens']} tok, cached {cached}")
    print(f"  {got1:>4} tok: {m1:6.2f}s   (runs {[round(t, 2) for t in t1s]})")
    print(f"  {got2:>4} tok: {m2:6.2f}s   (runs {[round(t, 2) for t in t2s]})")
    print(f"\ndecode (slope) = {rate:.1f} tok/s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
