#!/usr/bin/env python3
"""Locate where a k-token forward starts disagreeing with k single-token steps.

Teacher-forced: both paths see identical tokens, so any divergence is batching,
not drift. Captures the hidden state after every layer at the *last* position and
compares.

The layer layout tells us what to blame:
  compress_ratios[i] == 0  -> local attention only (RotatingKVCache)
  compress_ratios[i] != 0  -> pooled/compressed path (PoolingCache)

If divergence is flat noise from layer 0, it is bf16 reassociation. If it jumps
at the first pooled layer, the pooling boundary depends on how many tokens are
fed per call -- which would be a genuine batching bug affecting prefill vs decode
consistency, prompt caching, and chunked prefill, not just speculation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mlx.core as mx


def capture(model, prompt, follow, chunked: bool):
    from mlx_lm.models.cache import make_prompt_cache

    cache = make_prompt_cache(model)
    mx.eval(model(prompt, cache=cache))

    caught = {}
    inner = model.model
    # Dunders resolve on the type, so patch the class and key by instance id.
    index_of = {id(layer): i for i, layer in enumerate(inner.layers)}
    cls = type(inner.layers[0])
    orig = cls.__call__

    def wrapper(self, h, mask, c, ids):
        out = orig(self, h, mask, c, ids)
        caught[index_of[id(self)]] = out[:, -1]  # last position only
        return out

    cls.__call__ = wrapper
    try:
        if chunked:
            model(mx.array([follow]), cache=cache)
        else:
            for t in follow:
                model(mx.array([[t]]), cache=cache)
        snapshot = {i: mx.array(v) for i, v in caught.items()}
        mx.eval(list(snapshot.values()))
    finally:
        cls.__call__ = orig

    del cache
    mx.clear_cache()
    return snapshot


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("--ctx", type=int, default=128)
    ap.add_argument("--k", type=int, default=6)
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from omlx.patches.deepseek_v4 import apply_deepseek_v4_patch

    apply_deepseek_v4_patch()

    from mlx_lm.utils import load_model

    model, config = load_model(Path(args.model_dir))
    mx.eval(model.parameters())

    ratios = list(getattr(model.args, "compress_ratios", []))
    prompt = mx.array([[7] * args.ctx])
    follow = [11, 23, 42, 55, 61, 73][: args.k]

    single = capture(model, prompt, follow, chunked=False)
    multi = capture(model, prompt, follow, chunked=True)

    print(f"ctx {args.ctx}, k {args.k}\n")
    print(f"{'layer':>6} {'ratio':>6} {'max|d|':>10} {'rel':>10}")
    print("-" * 36)
    for i in sorted(single):
        a, b = single[i], multi[i]
        d = float(mx.max(mx.abs(a - b)))
        scale = float(mx.max(mx.abs(a))) or 1.0
        r = d / scale
        ratio = ratios[i] if i < len(ratios) else "?"
        flag = ""
        if i > 0:
            prev_a, prev_b = single[i - 1], multi[i - 1]
            pd = float(mx.max(mx.abs(prev_a - prev_b)))
            if pd < 1e-4 and d > 1e-3:
                flag = "   <-- DIVERGENCE STARTS HERE"
        print(f"{i:>6} {str(ratio):>6} {d:>10.5f} {r:>10.5f}{flag}")
        if i >= 8 and d > 1e-3:
            print("   (diverged; remaining layers omitted)")
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
