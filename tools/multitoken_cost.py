#!/usr/bin/env python3
"""Cost of a multi-token forward, and real prefill throughput.

Two questions:

1. Speculative decoding only wins if verifying `k` tokens costs less than `k`
   single-token steps. For a MoE that is not obvious: 6 tokens x 6 experts draws
   up to ~34 distinct experts of 256 per layer instead of 6, so verify reads far
   more weight than one decode step.

2. Prefill throughput on a genuinely long prompt -- short-prompt numbers are
   dominated by fixed setup cost and mean nothing.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mlx.core as mx


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("--reps", type=int, default=10)
    args = ap.parse_args()

    from omlx.patches.deepseek_v4 import apply_deepseek_v4_patch

    apply_deepseek_v4_patch()

    from mlx_lm.models.cache import make_prompt_cache
    from mlx_lm.utils import load_model

    model, _ = load_model(Path(args.model_dir))
    mx.eval(model.parameters())
    print(f"loaded; resident {mx.get_active_memory() / 1024**3:.1f} GiB\n")

    print("=== multi-token forward (one sequence, ctx 512) ===")
    print(f"{'tokens':>7} {'ms':>9} {'vs L=1':>8} {'break-even accept':>19}")
    print("-" * 48)

    base = None
    for L in (1, 2, 4, 6, 8):
        cache = make_prompt_cache(model)
        mx.eval(model(mx.array([[1] * 512]), cache=cache))
        tok = mx.array([[1] * L])

        for _ in range(2):
            mx.eval(model(tok, cache=cache))

        t0 = time.perf_counter()
        for _ in range(args.reps):
            mx.eval(model(tok, cache=cache))
        dt = (time.perf_counter() - t0) / args.reps * 1000

        if base is None:
            base = dt
        ratio = dt / base
        # verify emits at most L tokens; need tokens/cycle > ratio to win
        print(f"{L:>7} {dt:>9.1f} {ratio:>8.2f}x {ratio:>18.2f}")
        del cache
        mx.clear_cache()

    print("\n=== prefill throughput ===")
    print(f"{'prompt':>8} {'ms':>10} {'tok/s':>10}")
    print("-" * 30)
    for n in (512, 2048, 8192):
        cache = make_prompt_cache(model)
        prompt = mx.array([[1] * n])
        mx.eval(model(prompt[:, :8], cache=cache))
        del cache
        mx.clear_cache()

        cache = make_prompt_cache(model)
        t0 = time.perf_counter()
        mx.eval(model(prompt, cache=cache))
        dt = time.perf_counter() - t0
        print(f"{n:>8} {dt * 1000:>10.0f} {n / dt:>10.0f}")
        del cache
        mx.clear_cache()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
