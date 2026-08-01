#!/usr/bin/env python3
"""Isolate which patch layer costs prefill throughput at long context.

Server prefill measured 938 tok/s at 28.8K before speculative decoding was
wired in, and 436 tok/s at 25.3K after. The speculative hook also applies
oMLX's MTP model patch (for the rollback helpers) and wraps
`DeepseekV4Model.__call__` to capture hidden states, either of which could be
responsible.

Times prefill only, at the same length, under each combination.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import mlx.core as mx


def time_prefill(model, ids, reps: int = 2) -> float:
    """`ids` must be REAL text. A repeated or constant token makes every
    position route to the same MoE experts, which massively overstates prefill
    throughput -- that is exactly how a 938 tok/s figure got published."""
    from mlx_lm.models.cache import make_prompt_cache

    best = float("inf")
    for _ in range(reps):
        cache = make_prompt_cache(model)
        t0 = time.perf_counter()
        mx.eval(model(ids, cache=cache))
        best = min(best, time.perf_counter() - t0)
        del cache
        mx.clear_cache()
    return best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("--tokens", type=int, default=25000)
    ap.add_argument("--text-file", default=None)
    ap.add_argument(
        "--mode",
        required=True,
        choices=["stock", "windowed", "windowed+mtp", "all"],
        help="stock = oMLX only; windowed = + blocked prefill; "
        "windowed+mtp = + oMLX MTP model patch; all = + hidden capture",
    )
    args = ap.parse_args()

    os.environ["DS4_PATCHES"] = "0"  # keep the .pth boot hook out of it
    from omlx.patches.deepseek_v4 import apply_deepseek_v4_patch

    apply_deepseek_v4_patch()

    if args.mode != "stock":
        from ds4 import windowed_prefill

        windowed_prefill.apply()

    if args.mode in ("windowed+mtp", "all"):
        from omlx.patches.mlx_lm_mtp import cache_rollback
        from omlx.patches.mlx_lm_mtp import deepseek_v4_model as mtp_model

        cache_rollback.apply()
        mtp_model.apply()

    if args.mode == "all":
        from ds4 import engine_hook

        engine_hook._apply_hidden_capture()

    from mlx_lm.utils import load_model

    model, _ = load_model(Path(args.model_dir))
    mx.eval(model.parameters())

    from mlx_lm.tokenizer_utils import load as load_tokenizer

    tok = load_tokenizer(Path(args.model_dir))
    if args.text_file:
        toks = tok.encode(Path(args.text_file).read_text())
        while len(toks) < args.tokens:
            toks = toks + toks
        ids = mx.array([toks[: args.tokens]])
        kind = "real text"
    else:
        mx.random.seed(0)
        ids = mx.random.randint(0, 100000, (1, args.tokens))
        kind = "random tokens"

    dt = time_prefill(model, ids)
    print(f"{args.mode:<16} {args.tokens} tok ({kind}) in {dt:6.2f}s "
          f"= {args.tokens / dt:6.0f} tok/s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
