#!/usr/bin/env python3
"""Correctness + speed of the windowed prefill patch.

Correctness first: the blocked path must reproduce the dense path's logits to
within the model's own numerical noise (see FINDINGS §9 -- a 1 bf16 ULP
difference amplified by top-k routing is expected; a wrong mask is not).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mlx.core as mx


def prefill(model, ids):
    from mlx_lm.models.cache import make_prompt_cache

    cache = make_prompt_cache(model)
    out = model(ids, cache=cache)
    mx.eval(out)
    return out, cache


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("--lengths", default="2048,4096,8192")
    ap.add_argument("--block", type=int, default=256)
    ap.add_argument("--text-file", default=None)
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from omlx.patches.deepseek_v4 import apply_deepseek_v4_patch

    apply_deepseek_v4_patch()

    from mlx_lm.utils import load_model

    from ds4 import windowed_prefill

    model, _ = load_model(Path(args.model_dir))
    mx.eval(model.parameters())
    print(f"loaded; resident {mx.get_active_memory() / 1024**3:.1f} GiB\n")

    lens = [int(x) for x in args.lengths.split(",")]

    def make_ids(L):
        """Real text where available.

        Random tokens activate essentially every MoE expert (worst case) and a
        repeated token activates almost none (best case), so prefill numbers
        taken on either are not representative of real traffic.
        """
        if args.text_file:
            from mlx_lm.tokenizer_utils import load as load_tok

            toks = load_tok(Path(args.model_dir)).encode(
                Path(args.text_file).read_text()
            )
            while len(toks) < L:
                toks = toks + toks
            return mx.array([toks[:L]])
        mx.random.seed(0)
        return mx.random.randint(0, 100000, (1, L))


    dense = {}

    print(f"{'L':>7} {'dense tok/s':>13} {'windowed tok/s':>16} {'speedup':>9} {'max|dlogit|':>13}")
    print("-" * 62)

    for L in lens:
        ids = make_ids(L)

        prefill(model, ids)
        mx.clear_cache()
        t0 = time.perf_counter()
        out, cache = prefill(model, ids)
        dt_dense = time.perf_counter() - t0
        dense[L] = mx.array(out[:, -1])
        mx.eval(dense[L])
        del cache, out
        mx.clear_cache()
        dense[L] = (dense[L], dt_dense)

    windowed_prefill.apply(block=args.block)

    for L in lens:
        ids = make_ids(L)

        prefill(model, ids)
        mx.clear_cache()
        t0 = time.perf_counter()
        out, cache = prefill(model, ids)
        dt = time.perf_counter() - t0
        last = mx.array(out[:, -1])
        mx.eval(last)

        ref, dt_dense = dense[L]
        d = float(mx.max(mx.abs(last - ref)))
        print(
            f"{L:>7} {L / dt_dense:>13.0f} {L / dt:>16.0f} "
            f"{dt_dense / dt:>8.2f}x {d:>13.3f}"
        )
        del cache, out
        mx.clear_cache()

    print("\n(a max|dlogit| of ~1-2 is the model's own batching noise, see FINDINGS §9;")
    print(" a broken mask would show orders of magnitude more)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
