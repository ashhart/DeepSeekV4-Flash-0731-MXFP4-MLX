#!/usr/bin/env python3
"""Is a multi-token forward numerically equal to the same tokens fed one at a time?

Speculative decoding replaces k single-token steps with one k-token verify. If
those two paths do not produce bit-comparable logits, greedy output will diverge
eventually regardless of how correct the cache rollback is -- one flipped argmax
on a near-tie cascades. This separates "numerics" from "rollback bug".

Teacher-forced: both paths are fed exactly the same tokens, so any difference is
purely batching, not drift.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mlx.core as mx


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("--ctx", type=int, default=128)
    ap.add_argument("--k", type=int, default=6)
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from omlx.patches.deepseek_v4 import apply_deepseek_v4_patch

    apply_deepseek_v4_patch()

    from mlx_lm.models.cache import make_prompt_cache
    from mlx_lm.utils import load_model

    model, _ = load_model(Path(args.model_dir))
    mx.eval(model.parameters())

    prompt = mx.array([[7] * args.ctx])
    follow = [11, 23, 42, 55, 61, 73][: args.k]

    # Path A: one token at a time
    cache = make_prompt_cache(model)
    mx.eval(model(prompt, cache=cache))
    single = []
    for t in follow:
        lg = model(mx.array([[t]]), cache=cache)
        single.append(lg[:, -1])
    a = mx.stack(single, axis=1)[0]
    mx.eval(a)
    del cache
    mx.clear_cache()

    # Path B: all k in one forward
    cache = make_prompt_cache(model)
    mx.eval(model(prompt, cache=cache))
    b = model(mx.array([follow]), cache=cache)[0]
    mx.eval(b)

    diff = mx.abs(a - b)
    argmax_a = mx.argmax(a, axis=-1)
    argmax_b = mx.argmax(b, axis=-1)
    agree = int(mx.sum(argmax_a == argmax_b))

    print(f"ctx {args.ctx}, k {args.k}")
    print(f"max |logit diff| : {float(mx.max(diff)):.6f}")
    print(f"mean |logit diff|: {float(mx.mean(diff)):.6f}")
    print(f"argmax agreement : {agree}/{len(follow)}")

    # How close are the top-2 logits? Near-ties are what flip.
    srt = mx.sort(a, axis=-1)
    margin = srt[:, -1] - srt[:, -2]
    print(f"top-2 margin     : {[round(float(x), 3) for x in margin.tolist()]}")

    if float(mx.max(diff)) > 1e-3:
        print("\n=> multi-token forward is NOT numerically equal to single-token.")
        print("   Greedy divergence is expected and is not a rollback bug.")
    else:
        print("\n=> paths agree; divergence would indicate a real rollback bug.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
