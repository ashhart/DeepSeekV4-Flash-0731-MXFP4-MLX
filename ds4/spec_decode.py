#!/usr/bin/env python3
"""DSpark speculative decoding: real generation loop + throughput measurement.

Protocol (greedy, exactness-preserving -- output is token-identical to plain
greedy decoding):

  1. drafter proposes d[1..k] from the main model's fused hidden state
  2. main model verifies [t, d1..dk] in ONE forward -> logits L[0..k]
     L[i] is the main model's prediction for the token after input i
  3. accept the longest prefix where argmax(L[i]) == d[i+1]; the first
     disagreement is corrected by argmax(L[n]), so a cycle always commits at
     least one token and never commits a token the main model would not have
  4. trim the cache back to the accepted length

Because the drafter's only persistent state is a rotating window of main-model
KV, rejection costs nothing to undo on the drafter side.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import mlx.core as mx


def cache_offset(cache) -> int:
    from mlx_lm.models.cache import CacheList

    c = cache[0]
    if isinstance(c, CacheList):
        c = c[0]
    return c.offset


def _trim_rotating(c, n: int) -> None:
    """Drop the last `n` entries of a RotatingKVCache after a multi-token update.

    `is_trimmable()` returns False once `offset >= max_size`, because in the
    *in-place* (S==1) path the ring has wrapped and the buffer is no longer in
    temporal order. But a verify pass has S>1, which always takes
    `_update_concat` -- and that calls `_temporal_order` first, then rebinds
    `keys`/`values` to freshly concatenated arrays with the new entries last.
    So immediately after a verify the buffer *is* temporally ordered and the
    rejected tail can simply be sliced off. This is why oMLX's undo log only
    needed to cover S==2: the same property holds for any S>1.
    """
    if n <= 0 or c.keys is None:
        return
    c.keys = c.keys[..., :-n, :]
    c.values = c.values[..., :-n, :]
    c.offset -= n
    c._idx = c.keys.shape[2]


def trim_cache(cache, n: int) -> int:
    """Roll `n` rejected positions back out of every layer's cache."""
    if n <= 0:
        return 0
    for layer_cache in cache:
        entries = (
            list(layer_cache)
            if isinstance(layer_cache, (list, tuple))
            else [layer_cache]
        )
        for c in entries:
            if c is None:
                continue
            if type(c).__name__.endswith("RotatingKVCache"):
                _trim_rotating(c, n)
            elif hasattr(c, "trim"):
                c.trim(n)
            else:
                return -1
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("--max-tokens", type=int, default=192)
    ap.add_argument("--block-size", type=int, default=0, help="0 = checkpoint default")
    ap.add_argument("--prompt", default="def mergesort(arr):")
    ap.add_argument("--baseline", action="store_true", help="also time plain decode")
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    from omlx.patches.deepseek_v4 import apply_deepseek_v4_patch

    apply_deepseek_v4_patch()

    from mlx_lm.models.cache import make_prompt_cache
    from mlx_lm.tokenizer_utils import load as load_tokenizer
    from mlx_lm.utils import load_model

    from ds4.acceptance import patch_main_hidden
    from ds4.load_dspark import build_drafter

    model_path = Path(args.model_dir)
    config = json.loads((model_path / "config.json").read_text())
    patch_main_hidden(config["dspark_target_layer_ids"])

    model, _ = load_model(model_path)
    mx.eval(model.parameters())
    tokenizer = load_tokenizer(model_path)

    drafter, weights, dspark = build_drafter(model_path, config, model.args)
    drafter.load_weights(list(weights.items()), strict=True)
    mx.eval(drafter.parameters())
    del weights

    k = args.block_size or dspark["dspark_block_size"]
    drafter.block_size = k
    print(f"loaded; resident {mx.get_active_memory() / 1024**3:.1f} GiB, block_size {k}")

    prompt_ids = tokenizer.encode(args.prompt)

    # ---------- plain greedy baseline (for speedup + exactness) ----------
    ref_tokens = []
    if args.baseline:
        cache = make_prompt_cache(model)
        ids = mx.array([prompt_ids])
        logits = model(ids, cache=cache)
        tok = mx.argmax(logits[:, -1:], axis=-1)
        mx.eval(tok)
        t0 = time.perf_counter()
        for _ in range(args.max_tokens):
            ref_tokens.append(int(tok[0, 0]))
            logits = model(tok, cache=cache)
            tok = mx.argmax(logits[:, -1:], axis=-1)
        mx.eval(tok)
        base_dt = time.perf_counter() - t0
        print(f"\nbaseline greedy : {args.max_tokens / base_dt:>6.1f} tok/s")
        del cache
        mx.clear_cache()

    # ---------- speculative ----------
    cache = make_prompt_cache(model)
    ids = mx.array([prompt_ids])
    logits = model(ids, cache=cache)
    tok = mx.argmax(logits[:, -1:], axis=-1)
    window = drafter.push_window(drafter.new_window(1), model.model.main_hidden, 0)
    pos = len(prompt_ids) - 1
    mx.eval(tok, window)

    out = []
    cycles = 0
    accepted_total = 0
    t0 = time.perf_counter()

    while len(out) < args.max_tokens:
        draft_ids, _conf = drafter(
            tok, model.model.embed_tokens, model.lm_head, window, pos
        )
        mx.eval(draft_ids)

        # Verify [t, d1..dk] in one pass.
        cand = mx.concatenate([tok, draft_ids], axis=1)
        vlogits = model(cand, cache=cache)
        main_pred = mx.argmax(vlogits, axis=-1)[0]  # (k+1,)
        mx.eval(main_pred)

        pred = [int(x) for x in main_pred.tolist()]
        drafts = [int(x) for x in draft_ids[0].tolist()]

        n = 0
        while n < k and pred[n] == drafts[n]:
            n += 1

        committed = drafts[:n] + [pred[n]]
        out.extend(committed)
        cycles += 1
        accepted_total += n

        # Cache holds k+1 inputs; only the first n+1 were real.
        if trim_cache(cache, k - n) < 0:
            print("cache does not support trim -- aborting")
            return 1

        # Window needs one slot per genuinely consumed position. The verify pass
        # produced hidden states for inputs [t, d1..dn]; push exactly those.
        mh = model.model.main_hidden[:, : n + 1]
        for j in range(n + 1):
            pos += 1
            window = drafter.push_window(window, mh[:, j : j + 1], pos)

        tok = mx.array([[committed[-1]]])
        mx.eval(window, tok)

    dt = time.perf_counter() - t0
    n_out = len(out)
    print(f"\nspeculative     : {n_out / dt:>6.1f} tok/s")
    print(f"cycles {cycles}, mean accepted {accepted_total / cycles:.2f}/{k}, "
          f"{n_out / cycles:.2f} tokens/cycle")

    if ref_tokens:
        same = out[: len(ref_tokens)] == ref_tokens[: len(out)]
        print(f"token-identical to greedy: {'yes' if same else 'NO'}")
        print(f"speedup: {(n_out / dt) / (args.max_tokens / base_dt):.2f}x")

    print("\n" + tokenizer.decode(out)[:600])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
