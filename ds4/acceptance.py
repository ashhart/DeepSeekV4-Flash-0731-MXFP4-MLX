#!/usr/bin/env python3
"""Increment 3: is the DSpark port correct, and how often are its drafts right?

Measures acceptance *without* any cache rollback. Each step the drafter proposes
`block_size` tokens; the main model then decodes the same number autoregressively
and we compare. Generation always continues from the main model's tokens, so a
bad drafter cannot corrupt the sequence -- this isolates drafter correctness.

Acceptance is what decides the speedup:

    tokens per cycle = 1 + (expected accepted prefix length)
    speedup          = tokens_per_cycle / cycle_cost

A correct port should show position-1 acceptance well above chance. Near-zero
acceptance means the port is wrong, not that the model is bad.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mlx.core as mx


def patch_main_hidden(target_layer_ids):
    """Make DeepseekV4Model also return the hidden states DSpark fuses.

    The reference takes `h.mean(dim=2)` at each target layer -- the mean over the
    hc_mult Hyper-Connection copies -- then concatenates along the feature axis.
    """
    import mlx_lm.models.deepseek_v4 as dsv4
    from mlx_lm.models.base import create_attention_mask
    from mlx_lm.models.cache import CacheList

    cls = dsv4.DeepseekV4Model
    if getattr(cls, "_dspark_hidden_patched", False):
        return
    targets = set(target_layer_ids)

    def __call__(self, inputs, cache=None):
        h = self.embed_tokens(inputs)
        h = mx.contiguous(
            mx.broadcast_to(
                h[:, :, None, :],
                (h.shape[0], h.shape[1], self.args.hc_mult, h.shape[2]),
            )
        )
        if cache is None:
            cache = [None] * len(self.layers)

        first = cache[0]
        mask_cache = first[0] if isinstance(first, CacheList) else first
        mask = create_attention_mask(
            h[:, :, 0, :],
            mask_cache,
            window_size=self.args.sliding_window,
            return_array=True,
        )

        hiddens = []
        for i, (layer, layer_cache) in enumerate(zip(self.layers, cache)):
            h = layer(h, mask, layer_cache, inputs)
            if i in targets:
                hiddens.append(h.mean(axis=2))

        dsv4._materialize_cache_arrays(cache)
        self.main_hidden = mx.concatenate(hiddens, axis=-1) if hiddens else None
        return self.norm(self.hc_head(h))

    cls.__call__ = __call__
    cls._dspark_hidden_patched = True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("--cycles", type=int, default=20)
    ap.add_argument("--prompt", default="The history of the steam engine begins in")
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    from omlx.patches.deepseek_v4 import apply_deepseek_v4_patch

    apply_deepseek_v4_patch()

    from mlx_lm.models.cache import make_prompt_cache
    from mlx_lm.tokenizer_utils import load as load_tokenizer
    from mlx_lm.utils import load_model

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
    print(f"loaded; resident {mx.get_active_memory() / 1024**3:.1f} GiB")

    k = dspark["dspark_block_size"]
    ids = mx.array([tokenizer.encode(args.prompt)])
    cache = make_prompt_cache(model)

    logits = model(ids, cache=cache)
    main_hidden = model.model.main_hidden
    tok = mx.argmax(logits[:, -1:], axis=-1)
    pos = ids.shape[1] - 1  # position of the last token the main model consumed

    # Prefill fills the window with every prompt position at once.
    window = drafter.new_window(1, dtype=mx.bfloat16)
    window = drafter.push_window(window, main_hidden, 0)

    hits = [0] * k
    total = 0
    prefix_hist = [0] * (k + 1)

    for _ in range(args.cycles):
        draft_ids, conf = drafter(
            tok, model.model.embed_tokens, model.lm_head, window, pos
        )
        mx.eval(draft_ids, conf)

        # Ground truth: plain autoregressive decode of the next k tokens. Each
        # forward consumes one position, so push exactly one window slot per step.
        truth = []
        cur = tok
        for _ in range(k):
            logits = model(cur, cache=cache)
            pos += 1
            window = drafter.push_window(window, model.model.main_hidden, pos)
            cur = mx.argmax(logits[:, -1:], axis=-1)
            truth.append(int(cur[0, 0]))

        d = [int(x) for x in draft_ids[0].tolist()]
        run = 0
        for i in range(k):
            if d[i] == truth[i]:
                hits[i] += 1
                if run == i:
                    run = i + 1
        prefix_hist[run] += 1
        total += 1

        # Continue from the main model's tokens: a wrong draft cannot drift.
        tok = cur

    print(f"\ncycles: {total}   block_size: {k}")
    print(f"{'position':>9} {'accept':>8}")
    for i, h in enumerate(hits):
        print(f"{i + 1:>9} {h / total * 100:>7.0f}%")

    exp_prefix = sum(i * n for i, n in enumerate(prefix_hist)) / total
    print(f"\naccepted-prefix histogram: {prefix_hist}")
    print(f"expected accepted prefix : {exp_prefix:.2f} of {k}")
    print(f"=> tokens per cycle      : {exp_prefix + 1:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
