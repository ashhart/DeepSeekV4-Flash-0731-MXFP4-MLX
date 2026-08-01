#!/usr/bin/env python3
"""Quality cost of quantizing `lm_head`, measured rather than eyeballed.

The 1.06 GB bf16 output head is the biggest non-speculative bandwidth lever, but
quantizing it perturbs the logits directly -- unlike the expert/attention weights
it has no downstream layer to absorb the error. This measures that, teacher
forced over real text, against the untouched bf16 head:

  perplexity      the headline number; what actually matters
  top-1 agreement fraction of positions where argmax is unchanged (greedy
                  decoding follows argmax, so this is the practical metric)
  top-5 agreement whether the candidate set is preserved
  mean KL         distribution shift in nats, over the full 129280 vocab

Runs every config against the SAME cached hidden states, so the only thing that
varies is the head.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import mlx.core as mx


def measure(logits, targets, ref_logits=None):
    """NLL + agreement of `logits` against `targets`, optionally vs a reference."""
    logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    nll = -mx.take_along_axis(logprobs, targets[:, None], axis=-1).squeeze(-1)
    out = {"ppl": float(mx.exp(mx.mean(nll)))}

    if ref_logits is not None:
        ref_lp = ref_logits - mx.logsumexp(ref_logits, axis=-1, keepdims=True)
        out["top1"] = float(
            mx.mean(mx.argmax(logits, -1) == mx.argmax(ref_logits, -1))
        )
        k = 5
        top_ref = mx.argpartition(-ref_logits, kth=k - 1, axis=-1)[:, :k]
        top_new = mx.argpartition(-logits, kth=k - 1, axis=-1)[:, :k]
        # Set overlap per row, averaged.
        overlap = mx.sum(
            mx.any(top_ref[:, :, None] == top_new[:, None, :], axis=-1), axis=-1
        )
        out["top5_overlap"] = float(mx.mean(overlap)) / k
        # KL(ref || new): how much probability mass moved.
        out["kl"] = float(mx.mean(mx.sum(mx.exp(ref_lp) * (ref_lp - logprobs), -1)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("--text-file", default=None)
    ap.add_argument("--tokens", type=int, default=2048)
    args = ap.parse_args()

    from omlx.patches.deepseek_v4 import apply_deepseek_v4_patch

    apply_deepseek_v4_patch()

    import mlx.nn as nn
    from mlx_lm.models.cache import make_prompt_cache
    from mlx_lm.tokenizer_utils import load as load_tokenizer
    from mlx_lm.utils import load_model

    model_path = Path(args.model_dir)
    model, _ = load_model(model_path)
    mx.eval(model.parameters())
    tokenizer = load_tokenizer(model_path)

    if args.text_file:
        text = Path(args.text_file).read_text()
    else:
        text = (Path(__file__).parent.parent / "FINDINGS.md").read_text()
    ids = tokenizer.encode(text)[: args.tokens]
    inputs = mx.array([ids[:-1]])
    targets = mx.array(ids[1:])
    print(f"{len(ids) - 1} positions of teacher-forced text\n")

    # Cache the hidden states once: only the head varies between configs.
    cache = make_prompt_cache(model)
    hidden = model.model(inputs, cache)
    mx.eval(hidden)
    hidden2d = hidden[0]

    head = model.lm_head
    ref_logits = head(hidden2d).astype(mx.float32)
    mx.eval(ref_logits)
    base = measure(ref_logits, targets)
    print(f"{'config':<22} {'ppl':>9} {'d%':>7} {'top-1':>8} {'top-5':>8} {'KL':>9}")
    print("-" * 66)
    print(f"{'bf16 (stock)':<22} {base['ppl']:>9.4f} {'--':>7} {'--':>8} {'--':>8} {'--':>9}")

    for bits, group in ((8, 64), (8, 32), (6, 64), (4, 64)):
        q = nn.QuantizedLinear.from_linear(head, group_size=group, bits=bits)
        mx.eval(q.parameters())
        logits = q(hidden2d).astype(mx.float32)
        mx.eval(logits)
        m = measure(logits, targets, ref_logits)
        d = (m["ppl"] / base["ppl"] - 1) * 100
        nbytes = sum(
            v.size * v.dtype.size for v in (q.weight, q.scales, q.biases)
        )
        print(
            f"{f'{bits}-bit g{group} ({nbytes / 1e9:.2f}GB)':<22} "
            f"{m['ppl']:>9.4f} {d:>+6.2f}% {m['top1'] * 100:>7.2f}% "
            f"{m['top5_overlap'] * 100:>7.2f}% {m['kl']:>9.5f}"
        )
        del q, logits
        mx.clear_cache()

    print("\nd% = perplexity change vs the bf16 head. top-1 is the practical")
    print("metric for greedy decoding; KL is in nats over the full vocab.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
