#!/usr/bin/env python3
"""Benchmark the DSpark engine hook through mlx-lm's BatchGenerator.

This is the same code path the oMLX server uses, so it measures the integration
rather than the standalone harness. Runs the identical prompt twice -- hook off,
then hook on -- and reports both throughput and whether the text matches.

    DS4_SPEC is set per-run by this script; do not set it in the environment.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import mlx.core as mx


def run(model, tokenizer, prompt_ids, max_tokens, spec: bool):
    from mlx_lm.generate import BatchGenerator

    from ds4 import engine_hook

    os.environ["DS4_SPEC"] = "1" if spec else "0"

    gen = BatchGenerator(model, sampler=lambda lp: mx.argmax(lp, axis=-1))
    uids = gen.insert([prompt_ids], max_tokens=[max_tokens])

    out, n = [], 0
    t0 = None
    while True:
        responses = gen.next_generated()
        if not responses:
            if not gen.next():
                break
            continue
        for r in responses:
            if t0 is None:  # start timing after the prompt is processed
                t0 = time.perf_counter()
            out.append(r.token)
            n += 1
            if r.finish_reason is not None:
                dt = time.perf_counter() - t0
                return out, n / dt, n
        if n >= max_tokens:
            break
    dt = time.perf_counter() - (t0 or time.perf_counter())
    return out, n / max(dt, 1e-9), n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("--max-tokens", type=int, default=160)
    ap.add_argument("--prompt", default="Write a Python class implementing an LRU cache with get and put.")
    ap.add_argument("--context-file", default=None, help="prepend a file, to test long context")
    ap.add_argument("--context-chars", type=int, default=80000)
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from omlx.patches.deepseek_v4 import apply_deepseek_v4_patch

    apply_deepseek_v4_patch()

    from mlx_lm.tokenizer_utils import load as load_tokenizer
    from mlx_lm.utils import load_model

    from ds4 import engine_hook, windowed_prefill

    windowed_prefill.apply()
    engine_hook.apply()

    model, _ = load_model(Path(args.model_dir))
    # oMLX re-registers the deepseek_v4 module during load, so re-apply the
    # class-level patches against the classes the model was actually built from.
    windowed_prefill.apply()
    engine_hook.apply()
    model._ds4_model_path = str(args.model_dir)
    mx.eval(model.parameters())
    tokenizer = load_tokenizer(Path(args.model_dir))

    # Apply the chat template. Without it the model does freeform completion,
    # which is far higher entropy than real traffic and understates acceptance.
    text = args.prompt
    if args.context_file:
        text = Path(args.context_file).read_text()[: args.context_chars] + "\n\n" + text
    try:
        prompt_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], add_generation_prompt=True
        )
    except Exception:  # noqa: BLE001
        prompt_ids = tokenizer.encode(text)
    print(f"prompt tokens: {len(prompt_ids)}")

    print(f"loaded; resident {mx.get_active_memory() / 1024**3:.1f} GiB\n")

    base_out, base_tps, base_n = run(model, tokenizer, prompt_ids, args.max_tokens, False)
    print(f"hook off : {base_tps:6.1f} tok/s over {base_n} tokens")

    spec_out, spec_tps, spec_n = run(model, tokenizer, prompt_ids, args.max_tokens, True)
    print(f"hook on  : {spec_tps:6.1f} tok/s over {spec_n} tokens   "
          f"({spec_tps / base_tps:.2f}x)")
    print(f"           {engine_hook.stats_summary()}")

    same = base_out[: min(len(base_out), len(spec_out))] == spec_out[: min(len(base_out), len(spec_out))]
    print(f"\nsame tokens: {same}")
    print(f"\n--- hook on output ---\n{tokenizer.decode(spec_out)[:500]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
