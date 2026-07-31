#!/usr/bin/env python3
"""Load DeepSeek-V4-Flash MXFP4/MXFP8 under the oMLX mlx-lm patches and report
decode throughput.

Run with the oMLX-bundled interpreter so the patched mlx_lm and the `omlx`
package are both importable:

    R=/Applications/oMLX.app/Contents/Resources
    PYTHONPATH=$R:$R/Python/framework-mlx-base/lib/python3.11/site-packages \
      $R/Python/cpython-3.11/bin/python3.11 bench_ds4.py <model_dir> [--structural]

--structural stops after `load_weights` shape validation (lazy, no mx.eval), which
is enough to prove the quantization spec resolves and costs seconds instead of
minutes.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mlx.core as mx


def human(n: float) -> str:
    return f"{n / 1024**3:.1f} GiB"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("--structural", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--prompt", default="Write a Python function that merges two sorted lists.")
    args = ap.parse_args()

    from omlx.patches.deepseek_v4 import apply_deepseek_v4_patch

    apply_deepseek_v4_patch()

    from mlx_lm.utils import load_model

    model_path = Path(args.model_dir)

    t0 = time.perf_counter()
    model, config = load_model(model_path, lazy=args.structural)
    t_load = time.perf_counter() - t0
    print(f"load_model ok in {t_load:.1f}s  (lazy={args.structural})")

    # Report what the quantization spec actually resolved to, per mode.
    import mlx.nn as nn
    from mlx.utils import tree_flatten

    modes: dict[str, int] = {}
    for _, m in tree_flatten(model.leaf_modules(), is_leaf=nn.Module.is_module):
        mode = getattr(m, "mode", None)
        if mode is not None:
            key = f"{mode}/{getattr(m, 'bits', '?')}bit"
            modes[key] = modes.get(key, 0) + 1
    print(f"quantized modules resolved: {modes}")

    if args.structural:
        print("structural check passed -- shapes agree with the checkpoint")
        return 0

    mx.eval(model.parameters())
    print(f"weights resident: {human(mx.get_active_memory())}")

    # oMLX's tokenizer patch wraps `tokenizer_utils.load` to inject the DSML
    # chat template and tool parser for deepseek_v4.
    from mlx_lm.tokenizer_utils import load as load_tokenizer

    tokenizer = load_tokenizer(model_path)

    from mlx_lm.generate import stream_generate

    print(f"\nprompt: {args.prompt!r}")
    text, n_decode = [], 0
    prompt_tps = decode_tps = 0.0
    for resp in stream_generate(
        model, tokenizer, args.prompt, max_tokens=args.max_tokens
    ):
        text.append(resp.text)
        n_decode = resp.generation_tokens
        prompt_tps, decode_tps = resp.prompt_tps, resp.generation_tps

    print("".join(text))
    print(
        f"\n--- prompt {prompt_tps:.1f} tok/s | decode {decode_tps:.1f} tok/s "
        f"over {n_decode} tokens | peak {human(mx.get_peak_memory())}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
