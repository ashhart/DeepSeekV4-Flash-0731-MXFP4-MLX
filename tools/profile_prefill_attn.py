#!/usr/bin/env python3
"""Break down attention at prefill lengths, from a real forward pass.

MoE scales linearly and runs near peak; attention is ~71% of prefill and grows
superlinearly. Rather than rebuild masks by hand (the rotating cache expects
L + window-1 keys), this patches each component class and times a genuine
prefill. At prefill each call takes milliseconds, so the `mx.eval` barrier
(~0.06 ms) is negligible -- unlike the decode profile, these numbers are sound.

Any component whose ms-per-1k-tokens column climbs with L is superlinear.
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path

import mlx.core as mx

COMPONENTS = (
    "LocalAttention",
    "CompressedAttention",
    "SparseCompressedAttention",
    "Indexer",
    "Compressor",
    "DeepseekV4MoE",
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("--lengths", default="1024,2048,4096")
    args = ap.parse_args()

    from omlx.patches.deepseek_v4 import apply_deepseek_v4_patch

    apply_deepseek_v4_patch()

    import mlx_lm.models.deepseek_v4 as dsv4
    from mlx_lm.models.cache import make_prompt_cache
    from mlx_lm.utils import load_model

    model, _ = load_model(Path(args.model_dir))
    mx.eval(model.parameters())
    print(f"loaded; resident {mx.get_active_memory() / 1024**3:.1f} GiB\n")

    totals: dict = defaultdict(float)
    counts: dict = defaultdict(int)

    present = [c for c in COMPONENTS if getattr(dsv4, c, None) is not None]
    originals = {}
    for name in present:
        cls = getattr(dsv4, name)
        originals[name] = cls.__call__

        def make(nm, fn):
            def wrapper(self, *a, **kw):
                t = time.perf_counter()
                out = fn(self, *a, **kw)
                mx.eval(out)
                totals[nm] += time.perf_counter() - t
                counts[nm] += 1
                return out

            return wrapper

        cls.__call__ = make(name, originals[name])

    results = {}
    for L in [int(x) for x in args.lengths.split(",")]:
        ids = mx.zeros((1, L), dtype=mx.int32)
        cache = make_prompt_cache(model)
        mx.eval(model(ids, cache=cache))  # warm
        del cache
        mx.clear_cache()

        totals.clear()
        counts.clear()
        cache = make_prompt_cache(model)
        t0 = time.perf_counter()
        mx.eval(model(ids, cache=cache))
        wall = time.perf_counter() - t0
        results[L] = (dict(totals), dict(counts), wall)
        del cache
        mx.clear_cache()

    for name, fn in originals.items():
        getattr(dsv4, name).__call__ = fn

    lens = sorted(results)
    print(f"{'component':<28} " + "".join(f"{L:>10} " for L in lens))
    print(f"{'  (ms per 1k tokens)':<28} " + "-" * (11 * len(lens)))
    names = sorted(present, key=lambda n: -results[lens[-1]][0].get(n, 0))
    for name in names:
        row = ""
        for L in lens:
            t = results[L][0].get(name, 0.0)
            row += f"{t * 1e3 / (L / 1000):>10.0f} "
        print(f"{name:<28} {row}")

    print()
    row_total = "".join(
        f"{results[L][2] * 1e3 / (L / 1000):>10.0f} " for L in lens
    )
    print(f"{'FULL FORWARD':<28} {row_total}")
    row_acct = ""
    for L in lens:
        acct = sum(
            v for k, v in results[L][0].items() if k != "SparseCompressedAttention"
        ) + results[L][0].get("SparseCompressedAttention", 0)
        row_acct += f"{acct * 1e3 / (L / 1000):>10.0f} "
    print(f"{'  (sum of components)':<28} {row_acct}")

    print(f"\ncalls per forward at L={lens[-1]}: {results[lens[-1]][1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
