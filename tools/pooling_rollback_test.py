#!/usr/bin/env python3
"""Can a PoolingCache be rolled back after a k-token verify? Test before fixing.

`PoolingCache.trim(n)` refuses (`_can_undo`) whenever the replayed confirmed
prefix would complete a pool window, because `trim` discards what
`accumulate_windows` returns -- and that return value is the completed window the
Compressor must compress and append to `pooled`. Dropping it silently loses a
pooled entry.

The fix is to replay *through the compressor*. This script checks that claim the
only way that means anything: build the cache two ways and compare state.

    A. feed prefix, then k+1 tokens (verify), then roll back to n+1 accepted
    B. feed prefix, then the same n+1 tokens directly

If rollback is correct, A and B must agree on `pooled`, `remainder`, the live
part of `buf_kv`, and `offset`.

Runs on a synthetic Compressor with tiny dims -- no 146 GiB model load -- so both
ratio 4 (overlap) and ratio 128 (simple) can be checked in seconds.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

import mlx.core as mx


@dataclass
class TinyArgs:
    hidden_size: int = 64
    qk_rope_head_dim: int = 8
    rms_norm_eps: float = 1e-6
    compress_rope_theta: float = 160000.0
    rope_scaling: object = None
    max_position_embeddings: int = 4096


def build(ratio, head_dim, dsv4):
    return dsv4.Compressor(TinyArgs(), ratio, head_dim)


def snapshot(c):
    return {
        "offset": c.offset,
        "remainder": c.remainder,
        "pooled": None if c.pooled is None else mx.array(c.pooled),
        "buf": None
        if c.buf_kv is None or c.remainder == 0
        else mx.array(c.buf_kv[:, : c.remainder]),
    }


def compare(a, b):
    out = []
    for key in ("offset", "remainder"):
        if a[key] != b[key]:
            out.append(f"{key}: rollback={a[key]} reference={b[key]}")
    for key in ("pooled", "buf"):
        x, y = a[key], b[key]
        if (x is None) != (y is None):
            out.append(f"{key}: rollback={'None' if x is None else x.shape} "
                       f"reference={'None' if y is None else y.shape}")
        elif x is not None:
            if x.shape != y.shape:
                out.append(f"{key} shape: {x.shape} vs {y.shape}")
            else:
                d = float(mx.max(mx.abs(x - y)))
                if d > 1e-4:
                    out.append(f"{key}: max|diff| = {d:.6f}")
    return out


def run(ratio, prefix_len, k, n, dsv4, PoolingCache):
    head_dim = 16
    mx.random.seed(0)
    comp = build(ratio, head_dim, dsv4)
    mx.eval(comp.parameters())

    mx.random.seed(1)
    xs = mx.random.normal((1, prefix_len + k + 1, TinyArgs.hidden_size))

    # --- A: verify then roll back -------------------------------------
    ca = PoolingCache(ratio)
    comp(xs[:, :prefix_len], ca, 0)
    verify = xs[:, prefix_len : prefix_len + k + 1]
    comp(verify, ca, prefix_len)
    reject = k - n
    trimmed = ca.trim(reject)
    a = snapshot(ca)

    # --- B: reference, only the accepted tokens ------------------------
    cb = PoolingCache(ratio)
    comp(xs[:, :prefix_len], cb, 0)
    comp(xs[:, prefix_len : prefix_len + n + 1], cb, prefix_len)
    b = snapshot(cb)

    diffs = compare(a, b)
    status = "OK" if (trimmed == reject and not diffs) else "FAIL"
    if trimmed != reject:
        status = f"REFUSED (trim returned {trimmed}, wanted {reject})"
    print(f"  ratio={ratio:>3} prefix={prefix_len:>4} k={k} accept={n} -> {status}")
    for d in diffs:
        print(f"       {d}")
    return status == "OK"


def main() -> int:
    from omlx.patches.deepseek_v4 import apply_deepseek_v4_patch

    apply_deepseek_v4_patch()
    import mlx_lm.models.deepseek_v4 as dsv4
    from mlx_lm.models.cache import PoolingCache

    print("PoolingCache rollback after a k-token verify\n")
    ok = 0
    total = 0
    for ratio in (128, 4):
        for prefix_len in (100, 127, 200, 253):
            for n in range(0, 5):
                total += 1
                ok += run(ratio, prefix_len, 5, n, dsv4, PoolingCache)
        print()
    print(f"{ok}/{total} rollbacks exact")
    return 0 if ok == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
