#!/usr/bin/env python3
"""Reproduce the speculative rollback desync in isolation. No model load.

The server uses **BatchRotatingKVCache**, not the plain `RotatingKVCache` that
earlier isolation tests used. Its `offset` is an `mx.array` mutated in place and
it carries extra state (`_offset`, `rotated`, `left_padding`), so it is a
genuinely different state machine -- and it is the one that desyncs.

Replays exactly what a speculative cycle does:

    prefill -> some single-token decodes -> armed k+1 verify -> trim(k-n)

and checks the cache ends up where a plain sequence of n+1 single-token decodes
would have left it. Any mismatch here is the corruption that shows up as fluent
repetition in generated text.
"""

from __future__ import annotations

import mlx.core as mx

HEAD_DIM = 8
MAX = 128


def kv(n):
    return mx.zeros((1, 1, n, HEAD_DIM))


def off(c) -> int:
    o = c.offset
    return int(o.item()) if hasattr(o, "item") else int(o)


def describe(c) -> str:
    keys = 0 if c.keys is None else c.keys.shape[2]
    return f"offset={off(c):>4} keys={keys:>4} _idx={getattr(c, '_idx', '?')}"


def run_case(cls, name, prefill_len, decodes, k, n, armed):
    from omlx.patches.mlx_lm_mtp import cache_rollback

    cache_rollback.apply()

    def fresh():
        try:
            return cls(max_size=MAX)
        except TypeError:
            return cls(max_size=MAX, left_padding=[0])

    # --- speculative path -------------------------------------------------
    a = fresh()
    a.update_and_fetch(kv(prefill_len), kv(prefill_len))
    for _ in range(decodes):
        a.update_and_fetch(kv(1), kv(1))

    cache_rollback.set_undo_armed(armed)
    a.update_and_fetch(kv(k + 1), kv(k + 1))   # the verify
    cache_rollback.set_undo_armed(False)

    trimmed = a.trim(k - n) if k - n > 0 else 0

    # --- reference: only the committed tokens ------------------------------
    b = fresh()
    b.update_and_fetch(kv(prefill_len), kv(prefill_len))
    for _ in range(decodes):
        b.update_and_fetch(kv(1), kv(1))
    b.update_and_fetch(kv(n + 1), kv(n + 1))

    ok = off(a) == off(b)
    status = "OK" if ok else "DESYNC"
    if k - n > 0 and trimmed != k - n:
        status = f"TRIM REFUSED (returned {trimmed}, wanted {k - n})"
    print(f"  {name:<22} prefill={prefill_len:>4} k={k} n={n} armed={armed} -> {status}")
    if not ok:
        print(f"      speculative: {describe(a)}")
        print(f"      reference  : {describe(b)}")
    return ok


def main() -> int:
    from mlx_lm.models import cache as cache_mod

    classes = [
        (getattr(cache_mod, n, None), n)
        for n in ("RotatingKVCache", "BatchRotatingKVCache")
    ]

    total = passed = 0
    for cls, name in classes:
        if cls is None:
            print(f"{name}: not available")
            continue
        print(f"\n=== {name} ===")
        for prefill_len in (100, 200):
            for n in (0, 1, 2, 3):
                for armed in (True,):
                    total += 1
                    passed += run_case(cls, name, prefill_len, 3, 3, n, armed)

    print(f"\n{passed}/{total} rollbacks land where a plain decode would")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
