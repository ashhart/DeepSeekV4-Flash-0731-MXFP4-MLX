#!/usr/bin/env python3
"""Reproduce the RotatingKVCache rollback bug without loading a model.

Mirrors what a speculative cycle does to the cache:

    prefill (S=P, _update_concat)
      -> a few decode steps (S=1, _update_in_place)
      -> verify (S=k+1, _update_concat)
      -> roll back the rejected tail
      -> decode again (S=1)

and prints (offset, keys.shape[2], _idx) plus the mask width the model would
build, at every stage. The failure is a mask sized for `offset` keys against a
cache holding fewer.
"""

from __future__ import annotations

import mlx.core as mx
from mlx_lm.models.cache import RotatingKVCache

MAX = 128
HEAD_DIM = 8


def kv(n):
    return mx.zeros((1, 1, n, HEAD_DIM))


def show(tag, c):
    keys = 0 if c.keys is None else c.keys.shape[2]
    print(f"{tag:<34} offset={c.offset:>4} keys={keys:>4} _idx={c._idx:>4}")


def trim_rotating(c, n):
    if n <= 0 or c.keys is None:
        return
    c.keys = c.keys[..., :-n, :]
    c.values = c.values[..., :-n, :]
    c.offset -= n
    c._idx = c.keys.shape[2]


def main() -> int:
    for prompt_len in (15, 200):
        print(f"\n===== prompt {prompt_len} (max_size {MAX}) =====")
        c = RotatingKVCache(max_size=MAX)

        k, _ = c.update_and_fetch(kv(prompt_len), kv(prompt_len))
        show("after prefill", c)
        print(f"   fetched keys = {k.shape[2]}")

        for i in range(3):
            k, _ = c.update_and_fetch(kv(1), kv(1))
        show("after 3 decode steps", c)
        print(f"   fetched keys = {k.shape[2]}")

        before = (c.offset, c.keys.shape[2], c._idx)
        k, _ = c.update_and_fetch(kv(6), kv(6))
        show("after 6-token verify", c)
        print(f"   fetched keys = {k.shape[2]}")

        trim_rotating(c, 3)  # rejected 3 of 5 drafts
        show("after trim(3)", c)

        k, _ = c.update_and_fetch(kv(1), kv(1))
        show("after next decode step", c)
        print(f"   fetched keys = {k.shape[2]}")

        mask_w = min(c.offset, MAX)
        print(f"   model would build mask width ~{mask_w} vs fetched {k.shape[2]}"
              f"   {'MISMATCH' if mask_w != k.shape[2] else 'ok'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
