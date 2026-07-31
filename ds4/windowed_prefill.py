# SPDX-License-Identifier: MIT
"""Windowed (blocked) prefill attention for DeepSeek-V4.

Problem
-------
At one-shot prefill `RotatingKVCache._update_concat` returns the whole sequence
(S == L), and the layer then builds a dense `(L, L)` causal+window mask and runs
a full `L x L` SDPA -- even though `sliding_window` is 128. Almost every score is
computed and then masked away, so a sliding-window model pays quadratic cost.

Fix
---
Block the queries. For query block `[i, j)` only keys within `window` of it can
be unmasked, so each block costs `block x (block + w)` instead of `block x L`:

    total  L * (block + w)   instead of   L * L
    L=8192, block=512, w=128  ->  640 vs 8192 keys  (~12.8x less attention work)

Masks are built per block from index arithmetic, so the dense `(L, L)` mask is
never materialised either.

Why not just chunk the prompt (`scheduler.chunked_prefill`)? Chunking shrinks L
for the *whole layer*, and the MoE is markedly less efficient at small L (1812 vs
857 ms per 1k tokens at L=1024 vs 4096). Blocking inside attention keeps the MoE
at full width, so the two are not equivalent.

`CompressedAttention` also concatenates a pooled KV segment that every query may
attend to. That segment is only `L/compress_ratio` long, so it stays dense and is
appended to each block's key set.
"""

from __future__ import annotations

from typing import Optional

import mlx.core as mx

# Below this the dense path is already cheap and blocking only adds overhead.
MIN_PREFILL_LEN = 1024
# Measured on M3 Ultra at L=8192/16384: 256 edges out 512 (1.74x/3.25x vs
# 1.72x/3.20x) and clearly beats 1024. Smaller blocks waste less work on the
# window overhang; below 256 the per-block dispatch overhead starts to bite.
DEFAULT_BLOCK = 256

_PATCHED = False


def blocked_window_attention(
    q: mx.array,
    kv: mx.array,
    *,
    scale: float,
    window: int,
    q_start: int,
    kv_start: int,
    sinks: Optional[mx.array],
    n_local: int,
    pooled_mask: Optional[mx.array] = None,
    block: int = DEFAULT_BLOCK,
) -> mx.array:
    """Blocked sliding-window attention.

    q         (B, H, L, D), queries at absolute positions `q_start + [0, L)`
    kv        (B, Hkv, n_local + P, D); local segment first, pooled segment after
    kv_start  absolute position of `kv[..., 0, :]`
    pooled_mask  (L, P) validity of the pooled segment per query, or None
    """
    from mlx_lm.models.base import scaled_dot_product_attention

    L = q.shape[2]
    n_pooled = kv.shape[2] - n_local
    pooled = kv[:, :, n_local:, :] if n_pooled > 0 else None

    outs = []
    for i in range(0, L, block):
        j = min(i + block, L)

        # Queries [i, j) sit at absolute [q_start+i, q_start+j); a causal window
        # of `window` means keys in absolute [q_start+i-window+1, q_start+j).
        lo = max(0, q_start + i - window + 1 - kv_start)
        hi = min(n_local, q_start + j - kv_start)
        if hi <= lo:
            outs.append(mx.zeros_like(q[:, :, i:j, :]))
            continue

        qa = mx.arange(q_start + i, q_start + j)[:, None]
        ka = mx.arange(kv_start + lo, kv_start + hi)[None, :]
        m = (ka <= qa) & (ka > qa - window)

        k_blk = kv[:, :, lo:hi, :]
        if pooled is not None:
            k_blk = mx.concatenate([k_blk, pooled], axis=2)
            pm = (
                pooled_mask[i:j]
                if pooled_mask is not None
                else mx.ones((j - i, n_pooled), dtype=mx.bool_)
            )
            m = mx.concatenate([m, pm], axis=1)

        outs.append(
            scaled_dot_product_attention(
                q[:, :, i:j, :],
                k_blk,
                k_blk,
                cache=None,
                scale=scale,
                mask=m,
                sinks=sinks,
            )
        )

    return mx.concatenate(outs, axis=2)


def _project_q_kv(self, x, offset):
    B, L, _ = x.shape
    q = self.wq_b(self.q_norm(self.wq_a(x)))
    q = q.reshape(B, L, self.n_heads, self.head_dim)
    q = mx.fast.rms_norm(q, None, self.config.rms_norm_eps)
    q = q.transpose(0, 2, 1, 3)
    q = self.rope(q, offset)

    kv = self.kv_norm(self.wkv(x)).reshape(B, 1, L, self.head_dim)
    kv = self.rope(kv, offset)
    return q, kv


def _project_out(self, out, offset, L):
    B = out.shape[0]
    out = self.rope(out, offset, inverse=True)
    out = out.reshape(B, self.o_groups, -1, L, self.head_dim)
    out = out.transpose(0, 1, 3, 2, 4).flatten(-2)
    out = self.wo_a(out)
    out = out.transpose(0, 2, 1, 3).flatten(-2)
    out = self.wo_b(out)
    if self.sharding_group is not None:
        out = mx.distributed.all_sum(out, group=self.sharding_group)
    return out


def apply(block: int = DEFAULT_BLOCK, min_len: int = MIN_PREFILL_LEN) -> bool:
    """Patch LocalAttention / CompressedAttention with the blocked prefill path."""
    global _PATCHED
    if _PATCHED:
        return False

    import mlx_lm.models.deepseek_v4 as dsv4
    from mlx_lm.models.base import scaled_dot_product_attention

    local_orig = dsv4.LocalAttention.__call__
    comp_orig = dsv4.CompressedAttention.__call__

    def local_call(self, x, mask=None, cache=None):
        B, L, _ = x.shape
        if L < min_len:
            return local_orig(self, x, mask=mask, cache=cache)

        offset = int(cache.offset) if cache is not None else 0
        q, kv = _project_q_kv(self, x, offset)
        if cache is not None:
            kv, _ = cache.update_and_fetch(kv, mx.zeros((B, 1, L, 0)))

        out = blocked_window_attention(
            q,
            kv,
            scale=self.scale,
            window=self.config.sliding_window,
            q_start=offset,
            kv_start=offset + L - kv.shape[2],
            sinks=self.attn_sink.astype(q.dtype),
            n_local=kv.shape[2],
            block=block,
        )
        return _project_out(self, out, offset, L)

    def compressed_call(self, x, mask=None, cache=None):
        B, L, _ = x.shape
        if L < min_len:
            return comp_orig(self, x, mask=mask, cache=cache)

        local_cache = cache[0] if cache is not None else None
        pool_cache = cache[1] if cache is not None else None
        offset = int(local_cache.offset) if local_cache is not None else 0

        q, kv = _project_q_kv(self, x, offset)
        if local_cache is not None:
            kv, _ = local_cache.update_and_fetch(kv, mx.zeros((B, 1, L, 0)))
        n_local = kv.shape[2]
        kv_start = offset + L - n_local

        pooled = self.compressor(x, pool_cache, offset)
        pooled_mask = None
        if pooled.shape[1] > 0:
            if pool_cache is not None:
                pooled_mask = pool_cache.make_mask(L, offset)
            kv = mx.concatenate([kv, pooled[:, None]], axis=2)

        out = blocked_window_attention(
            q,
            kv,
            scale=self.scale,
            window=self.config.sliding_window,
            q_start=offset,
            kv_start=kv_start,
            sinks=self.attn_sink.astype(q.dtype),
            n_local=n_local,
            pooled_mask=pooled_mask,
            block=block,
        )
        return _project_out(self, out, offset, L)

    dsv4.LocalAttention.__call__ = local_call
    dsv4.CompressedAttention.__call__ = compressed_call
    _PATCHED = True
    return True
