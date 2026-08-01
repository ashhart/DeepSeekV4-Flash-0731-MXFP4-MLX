# SPDX-License-Identifier: MIT
"""Decode-shaped fused indexer scores (FINDINGS S28 surviving route).

The MLX fallback materialises relu(q @ pooled^T) as a (B,64,L,P) tensor across
several kernels, then weight-sums heads. This kernel computes the final (L,P)
weighted scores directly at natural decode shapes (L<=8, any P): one simdgroup
per pooled key, 64 head-dots of 128 dims split across 32 lanes.

topk stays on the proven native `dsa_topk_indices`, fed the small score matrix
padded to (64, P%64) with -inf -- padding SCORES is near-free; padding INPUTS
was the falsified 10.5 ms mistake.

Marker: ~/.omlx/ds4_indexer_decode. Exactness: matches the fallback chain to
bf16 tolerance; top-512 sets A/B-checked by the harness quality gate.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import mlx.core as mx

_MARKER = Path.home() / ".omlx" / "ds4_indexer_decode"
_cache = {"at": 0.0, "on": False}
_ENGAGED = False


def _enabled() -> bool:
    now = time.monotonic()
    if now - _cache["at"] > 1.0:
        _cache["on"] = _MARKER.exists()
        _cache["at"] = now
    return _cache["on"]


_SOURCE = """
    // grid: (32*PT, P_tiles, L). One simdgroup per pooled key p.
    // q: (L, H=64, D=128) f32-castable; pooled: (P, D); w: (L, H) prescaled.
    uint lane = thread_position_in_threadgroup.x % 32;
    uint sgid = thread_position_in_threadgroup.x / 32;   // 0..PT-1
    uint p = threadgroup_position_in_grid.y * PT + sgid;
    uint l = threadgroup_position_in_grid.z;
    if (p >= (uint)Pn[0]) return;

    constexpr int H = 64;
    constexpr int D = 128;
    const device T* qrow = q + (l * H) * D;
    const device T* key  = pooled + p * D;

    // each lane owns 4 dims: lane*4 .. lane*4+3
    float k0 = (float)key[lane*4+0], k1 = (float)key[lane*4+1],
          k2 = (float)key[lane*4+2], k3 = (float)key[lane*4+3];

    float acc = 0.0f;
    for (int h = 0; h < H; ++h) {
        const device T* qh = qrow + h * D + lane * 4;
        float part = (float)qh[0]*k0 + (float)qh[1]*k1
                   + (float)qh[2]*k2 + (float)qh[3]*k3;
        float dot = simd_sum(part);
        if (lane == 0) acc += (float)w[l * H + h] * metal::max(dot, 0.0f);
    }
    if (lane == 0) scores[l * (uint)Pn[0] + p] = acc;
"""

_kernel = None


def _get_kernel():
    global _kernel
    if _kernel is None:
        _kernel = mx.fast.metal_kernel(
            name="ds4_indexer_decode_scores",
            input_names=["q", "pooled", "w", "Pn"],
            output_names=["scores"],
            source=_SOURCE,
            ensure_row_contiguous=True,
        )
    return _kernel


def fused_scores(q, pooled, w):
    """q (L,H,D), pooled (P,D), w (L,H) prescaled -> scores (L,P) f32."""
    L, H, D = q.shape
    P = pooled.shape[0]
    PT = 8
    kern = _get_kernel()
    (scores,) = kern(
        inputs=[q, pooled, w, mx.array([P], dtype=mx.int32)],
        template=[("T", q.dtype), ("PT", PT)],
        grid=(32 * PT, (P + PT - 1) // PT, L),
        threadgroup=(32 * PT, 1, 1),
        output_shapes=[(L, P)],
        output_dtypes=[mx.float32],
    )
    return scores


def apply() -> bool:
    dsv4 = sys.modules.get("mlx_lm.models.deepseek_v4")
    if dsv4 is None:
        return False
    cls = getattr(dsv4, "Indexer", None)
    if cls is None or getattr(cls.__call__, "_ds4_indexer_decode", False):
        return cls is not None and getattr(cls.__call__, "_ds4_indexer_decode", False)
    original = cls.__call__

    def wrapped(self, x, q_residual, position_rope, pool_cache, offset):
        L = x.shape[1]
        if not (_enabled() and 1 <= L < 64 and x.shape[0] == 1
                and self.head_dim == 128 and self.n_heads == 64):
            return original(self, x, q_residual, position_rope, pool_cache, offset)
        try:
            from omlx.custom_kernels.glm_moe_dsa import fast as glm_fast
            if not glm_fast.has_symbol("dsa_topk_indices"):
                return original(self, x, q_residual, position_rope, pool_cache, offset)

            pooled = self.compressor(x, pool_cache, offset)  # REAL x only
            P = pooled.shape[1]
            if P <= self.index_topk:
                return original(self, x, q_residual, position_rope, pool_cache, offset)

            global _ENGAGED
            if not _ENGAGED:
                _ENGAGED = True
                try:
                    import logging
                    logging.getLogger("omlx.ds4").info("decode indexer ENGAGED")
                except Exception:  # noqa: BLE001
                    pass

            q = self.wq_b(q_residual).reshape(1, L, self.n_heads, self.head_dim)
            q = q.transpose(0, 2, 1, 3)
            q = position_rope(q, offset)              # (1,H,L,D)
            qf = q[0].transpose(1, 0, 2)              # (L,H,D)
            w = self.weights_proj(x)[0].astype(q.dtype) * (
                (self.n_heads ** -0.5) * self.scale
            )                                          # (L,H)

            scores = fused_scores(qf, pooled[0], w)   # (L,P) f32

            pmask = pool_cache.make_mask(L, offset) if pool_cache is not None else None
            if pmask is not None:
                pm = pmask if pmask.ndim == 2 else pmask[0]
                scores = mx.where(pm, scores, mx.finfo(scores.dtype).min)

            # Pad only the SCORE matrix for the proven native topk.
            LP, PP = 64, ((P + 63) // 64) * 64
            padded = mx.full((1, 1, LP, PP), mx.finfo(scores.dtype).min,
                             dtype=scores.dtype)
            padded[0, 0, :L, :P] = scores
            out = glm_fast.dsa_topk_indices(padded, self.index_topk, bucketed=True)[:, 0]
            return out[:, :L] if out.shape[1] == LP else out[..., :L, :]
        except Exception:  # noqa: BLE001
            return original(self, x, q_residual, position_rope, pool_cache, offset)

    wrapped._ds4_indexer_decode = True
    cls.__call__ = wrapped
    return True
