# SPDX-License-Identifier: MIT
"""Un-gate the native indexer kernels at decode via pad-to-shape (FINDINGS S25).

`Indexer.__call__` only uses the fast `dsa_indexer_scores`/`dsa_topk_indices`
kernels when `L % 64 == 0 and pooled % 64 == 0` -- prefill shapes. Every decode
verify (L=4-6) therefore runs the generic MLX fallback over ~5.9K pooled keys
x 64 heads x 21 layers: the located +7.4 ms/cycle context term.

This wrapper pads: q rows L -> 64 (zero queries; rows sliced off after) and
pooled -> next %64 (rows masked to -inf via pmask so topk never selects them).
CRITICAL: the compressor runs on the REAL x before any padding -- padding must
never reach pool_cache state.

Marker: ~/.omlx/ds4_indexer_pad (1 s TTL). Correctness gate: padded-key indices
must never appear in output (asserted in the unit test, not in the hot path).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import mlx.core as mx

_MARKER = Path.home() / ".omlx" / "ds4_indexer_pad"
_cache = {"at": 0.0, "on": False}
_ENGAGED = False


def _enabled() -> bool:
    now = time.monotonic()
    if now - _cache["at"] > 1.0:
        _cache["on"] = _MARKER.exists()
        _cache["at"] = now
    return _cache["on"]


def apply() -> bool:
    dsv4 = sys.modules.get("mlx_lm.models.deepseek_v4")
    if dsv4 is None:
        return False
    cls = getattr(dsv4, "Indexer", None)
    if cls is None or getattr(cls.__call__, "_ds4_indexer_pad", False):
        return cls is not None and getattr(cls.__call__, "_ds4_indexer_pad", False)

    original = cls.__call__

    def wrapped(self, x, q_residual, position_rope, pool_cache, offset):
        L = x.shape[1]
        if not (_enabled() and 1 <= L < 64):
            return original(self, x, q_residual, position_rope, pool_cache, offset)
        try:
            from omlx.custom_kernels.glm_moe_dsa import fast as glm_fast

            if not (glm_fast.has_symbol("dsa_indexer_scores")
                    and glm_fast.has_symbol("dsa_topk_indices")):
                return original(self, x, q_residual, position_rope, pool_cache, offset)

            global _ENGAGED
            if not _ENGAGED:
                _ENGAGED = True
                try:
                    import logging
                    logging.getLogger("omlx.ds4").info("indexer pad ENGAGED")
                except Exception:  # noqa: BLE001
                    pass
            B = x.shape[0]
            # Real x through the compressor FIRST: cache state must stay clean.
            pooled = self.compressor(x, pool_cache, offset)
            P = pooled.shape[1]
            if P <= self.index_topk or self.head_dim != 128 or self.n_heads not in (32, 64):
                # Fallback tail, replicated (small P or unsupported geometry).
                return original(self, x, q_residual, position_rope, pool_cache, offset)

            q = self.wq_b(q_residual).reshape(B, L, self.n_heads, self.head_dim)
            q = q.transpose(0, 2, 1, 3)
            q = position_rope(q, offset)

            pmask = pool_cache.make_mask(L, offset) if pool_cache is not None else None

            LP = 64
            PP = ((P + 63) // 64) * 64
            qp = mx.concatenate(
                [q, mx.zeros((B, self.n_heads, LP - L, self.head_dim), dtype=q.dtype)],
                axis=2,
            )
            pooledp = (
                mx.concatenate(
                    [pooled, mx.zeros((B, PP - P, pooled.shape[-1]), dtype=pooled.dtype)],
                    axis=1,
                )
                if PP > P
                else pooled
            )
            wp = self.weights_proj(x).astype(q.dtype) * ((self.n_heads ** -0.5) * self.scale)
            wp = mx.concatenate(
                [wp, mx.zeros((B, LP - L, wp.shape[-1]), dtype=wp.dtype)], axis=1
            )

            scores4 = glm_fast.dsa_indexer_scores(qp, pooledp[:, None], wp, causal=False)

            # Validity over padded keys: real pmask columns, -inf on pad columns
            # and don't care on pad rows (sliced off).
            valid = mx.zeros((LP, PP), dtype=mx.bool_)
            if pmask is not None:
                pm = pmask if pmask.ndim == 2 else pmask[0]
                valid[:L, :P] = pm
            else:
                valid[:L, :P] = True
            scores4 = mx.where(valid[None, None] if scores4.ndim == 4 else valid[None],
                               scores4, mx.finfo(scores4.dtype).min)

            out = glm_fast.dsa_topk_indices(scores4, self.index_topk, bucketed=True)[:, 0]
            return out[:, :L] if out.shape[1] == LP else out[..., :L, :]
        except Exception:  # noqa: BLE001 -- any doubt -> proven fallback
            return original(self, x, q_residual, position_rope, pool_cache, offset)

    wrapped._ds4_indexer_pad = True
    cls.__call__ = wrapped
    return True
