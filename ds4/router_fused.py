# SPDX-License-Identifier: MIT
"""Fused DeepSeek-V4 MoE router (attack #4).

The stock `_expert_select` is an mx.compile'd chain -- cast, sqrtsoftplus,
bias-add, argpartition(256), take_along_axis, sum, divide, scale -- costing
5.13 ms per L=4 target (12.3% of target GPU) across 43 layers despite reading
almost nothing: it is dispatch-bound, the Metal analogue of the launch overhead
the DGX recipes remove with CUDA graphs and prewarmed route-packs.

This replaces the whole chain with ONE kernel per call: one threadgroup per
token row, 256 threads (one per expert), top-k by iterated threadgroup argmax.

Numerics
--------
scores = sqrt(log1p(exp(x))) computed as softplus(x) = max(x,0) + log1p(exp(-|x|))
(the numerically stable form; matches mx.softplus to 1 ULP on the tested range).
Selection order is DESCENDING by biased score with lowest-index tie-break.
`argpartition`'s within-k order is unspecified, so per-position order may differ
from stock; the weighted expert SUM is order-invariant up to 1 ULP, which lands
inside the model's established batching noise. The A/B gate decides.

Scope: non-hash layers only (layers >= num_hash_layers). Hash layers route via
tid2eid lookup and stay stock.

Gating: ~/.omlx/ds4_router_fused, checked with a 1 s TTL cache so interleaved
same-process A/B works without a per-call stat storm.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Optional

import mlx.core as mx

_MARKER = Path.home() / ".omlx" / "ds4_router_fused"
_marker_cache = {"at": 0.0, "on": False}

E = 256   # n_routed_experts
K = 6     # num_experts_per_tok


def _enabled() -> bool:
    now = time.monotonic()
    if now - _marker_cache["at"] > 1.0:
        _marker_cache["on"] = _MARKER.exists()
        _marker_cache["at"] = now
    return _marker_cache["on"]


_SOURCE = """
    // One threadgroup per token row; 256 threads, one per expert.
    // NOTE mx.fast.metal_kernel's grid is measured in THREADS (dispatchThreads
    // semantics): grid=(E, rows, 1) with threadgroup=(E, 1, 1) yields one
    // 256-thread group per row on the y axis. grid=(rows,1,1) launched `rows`
    // threads total and every threadgroup buffer beyond them was uninitialised
    // garbage -- the unit test's 0/244.
    uint row  = threadgroup_position_in_grid.y;
    uint tid  = thread_position_in_threadgroup.x;
    uint lane = tid % 32;
    uint sg   = tid / 32;

    constexpr int NE = 256;
    constexpr int NK = 6;

    threadgroup float tg_scores[NE];
    threadgroup float tg_best_val[8];
    threadgroup uint  tg_best_idx[8];
    threadgroup float tg_sel_score[NK];
    threadgroup uint  tg_sel_idx[NK];

    // sqrtsoftplus, stable form. Matches mx.softplus's max+log1p formulation.
    float x = (float)logits[row * NE + tid];
    float sp = metal::max(x, 0.0f) + metal::log(1.0f + metal::exp(-metal::abs(x)));
    float score = metal::sqrt(sp);
    float biased = score + (float)bias[tid];

    tg_scores[tid] = score;
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);

    // Iterated argmax: k rounds, each a simd reduction then a cross-simd pick.
    // Tie-break: lowest expert index wins (encode index into the comparison).
    float my_val = biased;
    for (int k = 0; k < NK; ++k) {
        // simd-level argmax with lowest-index tie-break
        float v = my_val;
        uint  i = tid;
        for (uint off = 16; off > 0; off >>= 1) {
            float ov = simd_shuffle_down(v, off);
            uint  oi = simd_shuffle_down(i, off);
            // Metal defines shuffle-down reads from lanes >= 32 as UNDEFINED;
            // without this guard high lanes adopt garbage and the argmax is
            // nondeterministic (unit test: 0/244 self-consistent).
            if (lane + off < 32 && (ov > v || (ov == v && oi < i))) { v = ov; i = oi; }
        }
        if (lane == 0) { tg_best_val[sg] = v; tg_best_idx[sg] = i; }
        threadgroup_barrier(metal::mem_flags::mem_threadgroup);

        if (tid == 0) {
            float bv = tg_best_val[0]; uint bi = tg_best_idx[0];
            for (uint s = 1; s < 8; ++s) {
                float sv = tg_best_val[s]; uint si = tg_best_idx[s];
                if (sv > bv || (sv == bv && si < bi)) { bv = sv; bi = si; }
            }
            tg_sel_idx[k] = bi;
            tg_sel_score[k] = tg_scores[bi];
        }
        threadgroup_barrier(metal::mem_flags::mem_threadgroup);

        // Winner removes itself from the next round.
        if (tid == tg_sel_idx[k]) my_val = -INFINITY;
        threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    }

    // Normalize the K selected scores and write out.
    if (tid < NK) {
        float total = 0.0f;
        for (int k = 0; k < NK; ++k) total += tg_sel_score[k];
        float w = tg_sel_score[tid] / (total + 1e-20f) * scale[0];
        inds[row * NK + tid] = (int)tg_sel_idx[tid];
        weights[row * NK + tid] = w;
    }
"""

_kernel = None


def _get_kernel():
    global _kernel
    if _kernel is None:
        _kernel = mx.fast.metal_kernel(
            name="ds4_router_topk",
            input_names=["logits", "bias", "scale"],
            output_names=["inds", "weights"],
            source=_SOURCE,
            ensure_row_contiguous=True,
        )
    return _kernel


_ENGAGED = False


def fused_expert_select(logits: mx.array, bias: mx.array,
                        routed_scaling_factor: float):
    global _ENGAGED
    if not _ENGAGED:
        _ENGAGED = True
        try:
            import logging
            logging.getLogger("omlx.ds4").info("fused router ENGAGED")
        except Exception:  # noqa: BLE001
            pass
    rows = 1
    for d in logits.shape[:-1]:
        rows *= d
    kern = _get_kernel()
    # routed_scaling_factor rides as a 1-element input: metal_kernel template
    # args accept dtype/int/bool only, not float.
    inds, weights = kern(
        inputs=[logits.reshape(rows, E), bias,
                mx.array([routed_scaling_factor], dtype=mx.float32)],
        grid=(E, rows, 1),
        threadgroup=(E, 1, 1),
        output_shapes=[(rows, K), (rows, K)],
        output_dtypes=[mx.int32, mx.float32],
    )
    out_shape = tuple(logits.shape[:-1]) + (K,)
    return inds.reshape(out_shape), weights.reshape(out_shape)


def apply() -> bool:
    """Wrap dsv4._expert_select with the fused path (marker-gated per call)."""
    dsv4 = sys.modules.get("mlx_lm.models.deepseek_v4")
    if dsv4 is None:
        return False
    original = getattr(dsv4, "_expert_select", None)
    if original is None or getattr(original, "_ds4_router_fused", False):
        return original is not None and getattr(original, "_ds4_router_fused", False)

    def wrapped(logits, e_score_correction_bias, top_k, routed_scaling_factor,
                norm_topk_prob, scoring_func):
        if (
            _enabled()
            and top_k == K
            and logits.shape[-1] == E
            and scoring_func == "sqrtsoftplus"
            and norm_topk_prob
        ):
            return fused_expert_select(
                logits, e_score_correction_bias, routed_scaling_factor
            )
        return original(
            logits, e_score_correction_bias, top_k, routed_scaling_factor,
            norm_topk_prob, scoring_func,
        )

    wrapped._ds4_router_fused = True
    dsv4._expert_select = wrapped
    return True
