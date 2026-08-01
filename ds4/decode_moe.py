# SPDX-License-Identifier: MIT
"""Batch-1 MXFP4 routed-MoE kernels for DeepSeek V4 on Apple Silicon.

oMLX's bundled block-list kernels are excellent for prefill, but intentionally
engage only at 64+ routes. Decode has six routes per token and otherwise falls
through to generic ``mx.gather_qmm``. This patch uses a grid indexed directly
by those selected routes and the same E2M1/E8M0 arithmetic as MLX.

The gate/up projections share one command dispatch. The down projection uses a
second dispatch. The measured production policy is deliberately shape-aware:
the wide subgroup kernel handles true one-token decode, and the exact scalar
pair kernel handles DSpark's steady anchor-plus-three-drafts (L=4) verify.
Other multi-token and unsupported shapes retain oMLX's native gather path.
"""

from __future__ import annotations

from functools import lru_cache
import math
import os
from pathlib import Path

import mlx.core as mx


_PATCHED = False
_ORIGINAL_CALL = None
_MAX_ROUTES = 63
_TEST_MULTI_POLICY = None


_HEADER = r"""
#include <metal_simdgroup>
#include <metal_stdlib>
using namespace metal;

inline float ds4_fp4_e2m1(uchar bits) {
    half converted = as_type<half>(ushort((bits & 7u) << 9));
    converted *= half(16384.0);
    const float value = float(converted);
    return (bits & 8u) ? -value : value;
}

inline float ds4_e8m0(uchar bits) {
    const uint out = bits == 0u ? 0x00400000u : (uint(bits) << 23);
    return as_type<float>(out);
}

inline float ds4_qdot16(
    const device uchar* weights,
    thread const float* values,
    float scale) {
    float sum = 0.0f;
    const device ushort* packed =
        reinterpret_cast<const device ushort*>(weights);
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const ushort word = packed[i];
        sum += values[4 * i + 0] * ds4_fp4_e2m1(uchar(word));
        sum += values[4 * i + 1] * ds4_fp4_e2m1(uchar(word >> 4));
        sum += values[4 * i + 2] * ds4_fp4_e2m1(uchar(word >> 8));
        sum += values[4 * i + 3] * ds4_fp4_e2m1(uchar(word >> 12));
    }
    return scale * sum;
}

template <typename T>
inline void ds4_mxfp4_qmv_fast(
    const device uint* weight,
    const device uchar* scales,
    const device T* x,
    device T* y,
    int K,
    int N,
    uint3 tid,
    uint simd_gid,
    uint simd_lid) {
    constexpr int results_per_simdgroup = 4;
    constexpr int values_per_thread = 16;
    constexpr int block_size = values_per_thread * 32;

    const int weight_row_bytes = K / 2;
    const int scale_row_bytes = K / 32;
    const int out_row = int(tid.y) * 8 + int(simd_gid) * 4;

    const device uchar* ws =
        reinterpret_cast<const device uchar*>(weight) +
        size_t(out_row) * size_t(weight_row_bytes) +
        size_t(simd_lid) * 8u;
    const device uchar* ss =
        scales + size_t(out_row) * size_t(scale_row_bytes) +
        size_t(simd_lid / 2u);
    const device T* xv =
        x + size_t(tid.x) * size_t(K) + size_t(simd_lid) * 16u;
    device T* yv = y + size_t(tid.x) * size_t(N) + size_t(out_row);

    float accum[results_per_simdgroup] = {0.0f};
    float values[values_per_thread];
    for (int k0 = 0; k0 < K; k0 += block_size) {
        #pragma unroll
        for (int i = 0; i < values_per_thread; ++i) {
            values[i] = float(xv[i]);
        }
        #pragma unroll
        for (int row = 0; row < results_per_simdgroup; ++row) {
            const device uchar* wr =
                ws + size_t(row) * size_t(weight_row_bytes);
            const device uchar* sr =
                ss + size_t(row) * size_t(scale_row_bytes);
            accum[row] += ds4_qdot16(wr, values, ds4_e8m0(sr[0]));
        }
        ws += block_size / 2;
        ss += block_size / 32;
        xv += block_size;
    }
    #pragma unroll
    for (int row = 0; row < results_per_simdgroup; ++row) {
        accum[row] = simd_sum(accum[row]);
        if (simd_lid == 0u) {
            yv[row] = T(accum[row]);
        }
    }
}

template <typename T, int K_LANES>
inline void ds4_mxfp4_qmv_wide(
    const device uint* weight,
    const device uchar* scales,
    const device T* x,
    device T* y,
    int K,
    int N,
    uint3 tid,
    uint simd_gid,
    uint simd_lid) {
    constexpr int group_size = 32;
    constexpr int rows_per_simdgroup = 32 / K_LANES;
    constexpr int num_simdgroups = 2;
    constexpr int rows_per_threadgroup =
        rows_per_simdgroup * num_simdgroups;
    constexpr int float4_per_group = group_size / 4;

    const int k_lane = int(simd_lid) % K_LANES;
    const int simd_row = int(simd_lid) / K_LANES;
    const int out_row =
        int(tid.y) * rows_per_threadgroup +
        int(simd_gid) * rows_per_simdgroup + simd_row;
    if (out_row >= N) {
        return;
    }

    const int weight_row_bytes = K / 2;
    const int scale_row_bytes = K / group_size;
    const device uchar* wrow =
        reinterpret_cast<const device uchar*>(weight) +
        size_t(out_row) * size_t(weight_row_bytes);
    const device uchar* srow =
        scales + size_t(out_row) * size_t(scale_row_bytes);
    const device T* xrow = x + size_t(tid.x) * size_t(K);

    float result = 0.0f;
    for (int group = k_lane; group < scale_row_bytes; group += K_LANES) {
        const int k0 = group * group_size;
        const device ushort* packed =
            reinterpret_cast<const device ushort*>(wrow + k0 / 2);
        const device vec<T, 4>* xv =
            reinterpret_cast<const device vec<T, 4>*>(xrow + k0);
        float acc = 0.0f;
        #pragma unroll
        for (int i = 0; i < float4_per_group; ++i) {
            const ushort word = packed[i];
            const float4 weights(
                ds4_fp4_e2m1(uchar(word)),
                ds4_fp4_e2m1(uchar(word >> 4)),
                ds4_fp4_e2m1(uchar(word >> 8)),
                ds4_fp4_e2m1(uchar(word >> 12)));
            acc += dot(weights, float4(xv[i]));
        }
        // Scale once per 32-value microscaling group, after the dot. This is
        // the upstream qmv_wide optimization that removed 31 redundant scale
        // multiplies per group.
        result += ds4_e8m0(srow[group]) * acc;
    }

    if constexpr (K_LANES >= 32) {
        result += simd_shuffle_down(result, 16);
    }
    if constexpr (K_LANES >= 16) {
        result += simd_shuffle_down(result, 8);
    }
    if constexpr (K_LANES >= 8) {
        result += simd_shuffle_down(result, 4);
    }
    if constexpr (K_LANES >= 4) {
        result += simd_shuffle_down(result, 2);
    }
    if constexpr (K_LANES >= 2) {
        result += simd_shuffle_down(result, 1);
    }
    if (k_lane == 0) {
        y[size_t(tid.x) * size_t(N) + size_t(out_row)] = T(result);
    }
}

template <typename T, int K_LANES, int ROUTES, typename IndexPtr>
inline void ds4_mxfp4_qmv_reuse_single(
    const device uint* weight,
    const device uchar* scales,
    const device T* x,
    IndexPtr indices,
    device T* y,
    int K,
    int N,
    uint3 tid,
    uint simd_gid,
    uint simd_lid) {
    constexpr int max_vecs = 8;
    constexpr int group_size = 32;
    constexpr int rows_per_simdgroup = 32 / K_LANES;
    constexpr int rows_per_threadgroup = 2 * rows_per_simdgroup;

    const int leader = int(tid.x);
    const int expert = indices[leader];
    // Only the first occurrence launches work for an expert.  This comparison
    // is uniform across the threadgroup, so an early return cannot strand a
    // subgroup operation.
    for (int route = 0; route < leader; ++route) {
        if (indices[route] == expert) {
            return;
        }
    }

    const int k_lane = int(simd_lid) % K_LANES;
    const int simd_row = int(simd_lid) / K_LANES;
    const int out_row =
        int(tid.y) * rows_per_threadgroup +
        int(simd_gid) * rows_per_simdgroup + simd_row;
    if (out_row >= N) {
        return;
    }

    const int weight_row_bytes = K / 2;
    const int scale_row_bytes = K / group_size;
    const device uchar* wrow =
        reinterpret_cast<const device uchar*>(weight) +
        size_t(expert) * size_t(N) * size_t(weight_row_bytes) +
        size_t(out_row) * size_t(weight_row_bytes);
    const device uchar* srow =
        scales + size_t(expert) * size_t(N) * size_t(scale_row_bytes) +
        size_t(out_row) * size_t(scale_row_bytes);

    // A real DeepSeek top-k contains each expert at most once per token, so a
    // width-8 verify normally fits one batch.  The loop preserves correctness
    // for adversarial/repeated indices without an unbounded register array.
    int consumed = 0;
    while (true) {
        int selected[max_vecs];
        int count = 0;
        int seen = 0;
        for (int route = leader; route < ROUTES; ++route) {
            if (indices[route] == expert) {
                if (seen >= consumed && count < max_vecs) {
                    selected[count++] = route;
                }
                ++seen;
            }
        }
        if (count == 0) {
            break;
        }

        float result[max_vecs] = {0.0f};
        for (int group = k_lane; group < scale_row_bytes; group += K_LANES) {
            const int k0 = group * group_size;
            const device ushort* packed =
                reinterpret_cast<const device ushort*>(wrow + k0 / 2);
            float partial[max_vecs] = {0.0f};
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                const ushort word = packed[i];
                const float4 wv(
                    ds4_fp4_e2m1(uchar(word)),
                    ds4_fp4_e2m1(uchar(word >> 4)),
                    ds4_fp4_e2m1(uchar(word >> 8)),
                    ds4_fp4_e2m1(uchar(word >> 12)));
                for (int v = 0; v < count; ++v) {
                    const device vec<T, 4>* xv =
                        reinterpret_cast<const device vec<T, 4>*>(
                            x + size_t(selected[v]) * size_t(K) + size_t(k0));
                    partial[v] += dot(wv, float4(xv[i]));
                }
            }
            const float scale = ds4_e8m0(srow[group]);
            for (int v = 0; v < count; ++v) {
                result[v] += scale * partial[v];
            }
        }
        for (int v = 0; v < count; ++v) {
            float value = result[v];
            if constexpr (K_LANES >= 32) value += simd_shuffle_down(value, 16);
            if constexpr (K_LANES >= 16) value += simd_shuffle_down(value, 8);
            if constexpr (K_LANES >= 8) value += simd_shuffle_down(value, 4);
            if constexpr (K_LANES >= 4) value += simd_shuffle_down(value, 2);
            if constexpr (K_LANES >= 2) value += simd_shuffle_down(value, 1);
            if (k_lane == 0) {
                y[size_t(selected[v]) * size_t(N) + size_t(out_row)] = T(value);
            }
        }
        consumed += count;
    }
}

template <typename T, int K_LANES, int ROUTES, typename IndexPtr>
inline void ds4_mxfp4_qmv_reuse_pair(
    const device uint* up_weight,
    const device uchar* up_scales,
    const device uint* gate_weight,
    const device uchar* gate_scales,
    const device T* x,
    IndexPtr indices,
    device T* up_y,
    device T* gate_y,
    int K,
    int N,
    uint3 tid,
    uint simd_gid,
    uint simd_lid) {
    constexpr int max_vecs = 8;
    constexpr int group_size = 32;
    constexpr int rows_per_simdgroup = 32 / K_LANES;
    constexpr int rows_per_threadgroup = 2 * rows_per_simdgroup;

    const int leader = int(tid.x);
    const int expert = indices[leader];
    for (int route = 0; route < leader; ++route) {
        if (indices[route] == expert) {
            return;
        }
    }
    const int k_lane = int(simd_lid) % K_LANES;
    const int simd_row = int(simd_lid) / K_LANES;
    const int out_row =
        int(tid.y) * rows_per_threadgroup +
        int(simd_gid) * rows_per_simdgroup + simd_row;
    if (out_row >= N) {
        return;
    }

    const int weight_row_bytes = K / 2;
    const int scale_row_bytes = K / group_size;
    const size_t expert_offset =
        size_t(expert) * size_t(N) * size_t(weight_row_bytes) +
        size_t(out_row) * size_t(weight_row_bytes);
    const size_t scale_offset =
        size_t(expert) * size_t(N) * size_t(scale_row_bytes) +
        size_t(out_row) * size_t(scale_row_bytes);
    const device uchar* uw =
        reinterpret_cast<const device uchar*>(up_weight) + expert_offset;
    const device uchar* gw =
        reinterpret_cast<const device uchar*>(gate_weight) + expert_offset;
    const device uchar* us = up_scales + scale_offset;
    const device uchar* gs = gate_scales + scale_offset;

    int consumed = 0;
    while (true) {
        int selected[max_vecs];
        int count = 0;
        int seen = 0;
        for (int route = leader; route < ROUTES; ++route) {
            if (indices[route] == expert) {
                if (seen >= consumed && count < max_vecs) {
                    selected[count++] = route;
                }
                ++seen;
            }
        }
        if (count == 0) {
            break;
        }

        float up_result[max_vecs] = {0.0f};
        float gate_result[max_vecs] = {0.0f};
        for (int group = k_lane; group < scale_row_bytes; group += K_LANES) {
            const int k0 = group * group_size;
            const device ushort* up_packed =
                reinterpret_cast<const device ushort*>(uw + k0 / 2);
            const device ushort* gate_packed =
                reinterpret_cast<const device ushort*>(gw + k0 / 2);
            float up_partial[max_vecs] = {0.0f};
            float gate_partial[max_vecs] = {0.0f};
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                const ushort up_word = up_packed[i];
                const ushort gate_word = gate_packed[i];
                const float4 up_wv(
                    ds4_fp4_e2m1(uchar(up_word)),
                    ds4_fp4_e2m1(uchar(up_word >> 4)),
                    ds4_fp4_e2m1(uchar(up_word >> 8)),
                    ds4_fp4_e2m1(uchar(up_word >> 12)));
                const float4 gate_wv(
                    ds4_fp4_e2m1(uchar(gate_word)),
                    ds4_fp4_e2m1(uchar(gate_word >> 4)),
                    ds4_fp4_e2m1(uchar(gate_word >> 8)),
                    ds4_fp4_e2m1(uchar(gate_word >> 12)));
                for (int v = 0; v < count; ++v) {
                    const device vec<T, 4>* xv =
                        reinterpret_cast<const device vec<T, 4>*>(
                            x + size_t(selected[v]) * size_t(K) + size_t(k0));
                    const float4 value = float4(xv[i]);
                    up_partial[v] += dot(up_wv, value);
                    gate_partial[v] += dot(gate_wv, value);
                }
            }
            const float up_scale = ds4_e8m0(us[group]);
            const float gate_scale = ds4_e8m0(gs[group]);
            for (int v = 0; v < count; ++v) {
                up_result[v] += up_scale * up_partial[v];
                gate_result[v] += gate_scale * gate_partial[v];
            }
        }
        for (int v = 0; v < count; ++v) {
            float up_value = up_result[v];
            float gate_value = gate_result[v];
            if constexpr (K_LANES >= 32) {
                up_value += simd_shuffle_down(up_value, 16);
                gate_value += simd_shuffle_down(gate_value, 16);
            }
            if constexpr (K_LANES >= 16) {
                up_value += simd_shuffle_down(up_value, 8);
                gate_value += simd_shuffle_down(gate_value, 8);
            }
            if constexpr (K_LANES >= 8) {
                up_value += simd_shuffle_down(up_value, 4);
                gate_value += simd_shuffle_down(gate_value, 4);
            }
            if constexpr (K_LANES >= 4) {
                up_value += simd_shuffle_down(up_value, 2);
                gate_value += simd_shuffle_down(gate_value, 2);
            }
            if constexpr (K_LANES >= 2) {
                up_value += simd_shuffle_down(up_value, 1);
                gate_value += simd_shuffle_down(gate_value, 1);
            }
            if (k_lane == 0) {
                const size_t output =
                    size_t(selected[v]) * size_t(N) + size_t(out_row);
                up_y[output] = T(up_value);
                gate_y[output] = T(gate_value);
            }
        }
        consumed += count;
    }
}
"""

_SINGLE_SOURCE = r"""
    const int route = int(threadgroup_position_in_grid.x);
    const int expert = int(indices[route]);
    const device uint* route_weight =
        weight + size_t(expert) * size_t(N) * size_t(K / 8);
    const device uchar* route_scales =
        scales + size_t(expert) * size_t(N) * size_t(K / 32);
    const uint3 qmv_tid = uint3(route, threadgroup_position_in_grid.y, 0);
    ds4_mxfp4_qmv_fast<T>(
        route_weight, route_scales, x, out, K, N, qmv_tid,
        simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
"""

_PAIR_SOURCE = r"""
    const int route = int(threadgroup_position_in_grid.x);
    const int expert = int(indices[route]);
    const size_t weight_offset =
        size_t(expert) * size_t(N) * size_t(K / 8);
    const size_t scale_offset =
        size_t(expert) * size_t(N) * size_t(K / 32);
    const uint3 qmv_tid = uint3(route, threadgroup_position_in_grid.y, 0);
    ds4_mxfp4_qmv_fast<T>(
        up_weight + weight_offset, up_scales + scale_offset,
        x, up_out, K, N, qmv_tid,
        simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
    ds4_mxfp4_qmv_fast<T>(
        gate_weight + weight_offset, gate_scales + scale_offset,
        x, gate_out, K, N, qmv_tid,
        simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
"""

_WIDE_SINGLE_SOURCE = r"""
    const int route = int(threadgroup_position_in_grid.x);
    const int expert = int(indices[route]);
    const device uint* route_weight =
        weight + size_t(expert) * size_t(N) * size_t(K / 8);
    const device uchar* route_scales =
        scales + size_t(expert) * size_t(N) * size_t(K / 32);
    const uint3 qmv_tid = uint3(route, threadgroup_position_in_grid.y, 0);
    ds4_mxfp4_qmv_wide<T, K_LANES>(
        route_weight, route_scales, x, out, K, N, qmv_tid,
        simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
"""

_WIDE_PAIR_SOURCE = r"""
    const int route = int(threadgroup_position_in_grid.x);
    const int expert = int(indices[route]);
    const size_t weight_offset =
        size_t(expert) * size_t(N) * size_t(K / 8);
    const size_t scale_offset =
        size_t(expert) * size_t(N) * size_t(K / 32);
    const uint3 qmv_tid = uint3(route, threadgroup_position_in_grid.y, 0);
    ds4_mxfp4_qmv_wide<T, K_LANES>(
        up_weight + weight_offset, up_scales + scale_offset,
        x, up_out, K, N, qmv_tid,
        simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
    ds4_mxfp4_qmv_wide<T, K_LANES>(
        gate_weight + weight_offset, gate_scales + scale_offset,
        x, gate_out, K, N, qmv_tid,
        simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
"""

_REUSE_SINGLE_SOURCE = r"""
    const uint3 qmv_tid = uint3(
        threadgroup_position_in_grid.x, threadgroup_position_in_grid.y, 0);
    ds4_mxfp4_qmv_reuse_single<T, K_LANES, ROUTES>(
        weight, scales, x, indices, out, K, N, qmv_tid,
        simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
"""

_REUSE_PAIR_SOURCE = r"""
    const uint3 qmv_tid = uint3(
        threadgroup_position_in_grid.x, threadgroup_position_in_grid.y, 0);
    ds4_mxfp4_qmv_reuse_pair<T, K_LANES, ROUTES>(
        up_weight, up_scales, gate_weight, gate_scales, x, indices,
        up_out, gate_out, K, N, qmv_tid,
        simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
"""


@lru_cache(maxsize=None)
def _single_kernel(k: int, n: int):
    return mx.fast.metal_kernel(
        name=f"ds4_decode_mxfp4_single_k{k}_n{n}",
        input_names=["x", "weight", "scales", "indices"],
        output_names=["out"],
        header=_HEADER,
        source=_SINGLE_SOURCE,
        ensure_row_contiguous=True,
    )


@lru_cache(maxsize=None)
def _pair_kernel(k: int, n: int):
    return mx.fast.metal_kernel(
        name=f"ds4_decode_mxfp4_pair_k{k}_n{n}",
        input_names=[
            "x", "up_weight", "up_scales", "gate_weight", "gate_scales",
            "indices",
        ],
        output_names=["up_out", "gate_out"],
        header=_HEADER,
        source=_PAIR_SOURCE,
        ensure_row_contiguous=True,
    )


@lru_cache(maxsize=None)
def _wide_single_kernel(k: int, n: int, k_lanes: int):
    return mx.fast.metal_kernel(
        name=f"ds4_decode_mxfp4_wide_single_k{k}_n{n}_kl{k_lanes}",
        input_names=["x", "weight", "scales", "indices"],
        output_names=["out"],
        header=_HEADER,
        source=_WIDE_SINGLE_SOURCE,
        ensure_row_contiguous=True,
    )


@lru_cache(maxsize=None)
def _wide_pair_kernel(k: int, n: int, k_lanes: int):
    return mx.fast.metal_kernel(
        name=f"ds4_decode_mxfp4_wide_pair_k{k}_n{n}_kl{k_lanes}",
        input_names=[
            "x", "up_weight", "up_scales", "gate_weight", "gate_scales",
            "indices",
        ],
        output_names=["up_out", "gate_out"],
        header=_HEADER,
        source=_WIDE_PAIR_SOURCE,
        ensure_row_contiguous=True,
    )


@lru_cache(maxsize=None)
def _reuse_single_kernel(k: int, n: int, routes: int, k_lanes: int):
    return mx.fast.metal_kernel(
        name=(
            f"ds4_decode_mxfp4_reuse_single_k{k}_n{n}_r{routes}_kl{k_lanes}"
        ),
        input_names=["x", "weight", "scales", "indices"],
        output_names=["out"],
        header=_HEADER,
        source=_REUSE_SINGLE_SOURCE,
        ensure_row_contiguous=True,
    )


@lru_cache(maxsize=None)
def _reuse_pair_kernel(k: int, n: int, routes: int, k_lanes: int):
    return mx.fast.metal_kernel(
        name=f"ds4_decode_mxfp4_reuse_pair_k{k}_n{n}_r{routes}_kl{k_lanes}",
        input_names=[
            "x", "up_weight", "up_scales", "gate_weight", "gate_scales",
            "indices",
        ],
        output_names=["up_out", "gate_out"],
        header=_HEADER,
        source=_REUSE_PAIR_SOURCE,
        ensure_row_contiguous=True,
    )


def _dispatch_single(x, projection, indices):
    routes, _, k = x.shape
    n = projection.weight.shape[1]
    kernel = _single_kernel(k, n)
    (out,) = kernel(
        inputs=[x, projection.weight, projection.scales, indices],
        template=[("T", x.dtype), ("K", k), ("N", n)],
        grid=(64 * routes, math.ceil(n / 8), 1),
        threadgroup=(64, 1, 1),
        output_shapes=[(routes, 1, n)],
        output_dtypes=[x.dtype],
    )
    return out


def _dispatch_pair(x, up, gate, indices):
    routes, _, k = x.shape
    n = up.weight.shape[1]
    kernel = _pair_kernel(k, n)
    up_out, gate_out = kernel(
        inputs=[
            x, up.weight, up.scales, gate.weight, gate.scales, indices,
        ],
        template=[("T", x.dtype), ("K", k), ("N", n)],
        grid=(64 * routes, math.ceil(n / 8), 1),
        threadgroup=(64, 1, 1),
        output_shapes=[(routes, 1, n), (routes, 1, n)],
        output_dtypes=[x.dtype, x.dtype],
    )
    return up_out, gate_out


def _wide_grid(routes: int, n: int, k_lanes: int):
    if k_lanes not in (8, 16, 32):
        raise ValueError("k_lanes must be 8, 16, or 32")
    rows_per_threadgroup = 2 * (32 // k_lanes)
    return (64 * routes, math.ceil(n / rows_per_threadgroup), 1)


def _dispatch_single_wide(x, projection, indices, k_lanes: int = 16):
    """Autotuning candidate based on MLX 0.32's vectorized qmv_wide."""
    routes, _, k = x.shape
    n = projection.weight.shape[1]
    kernel = _wide_single_kernel(k, n, k_lanes)
    (out,) = kernel(
        inputs=[x, projection.weight, projection.scales, indices],
        template=[
            ("T", x.dtype), ("K", k), ("N", n), ("K_LANES", k_lanes),
        ],
        grid=_wide_grid(routes, n, k_lanes),
        threadgroup=(64, 1, 1),
        output_shapes=[(routes, 1, n)],
        output_dtypes=[x.dtype],
    )
    return out


def _dispatch_pair_wide(x, up, gate, indices, k_lanes: int = 16):
    """Gate/up pair using vectorized FP4 decode and subgroup K reduction."""
    routes, _, k = x.shape
    n = up.weight.shape[1]
    kernel = _wide_pair_kernel(k, n, k_lanes)
    up_out, gate_out = kernel(
        inputs=[
            x, up.weight, up.scales, gate.weight, gate.scales, indices,
        ],
        template=[
            ("T", x.dtype), ("K", k), ("N", n), ("K_LANES", k_lanes),
        ],
        grid=_wide_grid(routes, n, k_lanes),
        threadgroup=(64, 1, 1),
        output_shapes=[(routes, 1, n), (routes, 1, n)],
        output_dtypes=[x.dtype, x.dtype],
    )
    return up_out, gate_out


def _dispatch_single_reuse(x, projection, indices, k_lanes: int = 16):
    """MXFP4 QMV that shares a selected expert's weights across routes."""
    routes, _, k = x.shape
    n = projection.weight.shape[1]
    kernel = _reuse_single_kernel(k, n, routes, k_lanes)
    (out,) = kernel(
        inputs=[x, projection.weight, projection.scales, indices],
        template=[
            ("T", x.dtype), ("K", k), ("N", n),
            ("ROUTES", routes), ("K_LANES", k_lanes),
        ],
        grid=_wide_grid(routes, n, k_lanes),
        threadgroup=(64, 1, 1),
        output_shapes=[(routes, 1, n)],
        output_dtypes=[x.dtype],
    )
    return out


def _dispatch_pair_reuse(x, up, gate, indices, k_lanes: int = 16):
    """Fused gate/up MXFP4 QMV with cross-route expert weight reuse."""
    routes, _, k = x.shape
    n = up.weight.shape[1]
    kernel = _reuse_pair_kernel(k, n, routes, k_lanes)
    up_out, gate_out = kernel(
        inputs=[
            x, up.weight, up.scales, gate.weight, gate.scales, indices,
        ],
        template=[
            ("T", x.dtype), ("K", k), ("N", n),
            ("ROUTES", routes), ("K_LANES", k_lanes),
        ],
        grid=_wide_grid(routes, n, k_lanes),
        threadgroup=(64, 1, 1),
        output_shapes=[(routes, 1, n), (routes, 1, n)],
        output_dtypes=[x.dtype, x.dtype],
    )
    return up_out, gate_out


def _eligible_projection(projection, input_dims: int) -> bool:
    return (
        getattr(projection, "group_size", None) == 32
        and getattr(projection, "bits", None) == 4
        and getattr(projection, "mode", None) == "mxfp4"
        and projection.get("biases") is None
        and "bias" not in projection
        and projection.weight.dtype == mx.uint32
        and projection.scales.dtype == mx.uint8
        and input_dims % 512 == 0
        and projection.weight.shape[1] % 8 == 0
    )


def _eligible(layer, x, indices) -> bool:
    if getattr(layer, "training", False):
        return False
    if x.dtype not in (mx.float16, mx.bfloat16) or x.ndim < 2:
        return False
    if indices.size < 1 or indices.size > _MAX_ROUTES:
        return False
    if tuple(indices.shape[:-1]) != tuple(x.shape[:-1]):
        return False
    projections = (layer.up_proj, layer.gate_proj, layer.down_proj)
    if not all(hasattr(p, "weight") and hasattr(p, "scales") for p in projections):
        return False
    return (
        _eligible_projection(layer.up_proj, x.shape[-1])
        and _eligible_projection(layer.gate_proj, x.shape[-1])
        and layer.up_proj.weight.shape[1] == layer.gate_proj.weight.shape[1]
        and _eligible_projection(
            layer.down_proj, layer.up_proj.weight.shape[1]
        )
    )


def _route_inputs(x, indices):
    top_k = indices.shape[-1]
    routed = mx.broadcast_to(
        mx.expand_dims(x, -2), tuple(x.shape[:-1]) + (top_k, x.shape[-1])
    )
    return routed.reshape(-1, 1, x.shape[-1]), indices.reshape(-1).astype(mx.int32)


def _route_inputs_sorted(x, indices):
    """Route inputs grouped by expert, plus the permutation back to token order.

    Current oMLX ships an expert-grouped MXFP4 GEMM but only selects its block
    path at 64 routes.  A speculative verify has 12--48 routes and can still
    contain substantial expert reuse across adjacent tokens.  Keep this helper
    available for guarded A/Bs without silently changing the production path.
    """
    routed_x, flat_indices = _route_inputs(x, indices)
    order = mx.argsort(flat_indices)
    inverse = mx.argsort(order)
    return routed_x[order], flat_indices[order], inverse


def _dispatch_expert_grouped(x, projection, sorted_indices, variant: int = 0):
    """Call oMLX's compiled small-M expert-grouped MXFP4 projection."""
    from omlx.custom_kernels.glm_moe_dsa import fast as glm_fast

    if not glm_fast.has_symbol("deepseek_mxfp4_gather_qmm_expert"):
        raise RuntimeError("expert-grouped MXFP4 kernel is unavailable")
    return glm_fast.deepseek_mxfp4_gather_qmm_expert(
        x,
        projection.weight,
        projection.scales,
        sorted_indices,
        variant,
    )


def _expert_grouped_call(layer, x, indices, variant: int = 0):
    """Full SwitchGLU candidate that reuses weights for repeated experts."""
    sorted_x, sorted_indices, inverse = _route_inputs_sorted(x, indices)
    up = _dispatch_expert_grouped(
        sorted_x, layer.up_proj, sorted_indices, variant=variant
    )
    gate = _dispatch_expert_grouped(
        sorted_x, layer.gate_proj, sorted_indices, variant=variant
    )
    hidden = layer.activation(up, gate)
    out = _dispatch_expert_grouped(
        hidden, layer.down_proj, sorted_indices, variant=variant
    )
    return out[inverse].reshape(
        tuple(indices.shape) + (layer.down_proj.weight.shape[1],)
    )


def _reuse_call(layer, x, indices, k_lanes: int = 16):
    """Full SwitchGLU candidate with compact unsorted expert reuse."""
    routed_x, flat_indices = _route_inputs(x, indices)
    up, gate = _dispatch_pair_reuse(
        routed_x,
        layer.up_proj,
        layer.gate_proj,
        flat_indices,
        k_lanes=k_lanes,
    )
    hidden = layer.activation(up, gate)
    out = _dispatch_single_reuse(
        hidden,
        layer.down_proj,
        flat_indices,
        k_lanes=k_lanes,
    )
    return out.reshape(
        tuple(indices.shape) + (layer.down_proj.weight.shape[1],)
    )


def _wide_call(layer, x, indices, scores=None, k_lanes: int = 32):
    """Full one-token SwitchGLU using the vectorized subgroup QMV kernel."""
    del scores
    routed_x, flat_indices = _route_inputs(x, indices)
    up, gate = _dispatch_pair_wide(
        routed_x,
        layer.up_proj,
        layer.gate_proj,
        flat_indices,
        k_lanes=k_lanes,
    )
    hidden = layer.activation(up, gate)
    out = _dispatch_single_wide(
        hidden,
        layer.down_proj,
        flat_indices,
        k_lanes=k_lanes,
    )
    return out.reshape(
        tuple(indices.shape) + (layer.down_proj.weight.shape[1],)
    )


def _fast_call(layer, x, indices, scores=None):
    """Legacy scalar direct-route candidate retained for guarded A/Bs."""
    del scores
    routed_x, flat_indices = _route_inputs(x, indices)
    up, gate = _dispatch_pair(
        routed_x, layer.up_proj, layer.gate_proj, flat_indices
    )
    hidden = layer.activation(up, gate)
    out = _dispatch_single(hidden, layer.down_proj, flat_indices)
    return out.reshape(tuple(indices.shape) + (layer.down_proj.weight.shape[1],))


def set_test_multi_policy(policy=None) -> None:
    """Select a full-model multi-token A/B candidate without production I/O.

    ``None`` is the only production value.  The guarded one-load benchmark can
    switch this process-global between forwards so every policy sees the same
    real model, cache position, router outputs, and thermal state.
    """
    allowed = {
        None,
        "generic",
        "scalar",
        "wide8",
        "wide16",
        "wide32",
        "expert0",
    }
    if policy not in allowed:
        raise ValueError(f"unsupported test multi-token MoE policy: {policy}")
    global _TEST_MULTI_POLICY
    _TEST_MULTI_POLICY = policy


def apply() -> bool:
    """Patch oMLX's DeepSeek V4 SwitchGLU after its model patch is installed."""
    global _PATCHED, _ORIGINAL_CALL
    if (Path.home() / ".omlx" / "ds4_decode_moe_off").exists():
        return False
    if os.environ.get("DS4_DECODE_MOE", "1") == "0":
        return False

    from omlx.patches.deepseek_v4 import switch_layers

    cls = switch_layers.SwitchGLU
    if getattr(cls.__call__, "_ds4_decode_moe", False):
        _PATCHED = True
        return True

    original = cls.__call__
    _ORIGINAL_CALL = original

    def patched(self, x, indices, scores=None):
        if not _eligible(self, x, indices):
            return original(self, x, indices, scores=scores)
        # Real-model interleaved measurements: wide-KL32 wins one-token decode;
        # at L=4 the exact scalar pair path improves the complete 43-layer
        # target forward by 3.3%. Wider subgroup variants changed real-model
        # argmax IDs, and expert grouping lost 15%, so neither is production.
        # Token count is shape-known and avoids synchronizing router indices.
        token_count = x.size // x.shape[-1]
        if token_count == 1:
            return _wide_call(self, x, indices, scores=scores, k_lanes=32)
        policy = _TEST_MULTI_POLICY
        if policy == "generic":
            return original(self, x, indices, scores=scores)
        if policy is None:
            if token_count == 4:
                return _fast_call(self, x, indices, scores=scores)
            return original(self, x, indices, scores=scores)
        if policy == "scalar":
            return _fast_call(self, x, indices, scores=scores)
        if policy.startswith("wide"):
            return _wide_call(
                self,
                x,
                indices,
                scores=scores,
                k_lanes=int(policy[4:]),
            )
        if policy == "expert0":
            return _expert_grouped_call(self, x, indices, variant=0)
        raise RuntimeError(f"unreachable test policy: {policy}")

    patched._ds4_decode_moe = True
    patched._ds4_original = original
    cls.__call__ = patched
    _PATCHED = True
    return True


def original_call(layer, x, indices, scores=None):
    """Test helper: call the exact pre-patch SwitchGLU implementation."""
    if _ORIGINAL_CALL is None:
        raise RuntimeError("decode_moe.apply() has not installed the patch")
    return _ORIGINAL_CALL(layer, x, indices, scores=scores)
