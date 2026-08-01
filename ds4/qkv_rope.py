# SPDX-License-Identifier: MIT
"""Fused DeepSeek-V4 small-L attention layout kernels.

V4 normalizes 64 query heads plus one shared KV row at width 512, then runs a
traditional RoPE whose first 224 frequency pairs are infinity (identity) and
whose final 32 pairs rotate the 64-dimensional tail.  oMLX currently launches
four separate Metal operations and materializes a transposed query view.

This kernel follows the accepted MLXFast QK-norm/RoPE fusion, adapted to V4's
512-wide rows and exact stock reduction geometry: one 128-thread group (four
SIMD groups, four values per thread) per token/head row.  It writes the query
output directly as [B, H, L, D] and KV as [B, 1, L, D].

The same source rewrite can optionally fuse inverse RoPE with the attention
output's H->group permutation before ``wo_a``.  That path keeps the stock
per-pair arithmetic and writes directly as [B, G, L, H/G*D].  It is separately
gated so full-model A/B can reject it without disturbing the proven Q/KV path.
"""

from __future__ import annotations

from functools import lru_cache
import functools
import inspect
import os
from pathlib import Path
import sys
import textwrap

import mlx.core as mx


_PATCHED = False
_OUTPUT_LAYOUT_PRODUCTION = False
_TEST_OUTPUT_LAYOUT: bool | None = None


_HEADER = r"""
#include <metal_common>
#include <metal_math>
#include <metal_simdgroup>
using namespace metal;
"""


_SOURCE = r"""
    constexpr uint head_dim = 512;
    constexpr uint query_heads = 64;
    constexpr uint reads_per_thread = 4;
    constexpr uint simd_size = 32;

    const uint row = threadgroup_position_in_grid.x;
    const uint token = threadgroup_position_in_grid.y;
    const uint length = threadgroups_per_grid.y;
    const uint lid = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup;
    const uint simd = simdgroup_index_in_threadgroup;
    const uint base = lid * reads_per_thread;

    const device T* input;
    device T* output;
    const bool is_query = row < query_heads;
    if (is_query) {
        input = raw_queries + (token * query_heads + row) * head_dim;
        output = queries + (row * length + token) * head_dim;
    } else {
        input = raw_kv + token * head_dim;
        output = keys + token * head_dim;
    }

    threadgroup float local_inverse_rms[1];
    threadgroup float local_sums[simd_size];
    thread float values[reads_per_thread];
    float sum = 0.0f;
    #pragma clang loop unroll(full)
    for (uint i = 0; i < reads_per_thread; ++i) {
        const float value = float(input[base + i]);
        values[i] = value;
        sum += value * value;
    }
    sum = simd_sum(sum);

    // Reproduce rms_single_row's four-SIMD reduction exactly.  Keeping the
    // same per-thread four-value accumulation and shared reduction order is
    // required at width 512; a one-SIMD 16-value shortcut changes rounding.
    if (simd == 0) {
        local_sums[lane] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) {
        local_sums[simd] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd == 0) {
        sum = simd_sum(local_sums[lane]);
        if (lane == 0) {
            local_inverse_rms[0] =
                metal::precise::rsqrt(sum / 512.0f + 1.0e-6f);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    thread T normalized[reads_per_thread];
    #pragma clang loop unroll(full)
    for (uint i = 0; i < reads_per_thread; ++i) {
        const T scaled = T(values[i] * local_inverse_rms[0]);
        normalized[i] = is_query ? scaled : kv_weight[base + i] * scaled;
    }

    // traditional=True pairs adjacent elements.  freqs contains 224 +inf
    // entries followed by V4's 32 real frequencies, so this same loop copies
    // the non-rotary prefix (theta=0) and rotates only the final 64 values.
    const float position = float(int(offsets[0]) + int(token));
    #pragma clang loop unroll(full)
    for (uint pair_local = 0; pair_local < 2; ++pair_local) {
        const uint element = 2 * pair_local;
        const uint pair = base / 2 + pair_local;
        const float theta = position * (1.0f / frequencies[pair]);
        const float cosine = metal::fast::cos(theta);
        const float sine = metal::fast::sin(theta);
        const float first = float(normalized[element]);
        const float second = float(normalized[element + 1]);
        output[base + element] = T(first * cosine - second * sine);
        output[base + element + 1] = T(first * sine + second * cosine);
    }
"""


_OUTPUT_LAYOUT_SOURCE = r"""
    constexpr uint head_dim = 512;
    constexpr uint query_heads = 64;
    constexpr uint output_groups = 8;
    constexpr uint heads_per_group = query_heads / output_groups;
    constexpr uint reads_per_thread = 4;

    const uint head = threadgroup_position_in_grid.x;
    const uint token = threadgroup_position_in_grid.y;
    const uint length = threadgroups_per_grid.y;
    const uint lid = thread_position_in_threadgroup.x;
    const uint base = lid * reads_per_thread;
    const uint group = head / heads_per_group;
    const uint head_in_group = head % heads_per_group;

    const device T* input = values + (head * length + token) * head_dim;
    device T* output = grouped_values
        + ((group * length + token) * heads_per_group + head_in_group) * head_dim;

    const float position = float(int(offsets[0]) + int(token));
    #pragma clang loop unroll(full)
    for (uint pair_local = 0; pair_local < 2; ++pair_local) {
        const uint element = 2 * pair_local;
        const uint pair = base / 2 + pair_local;
        const float theta = position * (1.0f / frequencies[pair]);
        const float cosine = metal::fast::cos(theta);
        const float sine = metal::fast::sin(theta);
        const float first = float(input[base + element]);
        const float second = float(input[base + element + 1]);
        output[base + element] = T(first * cosine - second * sine);
        output[base + element + 1] = T(first * sine + second * cosine);
    }
"""


@lru_cache(maxsize=1)
def _kernel():
    return mx.fast.metal_kernel(
        name="ds4_qkv_norm_rope_bf16_512_v1",
        input_names=["raw_queries", "raw_kv", "kv_weight", "frequencies", "offsets"],
        output_names=["queries", "keys"],
        header=_HEADER,
        source=_SOURCE,
        ensure_row_contiguous=True,
    )


@lru_cache(maxsize=1)
def _output_layout_kernel():
    return mx.fast.metal_kernel(
        name="ds4_attn_inverse_rope_layout_bf16_512_v1",
        input_names=["values", "frequencies", "offsets"],
        output_names=["grouped_values"],
        header=_HEADER,
        source=_OUTPUT_LAYOUT_SOURCE,
        ensure_row_contiguous=True,
    )


def fused_qkv_norm_rope(raw_queries, raw_kv, kv_weight, frequencies, offset):
    """Return Q [1,64,L,512] and KV [1,1,L,512] after norm + RoPE."""
    if raw_queries.ndim != 4 or tuple(raw_queries.shape[:1]) != (1,):
        raise ValueError("raw_queries must have shape [1, L, 64, 512]")
    length = int(raw_queries.shape[1])
    if tuple(raw_queries.shape[2:]) != (64, 512):
        raise ValueError("raw_queries must have shape [1, L, 64, 512]")
    if tuple(raw_kv.shape) != (1, length, 512):
        raise ValueError("raw_kv must have shape [1, L, 512]")
    if tuple(kv_weight.shape) != (512,) or tuple(frequencies.shape) != (256,):
        raise ValueError("expected kv_weight[512] and frequencies[256]")
    if raw_queries.dtype != mx.bfloat16 or raw_kv.dtype != mx.bfloat16:
        raise ValueError("the production V4 fusion requires BF16 inputs")
    if kv_weight.dtype != mx.bfloat16 or frequencies.dtype != mx.float32:
        raise ValueError("expected BF16 KV weight and FP32 frequencies")

    offsets = (
        offset.astype(mx.int32).reshape(-1)
        if isinstance(offset, mx.array)
        else mx.array([int(offset)], dtype=mx.int32)
    )
    kernel = _kernel()
    queries, keys = kernel(
        inputs=[raw_queries, raw_kv, kv_weight, frequencies, offsets],
        template=[("T", raw_queries.dtype)],
        grid=(65 * 128, length, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(1, 64, length, 512), (1, 1, length, 512)],
        output_dtypes=[raw_queries.dtype, raw_queries.dtype],
    )
    return queries, keys


def fused_attn_inverse_rope_layout(values, frequencies, offset, groups=8):
    """Apply inverse RoPE and write [1,H,L,D] as [1,G,L,H/G*D]."""
    if values.ndim != 4 or tuple(values.shape[:2]) != (1, 64):
        raise ValueError("values must have shape [1, 64, L, 512]")
    length = int(values.shape[2])
    if values.shape[3] != 512:
        raise ValueError("values must have shape [1, 64, L, 512]")
    if groups != 8:
        raise ValueError("the production output fusion requires 8 groups")
    if tuple(frequencies.shape) != (256,):
        raise ValueError("expected inverse frequencies[256]")
    if values.dtype != mx.bfloat16 or frequencies.dtype != mx.float32:
        raise ValueError("expected BF16 values and FP32 frequencies")

    offsets = (
        offset.astype(mx.int32).reshape(-1)
        if isinstance(offset, mx.array)
        else mx.array([int(offset)], dtype=mx.int32)
    )
    kernel = _output_layout_kernel()
    return kernel(
        inputs=[values, frequencies, offsets],
        template=[("T", values.dtype)],
        grid=(64 * 128, length, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(1, groups, length, (64 // groups) * 512)],
        output_dtypes=[values.dtype],
    )[0]


def set_test_output_layout(value: bool | None) -> None:
    """Override the output-layout policy for guarded one-process A/B tests."""
    global _TEST_OUTPUT_LAYOUT
    _TEST_OUTPUT_LAYOUT = value


def _use_output_layout() -> bool:
    return (
        _OUTPUT_LAYOUT_PRODUCTION
        if _TEST_OUTPUT_LAYOUT is None
        else _TEST_OUTPUT_LAYOUT
    )


def _attn_inverse_rope_layout(values, module, offset, batch, length):
    if (
        _use_output_layout()
        and batch == 1
        and length <= 8
        and module.n_heads == 64
        and module.head_dim == 512
        and module.o_groups == 8
        and values.dtype == mx.bfloat16
    ):
        return fused_attn_inverse_rope_layout(
            values,
            module.rope._get_freqs(module.head_dim, True),
            offset,
            module.o_groups,
        )

    values = module.rope(values, offset, inverse=True)
    values = values.reshape(
        batch, module.o_groups, -1, length, module.head_dim
    )
    return values.transpose(0, 1, 3, 2, 4).flatten(-2)


_COMMON_STOCK = """\
q = self.wq_b(self.q_norm(self.wq_a(x)))
q = q.reshape(B, L, self.n_heads, self.head_dim)
q = mx.fast.rms_norm(q, None, self.config.rms_norm_eps)
q = q.transpose(0, 2, 1, 3)
q = self.rope(q, offset)

kv = self.kv_norm(self.wkv(x)).reshape(B, 1, L, self.head_dim)
kv = self.rope(kv, offset)
"""


_SPARSE_STOCK = """\
q_residual = self.q_norm(self.wq_a(x))
q = self.wq_b(q_residual).reshape(B, L, self.n_heads, self.head_dim)
q = mx.fast.rms_norm(q, None, self.config.rms_norm_eps)
q = q.transpose(0, 2, 1, 3)
q = self.rope(q, offset)

kv = self.kv_norm(self.wkv(x)).reshape(B, 1, L, self.head_dim)
kv = self.rope(kv, offset)
"""


_COMMON_FUSED = """\
q = self.wq_b(self.q_norm(self.wq_a(x)))
q = q.reshape(B, L, self.n_heads, self.head_dim)
raw_kv = self.wkv(x)
if (
    B == 1
    and L <= 8
    and self.n_heads == 64
    and self.head_dim == 512
    and q.dtype == mx.bfloat16
    and raw_kv.dtype == mx.bfloat16
    and self.kv_norm.weight.dtype == mx.bfloat16
):
    q, kv = _ds4_fused_qkv_norm_rope(
        q,
        raw_kv,
        self.kv_norm.weight,
        self.rope._get_freqs(self.head_dim, False),
        offset,
    )
else:
    q = mx.fast.rms_norm(q, None, self.config.rms_norm_eps)
    q = q.transpose(0, 2, 1, 3)
    q = self.rope(q, offset)
    kv = self.kv_norm(raw_kv).reshape(B, 1, L, self.head_dim)
    kv = self.rope(kv, offset)
"""


_SPARSE_FUSED = """\
q_residual = self.q_norm(self.wq_a(x))
q = self.wq_b(q_residual).reshape(B, L, self.n_heads, self.head_dim)
raw_kv = self.wkv(x)
if (
    B == 1
    and L <= 8
    and self.n_heads == 64
    and self.head_dim == 512
    and q.dtype == mx.bfloat16
    and raw_kv.dtype == mx.bfloat16
    and self.kv_norm.weight.dtype == mx.bfloat16
):
    q, kv = _ds4_fused_qkv_norm_rope(
        q,
        raw_kv,
        self.kv_norm.weight,
        self.rope._get_freqs(self.head_dim, False),
        offset,
    )
else:
    q = mx.fast.rms_norm(q, None, self.config.rms_norm_eps)
    q = q.transpose(0, 2, 1, 3)
    q = self.rope(q, offset)
    kv = self.kv_norm(raw_kv).reshape(B, 1, L, self.head_dim)
    kv = self.rope(kv, offset)
"""


def enabled() -> bool:
    value = os.environ.get("DS4_QKV_ROPE")
    if value is not None:
        return value == "1"
    return (Path.home() / ".omlx" / "ds4_qkv_rope").exists()


def output_layout_enabled() -> bool:
    value = os.environ.get("DS4_ATTN_OUTPUT_LAYOUT")
    if value is not None:
        return value == "1"
    return (Path.home() / ".omlx" / "ds4_attn_output_layout").exists()


def _rewrite_call(cls, first_line: str, replacement: str) -> None:
    original = cls.__call__
    if getattr(original, "_ds4_qkv_rope", False):
        return
    source = textwrap.dedent(inspect.getsource(original))
    start_marker = "    " + first_line
    end_marker = "    kv = self.rope(kv, offset)"
    replacement = textwrap.indent(replacement, "    ")
    if source.count(start_marker) != 1 or source.count(end_marker) != 1:
        raise RuntimeError(
            f"{cls.__name__} attention source drifted; refusing QKV rewrite"
        )
    start = source.index(start_marker)
    end = source.index(end_marker, start) + len(end_marker)
    source = source[:start] + replacement.rstrip("\n") + source[end:]
    output_stock = textwrap.indent(
        """\
out = self.rope(out, offset, inverse=True)

out = out.reshape(B, self.o_groups, -1, L, self.head_dim)
out = out.transpose(0, 1, 3, 2, 4).flatten(-2)
""",
        "    ",
    )
    output_fused = textwrap.indent(
        "out = _ds4_attn_inverse_rope_layout(out, self, offset, B, L)\n",
        "    ",
    )
    if source.count(output_stock) != 1:
        raise RuntimeError(
            f"{cls.__name__} output source drifted; refusing layout rewrite"
        )
    source = source.replace(output_stock, output_fused, 1)
    namespace = original.__globals__
    namespace["_ds4_fused_qkv_norm_rope"] = fused_qkv_norm_rope
    namespace["_ds4_attn_inverse_rope_layout"] = _attn_inverse_rope_layout
    compiled: dict = {}
    exec(compile(source, f"<ds4-{cls.__name__}-qkv>", "exec"), namespace, compiled)
    rewritten = compiled["__call__"]
    functools.update_wrapper(rewritten, original)
    rewritten._ds4_qkv_rope = True
    rewritten._ds4_original = original
    cls.__call__ = rewritten


def apply(force: bool = False) -> bool:
    """Install the decode/verify-only attention rewrite, fail-closed on drift."""
    global _PATCHED, _OUTPUT_LAYOUT_PRODUCTION
    if not force and not enabled():
        return False
    _OUTPUT_LAYOUT_PRODUCTION = output_layout_enabled()
    dsv4 = sys.modules.get("mlx_lm.models.deepseek_v4")
    if dsv4 is None:
        return False
    classes = (
        (
            getattr(dsv4, "LocalAttention", None),
            "q = self.wq_b(self.q_norm(self.wq_a(x)))",
            _COMMON_FUSED,
        ),
        (
            getattr(dsv4, "CompressedAttention", None),
            "q = self.wq_b(self.q_norm(self.wq_a(x)))",
            _COMMON_FUSED,
        ),
        (
            getattr(dsv4, "SparseCompressedAttention", None),
            "q_residual = self.q_norm(self.wq_a(x))",
            _SPARSE_FUSED,
        ),
    )
    if any(cls is None for cls, _needle, _replacement in classes):
        return False
    if all(getattr(cls, "_ds4_qkv_rope_enabled", False) for cls, _, _ in classes):
        return True
    for cls, needle, replacement in classes:
        _rewrite_call(cls, needle, replacement)
        cls._ds4_qkv_rope_enabled = True
    _PATCHED = True
    return True
