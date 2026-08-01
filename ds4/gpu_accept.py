# SPDX-License-Identifier: MIT
"""GPU-side speculative acceptance: the single-readback cycle.

Today the cycle's structural host sync is reading ALL k+1 sampled target tokens
back to Python (`tolist`) to compute the accepted-prefix length and pick the
correction token. That round-trip is the Metal-side shape of what CUDA graphs
remove on the DGX recipes.

This kernel does the comparison on the GPU:

    inputs : target_ids (k+1) int32   -- argmax of the verify logits
             draft_ids  (k)   int32   -- the drafter's proposals
    outputs: n_accept   (1)   int32   -- longest prefix with target==draft
             emit_ids   (k+1) int32   -- the committed sequence: accepted
                                         drafts then the correction token,
                                         padded with -1

The host then reads back ONE int32 (`n_accept`) for the cache-rollback branch;
`emit_ids` stays on-GPU for the queue/window path until the existing (already
batched) materialisation points.

Correctness note: this must reproduce the host loop EXACTLY:
    n = 0; while n < k and samples[n] == drafts[n]: n += 1
A single thread does it -- k <= 8, so parallelism buys nothing and sequential
scan is trivially exact.
"""

from __future__ import annotations

import mlx.core as mx

_SOURCE = """
    // One thread does the whole job: k <= 8.
    uint tid = thread_position_in_grid.x;
    if (tid != 0) return;

    int k = draft_n[0];
    int n = 0;
    while (n < k && target_ids[n] == draft_ids[n]) { n += 1; }
    n_accept[0] = n;

    // Committed tokens: accepted drafts then the target's correction.
    for (int i = 0; i < n; ++i) emit_ids[i] = draft_ids[i];
    emit_ids[n] = target_ids[n];
    for (int i = n + 1; i <= k; ++i) emit_ids[i] = -1;
"""

_kernel = None


def _get_kernel():
    global _kernel
    if _kernel is None:
        _kernel = mx.fast.metal_kernel(
            name="ds4_gpu_accept",
            input_names=["target_ids", "draft_ids", "draft_n"],
            output_names=["n_accept", "emit_ids"],
            source=_SOURCE,
            ensure_row_contiguous=True,
        )
    return _kernel


def gpu_accept(target_ids: mx.array, draft_ids: mx.array):
    """target_ids (k+1,) int32, draft_ids (k,) int32 -> (n_accept, emit_ids).

    Both outputs are lazy; the caller reads back n_accept only.
    grid=(1,1,1): one thread total (grid is measured in THREADS -- the router
    kernel's 0/244 lesson, encoded here on purpose).
    """
    k = draft_ids.shape[0]
    kern = _get_kernel()
    n_accept, emit_ids = kern(
        inputs=[
            target_ids.astype(mx.int32),
            draft_ids.astype(mx.int32),
            mx.array([k], dtype=mx.int32),
        ],
        grid=(1, 1, 1),
        threadgroup=(1, 1, 1),
        output_shapes=[(1,), (k + 1,)],
        output_dtypes=[mx.int32, mx.int32],
    )
    return n_accept, emit_ids
