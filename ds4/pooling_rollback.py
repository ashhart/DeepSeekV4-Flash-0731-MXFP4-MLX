# SPDX-License-Identifier: MIT
"""Prefix-preserving rollback for DeepSeek-V4 pooling caches.

oMLX's one-update undo can only replay a confirmed prefix when that prefix does
not complete a pooling window.  Wider speculative blocks therefore clamp good
drafts or undo and run the target model a second time at ratio-4 boundaries.

The verifier has already computed every pooled output needed by an accepted
prefix.  Pooling is causal by window, so rollback can keep the leading pooled
outputs, rebuild the small raw remainder, and restore the overlap carry without
re-running the compressor.  This patch is intentionally limited to the
single-request cache layout used by DS4 speculation; ordinary batching retains
oMLX's implementation.
"""

from __future__ import annotations

import sys
from typing import Any

_PATCHED_CLASSES: set[type] = set()


def _copy_prefix(value, length: int):
    """Detach the live prefix from an MLX buffer that may be mutated in place."""
    if value is None or length <= 0:
        return None
    return value[:, :length] + 0


def _join_prefix(mx, old, new, keep: int):
    pieces = []
    if old is not None and old.shape[1]:
        pieces.append(old)
    if keep:
        pieces.append(new[:, :keep])
    if not pieces:
        return new[:, :0]
    if len(pieces) == 1:
        return pieces[0]
    return mx.concatenate(pieces, axis=1)


def _new_buffer(mx, template, live):
    out = mx.zeros_like(template)
    if live.shape[1]:
        out[:, : live.shape[1]] = live
    return out


def _last_window(mx, combined, completed: int, ratio: int):
    start = (completed - 1) * ratio
    window = combined[:, start : start + ratio]
    return mx.reshape(window, (window.shape[0], 1, ratio, window.shape[-1]))


def _wrap_pooling(cls: type) -> None:
    if cls in _PATCHED_CLASSES or getattr(cls, "_ds4_prefix_rollback", False):
        return

    import mlx.core as mx

    original_accumulate = cls.accumulate_windows
    original_can_undo = cls._can_undo
    original_trim = cls.trim

    def accumulate_windows(self, kv, gate, offset):
        length = int(kv.shape[1])
        if length <= 8:
            remainder = int(self.remainder)
            self._ds4_prefix_undo = {
                "length": length,
                "kv": kv,
                "gate": gate,
                "remainder": remainder,
                # PoolingCache's multi-token path writes a new remainder into
                # the existing buffer, so a view is not a snapshot.  +0 keeps
                # this tiny prefix on its own lazy graph node without a sync.
                "buf_kv": _copy_prefix(self.buf_kv, remainder),
                "buf_gate": _copy_prefix(self.buf_gate, remainder),
                "pooled": self.pooled,
                "pool_length": 0 if self.pooled is None else self.pooled.shape[1],
                "prev_kv": self.prev_win_kv,
                "prev_gate": self.prev_win_gate,
            }
        else:
            self._ds4_prefix_undo = None
        return original_accumulate(self, kv, gate, offset)

    def can_undo(self, n):
        undo = getattr(self, "_ds4_prefix_undo", None)
        if undo is not None and 0 <= int(n) <= undo["length"]:
            return True
        return original_can_undo(self, n)

    def trim(self, n):
        n = int(n)
        if n <= int(self.remainder):
            result = original_trim(self, n)
            self._ds4_prefix_undo = None
            return result

        undo = getattr(self, "_ds4_prefix_undo", None)
        if undo is None or n < 0 or n > undo["length"]:
            return original_trim(self, n)

        keep = undo["length"] - n
        ratio = int(self.ratio)
        combined_kv = _join_prefix(mx, undo["buf_kv"], undo["kv"], keep)
        combined_gate = _join_prefix(mx, undo["buf_gate"], undo["gate"], keep)
        completed = int(combined_kv.shape[1]) // ratio
        consumed = completed * ratio
        live_kv = combined_kv[:, consumed:]
        live_gate = combined_gate[:, consumed:]

        full_pooled = self.pooled
        new_pool_length = undo["pool_length"] + completed
        if completed == 0:
            self.pooled = undo["pooled"]
        else:
            self.pooled = full_pooled[:, :new_pool_length]

        self.buf_kv = _new_buffer(mx, self.buf_kv, live_kv)
        self.buf_gate = _new_buffer(mx, self.buf_gate, live_gate)
        self.remainder = int(live_kv.shape[1])

        if ratio == 4 and completed:
            self.prev_win_kv = _last_window(mx, combined_kv, completed, ratio)
            self.prev_win_gate = _last_window(mx, combined_gate, completed, ratio)
        else:
            self.prev_win_kv = undo["prev_kv"]
            self.prev_win_gate = undo["prev_gate"]

        self._undo = None
        self._ds4_prefix_undo = None
        return n

    cls.accumulate_windows = accumulate_windows
    cls._can_undo = can_undo
    cls.trim = trim
    cls._ds4_prefix_rollback = True
    _PATCHED_CLASSES.add(cls)


def _wrap_batch_pooling(cls: type) -> None:
    if cls in _PATCHED_CLASSES or getattr(cls, "_ds4_prefix_rollback", False):
        return

    import mlx.core as mx

    original_accumulate = cls.accumulate_windows
    original_can_undo = cls._can_undo
    original_trim = cls.trim

    def accumulate_windows(self, kv, gate, offset):
        length = int(kv.shape[1])
        # DS4 engages only for a one-sequence GenerationBatch.  Avoid changing
        # the semantics or snapshot cost of unrelated multi-request batching.
        if kv.shape[0] == 1 and length <= 8:
            remainder = int(self.remainder[0])
            self._ds4_prefix_undo = {
                "length": length,
                "kv": kv,
                "gate": gate,
                "remainder": remainder,
                # When a window completes BatchPoolingCache rebinds buf_* to
                # fresh arrays, so these references remain the pre-update data.
                # When none completes, trim uses the cheap remainder fast path.
                "buf_kv": self.buf_kv,
                "buf_gate": self.buf_gate,
                "pool_length": int(self._pool_lengths[0]),
                "processed": int(self._processed[0]),
                "prev_kv": self.prev_win_kv,
                "prev_gate": self.prev_win_gate,
                "prev_valid": bool(self._prev_valid[0]),
            }
        else:
            self._ds4_prefix_undo = None
        return original_accumulate(self, kv, gate, offset)

    def can_undo(self, n):
        undo = getattr(self, "_ds4_prefix_undo", None)
        if undo is not None and 0 <= int(n) <= undo["length"]:
            return True
        return original_can_undo(self, n)

    def trim(self, n):
        n = int(n)
        if len(self.remainder) != 1 or n <= min(self.remainder):
            result = original_trim(self, n)
            self._ds4_prefix_undo = None
            return result

        undo = getattr(self, "_ds4_prefix_undo", None)
        if undo is None or n < 0 or n > undo["length"]:
            return original_trim(self, n)

        keep = undo["length"] - n
        ratio = int(self.ratio)
        old_kv = (
            None
            if undo["buf_kv"] is None or undo["remainder"] == 0
            else undo["buf_kv"][:, : undo["remainder"]]
        )
        old_gate = (
            None
            if undo["buf_gate"] is None or undo["remainder"] == 0
            else undo["buf_gate"][:, : undo["remainder"]]
        )
        combined_kv = _join_prefix(mx, old_kv, undo["kv"], keep)
        combined_gate = _join_prefix(mx, old_gate, undo["gate"], keep)
        completed = int(combined_kv.shape[1]) // ratio
        consumed = completed * ratio
        live_kv = combined_kv[:, consumed:]
        live_gate = combined_gate[:, consumed:]
        new_pool_length = undo["pool_length"] + completed

        if self.pooled is not None:
            self.pooled = self.pooled[:, :new_pool_length]
        self._pool_lengths = [new_pool_length]
        self._processed = [undo["processed"] + keep]
        self.remainder = [int(live_kv.shape[1])]
        self.buf_kv = _new_buffer(mx, self.buf_kv, live_kv)
        self.buf_gate = _new_buffer(mx, self.buf_gate, live_gate)
        self._last_usable = [completed * ratio]

        if ratio == 4 and completed:
            self.prev_win_kv = _last_window(mx, combined_kv, completed, ratio)
            self.prev_win_gate = _last_window(mx, combined_gate, completed, ratio)
            self._prev_valid = [True]
        else:
            self.prev_win_kv = undo["prev_kv"]
            self.prev_win_gate = undo["prev_gate"]
            self._prev_valid = [undo["prev_valid"]]

        self._undo = None
        self._ds4_prefix_undo = None
        return n

    cls.accumulate_windows = accumulate_windows
    cls._can_undo = can_undo
    cls.trim = trim
    cls._ds4_prefix_rollback = True
    _PATCHED_CLASSES.add(cls)


def apply() -> bool:
    """Patch the live oMLX-injected cache classes, idempotently."""
    cache_module: Any = sys.modules.get("mlx_lm.models.cache")
    if cache_module is None:
        return False
    pooling = getattr(cache_module, "PoolingCache", None)
    batch_pooling = getattr(cache_module, "BatchPoolingCache", None)
    if pooling is None or batch_pooling is None:
        return False
    _wrap_pooling(pooling)
    _wrap_batch_pooling(batch_pooling)
    return True

