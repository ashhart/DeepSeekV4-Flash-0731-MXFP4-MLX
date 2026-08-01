# SPDX-License-Identifier: MIT
"""Opt-in asynchronous DeepSeek-V4 cache-graph materialization.

oMLX evaluates every functional cache leaf after the transformer stack to
detach the just-built update graph from prior decode steps. A blocking eval is
safe but creates a mid-forward CPU barrier. ``mx.async_eval`` requests the same
materialization while allowing the final norm/head graph to be enqueued before
the host waits for token selection.

The patch is off unless ``DS4_ASYNC_CACHE=1`` or
``~/.omlx/ds4_async_cache`` exists. It is installed after oMLX registers its
DeepSeek-V4 module and is idempotent across model reloads.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Any, Optional

import mlx.core as mx


def enabled() -> bool:
    value = os.environ.get("DS4_ASYNC_CACHE")
    if value is not None:
        return value == "1"
    return (Path.home() / ".omlx" / "ds4_async_cache").exists()


def apply(force: bool = False) -> bool:
    """Replace the registered model's blocking cache eval when enabled."""
    if not force and not enabled():
        return False

    dsv4 = sys.modules.get("mlx_lm.models.deepseek_v4")
    if dsv4 is None:
        return False
    original = getattr(dsv4, "_materialize_cache_arrays", None)
    if original is None:
        return False
    if getattr(original, "_ds4_async_cache", False):
        return True

    def _materialize_cache_arrays(cache: Optional[Any]) -> None:
        if cache is None:
            return

        cache_arrays = []
        for layer_cache in cache:
            if layer_cache is None:
                continue
            leaves = getattr(layer_cache, "caches", None) or (layer_cache,)
            for leaf in leaves:
                if leaf is None:
                    continue
                for value in vars(leaf).values():
                    if isinstance(value, mx.array):
                        cache_arrays.append(value)

        if cache_arrays:
            mx.async_eval(*cache_arrays)

    _materialize_cache_arrays._ds4_async_cache = True
    _materialize_cache_arrays._ds4_original = original
    dsv4._materialize_cache_arrays = _materialize_cache_arrays
    return True
