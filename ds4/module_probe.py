# SPDX-License-Identifier: MIT
"""Attention-vs-MoE attribution inside the target layer (FINDINGS S30 follow-up).

The cycle ledger shows the 43 target layers cost 44.6 ms at 23K, uniformly
~0.98 ms/layer, against a ~0.185 ms/layer weight roofline -- roughly 5x off.
Before writing any MoE kernel, we need to know how that 0.98 ms splits between
attention and the routed MoE. Two indexer kernels already died for want of this
kind of measurement.

Mechanism: wrap the attention and MoE classes; after each returns its lazy
graph, set a Metal label and async_eval it so the resulting command buffers are
attributed. Labels reuse the profiler's existing rule -- anything matching
`verify/target/layers_*` is bucketed by its suffix in `target_layer_pairs_ms`
(tools/profile_server_cycle.py:129) -- so no profiler change is needed.

Labels are per-KIND, not per-layer (`layers_all_attn`, `layers_all_ffn`), so
the profiler sums each kind across all 43 layers and reports one number per
cycle for each.

DIAGNOSTIC ONLY. Forcing ~86 extra evals per forward inflates absolute time;
only the attn:ffn RATIO is meaningful. Marker: ~/.omlx/ds4_module_probe.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import mlx.core as mx

_MARKER = Path.home() / ".omlx" / "ds4_module_probe"
_cache = {"at": 0.0, "on": False}

ATTENTION_CLASSES = ("LocalAttention", "CompressedAttention",
                     "SparseCompressedAttention")
MOE_CLASS = "DeepseekV4MoE"
# Round 2: attention is 19.55 ms at SHORT context against a 2.4 ms roofline
# (12%) -- that is op count, not bytes. Split it: hyper-connections and the
# indexer are called inside the attention block, so their labels subtract from
# the attn remainder and the four numbers still sum to the layer total.
INNER_CLASSES = (("HyperConnection", "hc"), ("Indexer", "idx"))


def _enabled() -> bool:
    now = time.monotonic()
    if now - _cache["at"] > 1.0:
        _cache["on"] = _MARKER.exists()
        _cache["at"] = now
    return _cache["on"]


def _wrap(cls, kind: str) -> bool:
    if cls is None or getattr(cls.__call__, "_ds4_module_probe", False):
        return False
    inner = cls.__call__

    def wrapped(self, *a, **kw):
        out = inner(self, *a, **kw)
        if _enabled():
            try:
                from ds4 import engine_hook as eh

                phase = getattr(eh._METAL_STATE, "phase", "model")
                # Only meaningful inside the target verify; the drafter and
                # prefill have their own phases and are left alone.
                if phase.endswith("/verify/target"):
                    # HyperConnection returns a tuple; Indexer can return None.
                    # async_eval on those would raise into the swallow-all
                    # below, silently zeroing this bucket and inflating attn.
                    items = out if isinstance(out, (tuple, list)) else (out,)
                    items = [t for t in items if isinstance(t, mx.array)]
                    if items:
                        eh._metal_label(f"{phase}/layers_all_{kind}")
                        mx.async_eval(*items)
            except Exception:  # noqa: BLE001 -- probing must never break decode
                pass
        return out

    wrapped._ds4_module_probe = True
    cls.__call__ = wrapped
    return True


def apply() -> bool:
    dsv4 = sys.modules.get("mlx_lm.models.deepseek_v4")
    if dsv4 is None:
        return False
    wrapped_any = False
    for name in ATTENTION_CLASSES:
        wrapped_any |= _wrap(getattr(dsv4, name, None), "attn")
    wrapped_any |= _wrap(getattr(dsv4, MOE_CLASS, None), "ffn")
    for name, kind in INNER_CLASSES:
        wrapped_any |= _wrap(getattr(dsv4, name, None), kind)
    return wrapped_any
