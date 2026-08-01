# SPDX-License-Identifier: MIT
"""Reduce hyper-connection dispatch count (FINDINGS S33).

HyperConnection is 6.82 ms/cycle -- 14% of the 42.4 ms base -- while moving
~22 MB, i.e. ~0.4% of roofline. It is not bandwidth, it is launch latency:
86 calls per forward, each doing cast -> rms_norm -> mixes matmul before
oMLX's fused sinkhorn kernel, with only 4 threadgroups of real work.

Cheapest intervention first: put the three pre-ops behind one `mx.compile`
so MLX fuses what it can and the graph is enqueued as fewer kernels. No new
Metal, no numerics rewrite -- the sinkhorn/collapse kernel is untouched.

If this measures null, the follow-up is a real fused pre-kernel (cast +
rms_norm + the (24, hc_mult*D) matmul in one dispatch), which is a bigger
build with the same target.

Marker: ~/.omlx/ds4_hc_fuse. Engagement is logged once, per the router lesson.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import mlx.core as mx

_MARKER = Path.home() / ".omlx" / "ds4_hc_fuse"
_cache = {"at": 0.0, "on": False}
_ENGAGED = False


def _enabled() -> bool:
    now = time.monotonic()
    if now - _cache["at"] > 1.0:
        _cache["on"] = _MARKER.exists()
        _cache["at"] = now
    return _cache["on"]


@mx.compile
def _pre(x, fn, norm_eps: float):
    y = x.astype(mx.float32)
    z = mx.fast.rms_norm(y.flatten(-2), None, norm_eps)
    return y, z @ fn.T


def apply() -> bool:
    dsv4 = sys.modules.get("mlx_lm.models.deepseek_v4")
    if dsv4 is None:
        return False
    cls = getattr(dsv4, "HyperConnection", None)
    if cls is None or getattr(cls.__call__, "_ds4_hc_fuse", False):
        return cls is not None and getattr(cls.__call__, "_ds4_hc_fuse", False)
    original = cls.__call__
    mod = sys.modules.get(cls.__module__)
    hc_kernel = getattr(mod, "_hc_kernel", None)
    hc_ops = getattr(mod, "_hc_ops", None)
    if hc_kernel is None or hc_ops is None:
        return False

    def wrapped(self, x: mx.array):
        if not _enabled():
            return original(self, x)
        try:
            global _ENGAGED
            if not _ENGAGED:
                _ENGAGED = True
                try:
                    import logging
                    logging.getLogger("omlx.ds4").info("hc fuse ENGAGED")
                except Exception:  # noqa: BLE001
                    pass
            y, mixes = _pre(x, self.fn, self.norm_eps)
            use_ops = (
                self.training
                or mx.default_device() != mx.gpu
                or not mx.metal.is_available()
            )
            fn = hc_ops if use_ops else hc_kernel
            return fn(x, y, mixes, self.scale, self.base,
                      self.hc_mult, self.sinkhorn_iters, self.hc_eps)
        except Exception:  # noqa: BLE001 -- any doubt -> untouched original
            return original(self, x)

    wrapped._ds4_hc_fuse = True
    cls.__call__ = wrapped
    return True
