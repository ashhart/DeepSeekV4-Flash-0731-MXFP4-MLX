#!/usr/bin/env python3
"""Fused decode-indexer scores vs the fallback chain. No model load."""
import sys
import mlx.core as mx
sys.path.insert(0, "/Users/ash/ds4")
import studio_guard
studio_guard.assert_safe(required_free_gib=4.0)
from ds4.indexer_decode import fused_scores

fails = checked = 0
mx.random.seed(3)
for L, P in [(1, 577), (4, 5891), (6, 129), (8, 64)]:
    q = (mx.random.normal((L, 64, 128)) * 0.3).astype(mx.bfloat16)
    pooled = (mx.random.normal((P, 128)) * 0.3).astype(mx.bfloat16)
    w = (mx.random.normal((L, 64)) * 0.1).astype(mx.bfloat16)
    # reference = fallback chain semantics
    ref = mx.maximum(q.astype(mx.float32) @ pooled.astype(mx.float32).T, 0)
    ref = (ref * w.astype(mx.float32)[..., None]).sum(axis=1)  # (L,P)
    got = fused_scores(q, pooled, w)
    got2 = fused_scores(q, pooled, w)
    mx.eval(ref, got, got2)
    d = float(mx.max(mx.abs(ref - got)))
    det = bool(mx.all(got == got2))
    checked += 1
    scale = float(mx.max(mx.abs(ref))) or 1.0
    if d / scale > 2e-3 or not det:
        fails += 1
        print(f"FAIL L={L} P={P} maxdiff={d:.5f} rel={d/scale:.5f} det={det}")
    else:
        print(f"ok   L={L} P={P} rel={d/scale:.2e} det={det}")
print(f"{checked-fails}/{checked} pass")
sys.exit(1 if fails else 0)
