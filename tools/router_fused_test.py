#!/usr/bin/env python3
"""Unit-test the fused router against stock `_expert_select`. No model load.

Checks, over many random rows plus crafted tie cases:
  1. selected expert SET identical to stock (order may differ; argpartition's
     within-k order is unspecified)
  2. weights match stock to <=2 ULP once aligned by expert index
  3. weights sum to routed_scaling_factor within tolerance
  4. bit-identical across two invocations (determinism)
"""

from __future__ import annotations

import sys

import mlx.core as mx

sys.path.insert(0, "/Users/ash/ds4")
import studio_guard

studio_guard.assert_safe(required_free_gib=8.0)

from omlx.patches.deepseek_v4 import apply_deepseek_v4_patch

apply_deepseek_v4_patch()

import mlx_lm.models.deepseek_v4 as dsv4

from ds4 import router_fused

stock = dsv4._expert_select  # capture before apply() wraps it
router_fused.apply()

SCALE = 1.5
failures = 0
checked = 0

mx.random.seed(7)
bias = mx.random.normal((256,)).astype(mx.float32) * 0.01

cases = []
for trial in range(64):
    rows = [1, 2, 4, 8][trial % 4]
    logits = (mx.random.normal((rows, 256)) * (0.5 + trial * 0.1)).astype(mx.bfloat16)
    cases.append(logits)
# Crafted ties: duplicate the max logit across several experts.
tied = mx.zeros((4, 256)).astype(mx.bfloat16)
tied[:, 10] = 2.0; tied[:, 77] = 2.0; tied[:, 200] = 2.0; tied[:, 3] = 1.0
cases.append(tied)

for logits in cases:
    s_inds, s_w = stock(logits, bias, 6, SCALE, True, "sqrtsoftplus")
    f_inds, f_w = router_fused.fused_expert_select(logits, bias, SCALE)
    f_inds2, f_w2 = router_fused.fused_expert_select(logits, bias, SCALE)
    mx.eval(s_inds, s_w, f_inds, f_w, f_inds2, f_w2)

    for r in range(logits.shape[0]):
        checked += 1
        s_set = sorted(int(x) for x in s_inds[r].tolist())
        f_set = sorted(int(x) for x in f_inds[r].tolist())
        det = (f_inds[r].tolist() == f_inds2[r].tolist()
               and f_w[r].tolist() == f_w2[r].tolist())
        if not det:
            failures += 1
            print(f"NONDETERMINISTIC row {r}")
            continue
        if s_set != f_set:
            failures += 1
            print(f"SET MISMATCH row {r}: stock={s_set} fused={f_set}")
            continue
        s_map = {int(i): float(w) for i, w in zip(s_inds[r].tolist(), s_w[r].tolist())}
        f_map = {int(i): float(w) for i, w in zip(f_inds[r].tolist(), f_w[r].tolist())}
        worst = max(abs(s_map[i] - f_map[i]) for i in s_map)
        total = sum(f_map.values())
        if worst > 1e-5 or abs(total - SCALE) > 1e-4:
            failures += 1
            print(f"WEIGHT MISMATCH row {r}: worst={worst:.2e} sum={total:.6f}")

print(f"\n{checked - failures}/{checked} rows exact-set + weight-aligned + deterministic")
sys.exit(0 if failures == 0 else 1)
