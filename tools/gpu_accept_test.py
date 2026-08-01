#!/usr/bin/env python3
"""Unit-test the GPU-side acceptance kernel against the host loop. No model.

The kernel must reproduce EXACTLY:
    n = 0
    while n < k and samples[n] == drafts[n]: n += 1
    committed = drafts[:n] + [samples[n]]
for every k in 1..8, including n=0 (immediate reject), n=k (accept all), and
mid-prefix breaks -- plus bit-determinism across double invocation.
"""

from __future__ import annotations

import random
import sys

import mlx.core as mx

sys.path.insert(0, "/Users/ash/ds4")
import studio_guard

studio_guard.assert_safe(required_free_gib=4.0)

from ds4.gpu_accept import gpu_accept

random.seed(11)
failures = 0
checked = 0

cases = []
for k in range(1, 9):
    # accept-all, reject-all, and every mid break point
    for n_true in range(0, k + 1):
        drafts = [random.randrange(0, 129280) for _ in range(k)]
        samples = list(drafts[:n_true])
        if n_true < k:
            # force a mismatch at position n_true
            bad = drafts[n_true]
            while bad == drafts[n_true]:
                bad = random.randrange(0, 129280)
            samples.append(bad)
        # target has k+1 entries; fill the rest arbitrarily
        while len(samples) < k + 1:
            samples.append(random.randrange(0, 129280))
        cases.append((k, drafts, samples))
# plus random fuzz
for _ in range(200):
    k = random.randrange(1, 9)
    drafts = [random.randrange(0, 50) for _ in range(k)]   # small vocab -> natural matches
    samples = [random.randrange(0, 50) for _ in range(k + 1)]
    cases.append((k, drafts, samples))

for k, drafts, samples in cases:
    checked += 1
    # host reference
    n_ref = 0
    while n_ref < k and samples[n_ref] == drafts[n_ref]:
        n_ref += 1
    emit_ref = drafts[:n_ref] + [samples[n_ref]] + [-1] * (k - n_ref)

    t = mx.array(samples, dtype=mx.int32)
    d = mx.array(drafts, dtype=mx.int32)
    n1, e1 = gpu_accept(t, d)
    n2, e2 = gpu_accept(t, d)
    mx.eval(n1, e1, n2, e2)

    if int(n1[0]) != n_ref or e1.tolist() != emit_ref:
        failures += 1
        print(f"MISMATCH k={k} ref_n={n_ref} got_n={int(n1[0])} "
              f"ref_emit={emit_ref} got={e1.tolist()}")
    elif int(n2[0]) != int(n1[0]) or e2.tolist() != e1.tolist():
        failures += 1
        print(f"NONDETERMINISTIC k={k}")

print(f"\n{checked - failures}/{checked} cases exact vs host loop")
sys.exit(0 if failures == 0 else 1)
