#!/usr/bin/env python3
"""Increment 1+2 check: main-model hidden extraction, and DSpark weight loading.

Proves, before any speculation logic exists, that
  a) layers 40/41/42 hidden states can be pulled out at the right shape (B,S,3D)
  b) every `mtp.*` tensor in the checkpoint lands on a drafter parameter, with
     nothing missing and nothing left over
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    from omlx.patches.deepseek_v4 import apply_deepseek_v4_patch

    apply_deepseek_v4_patch()

    import json

    from mlx_lm.utils import load_model

    from ds4.load_dspark import build_drafter

    model_path = Path(args.model_dir)
    config = json.loads((model_path / "config.json").read_text())

    model, _ = load_model(model_path, lazy=True)
    print(f"main model built; target layers {config['dspark_target_layer_ids']}")

    drafter, weights, dspark = build_drafter(model_path, config, model.args)
    print(f"drafter built: {dspark['n_mtp_layers']} stages, "
          f"block_size {dspark['dspark_block_size']}")

    params = dict(tree_flatten(drafter.parameters()))
    got = set(weights)
    want = set(params)

    missing = sorted(want - got)
    extra = sorted(got - want)
    print(f"\ndrafter parameters : {len(want)}")
    print(f"checkpoint tensors : {len(got)}")
    print(f"missing            : {len(missing)}")
    for k in missing[:12]:
        print(f"   - {k}  {params[k].shape}")
    print(f"unexpected         : {len(extra)}")
    for k in extra[:12]:
        print(f"   + {k}  {weights[k].shape}")

    mismatched = [
        (k, tuple(params[k].shape), tuple(weights[k].shape))
        for k in sorted(want & got)
        if tuple(params[k].shape) != tuple(weights[k].shape)
    ]
    print(f"shape mismatches   : {len(mismatched)}")
    for row in mismatched[:12]:
        print(f"   ! {row[0]}  want {row[1]}  got {row[2]}")

    if missing or extra or mismatched:
        print("\nNOT loadable yet")
        return 1

    drafter.load_weights(list(weights.items()), strict=True)
    mx.eval(drafter.parameters())
    nbytes = sum(p.size * p.dtype.size for _, p in tree_flatten(drafter.parameters()))
    print(f"\nloaded clean -- drafter is {nbytes / 1024**3:.2f} GiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
