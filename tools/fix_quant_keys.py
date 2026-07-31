#!/usr/bin/env python3
"""Add MLX module-path keys to the DeepSeek-V4-Flash checkpoint's
`config.json["quantization"]` block.

Why
---
MLX's mixed-precision convention keys `quantization` by *module path* -- that is
what `nn.quantize`'s `class_predicate(path, module)` receives, and what mlx-lm's
own `make_quantization_config` emits. This checkpoint instead keys it by
*checkpoint tensor* name (`layers.0.attn.wq_a`), which is what the safetensors
shards use.

`Model.sanitize` renames the tensors on the way in, so by the time the predicate
runs the two namespaces have diverged and every lookup misses. All 390 mxfp8
attention / shared-expert modules then silently fall back to the mxfp4 default,
and loading dies on the first one:

    ValueError: Expected shape (1024, 512) but received shape (1024, 1024)
                 for parameter model.layers.0.attn.wq_a.weight

    (4096/8 = 512 lanes at 4 bits, vs the true 4096/4 = 1024 lanes at 8 bits)

What this does
--------------
Purely additive: every original key is preserved, and the module-path spelling
is added alongside it. Nothing that resolves today can stop resolving.

The rename rules mirror `sanitize` exactly:

    layers.<N>.<rest>                   -> model.layers.<N>.<rest>
    embed                               -> model.embed_tokens
    head                                -> lm_head
    norm                                -> model.norm
    <p>.ffn.shared_experts.w1|w2|w3     -> <p>.ffn.shared_experts.gate|down|up_proj
    <p>.ffn.experts.<E>.w1|w2|w3        -> <p>.ffn.switch_mlp.gate|down|up_proj
                                           (all E collapse onto one stacked module)

`mtp.*` keys are left in place as well as remapped: `sanitize` drops mtp tensors
from the base model, and the separate oMLX MTP patch owns that subtree.

Writes config.json.bak once, then rewrites config.json. Idempotent.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

W_TO_PROJ = {"w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"}
TOP_LEVEL = {"embed": "model.embed_tokens", "head": "lm_head", "norm": "model.norm"}
SCALARS = {"group_size", "bits", "mode"}


# `mtp.<i>.<rest>` is nested under `.block.` only for these prefixes -- see
# mlx_lm_mtp/deepseek_v4_model.py. `main_proj`, `confidence_head.*` and
# `markov_head.*` stay at the MTP head's top level.
MTP_BLOCK_SUBS = ("attn.", "ffn.", "attn_norm.", "ffn_norm.", "hc_attn", "hc_ffn")


def _remap_experts(key: str) -> str:
    """Collapse per-expert names onto the stacked SwitchGLU / shared MLP."""
    if ".ffn.experts." in key:
        head, tail = key.split(".ffn.experts.", 1)
        parts = tail.split(".", 1)
        if len(parts) == 2 and parts[0].isdigit():
            w = parts[1]
            return f"{head}.ffn.switch_mlp.{W_TO_PROJ.get(w, w)}"
    elif ".ffn.shared_experts." in key:
        head, w = key.split(".ffn.shared_experts.", 1)
        return f"{head}.ffn.shared_experts.{W_TO_PROJ.get(w, w)}"
    return key


def expand(key: str) -> list[str]:
    """Checkpoint tensor name -> every MLX module path it can resolve to.

    Returns both the base-model spelling and, for `mtp.*`, the `.block.`-nested
    spelling the MTP patch produces, since the two loaders build different trees
    from the same checkpoint.
    """
    if key in TOP_LEVEL:
        return [TOP_LEVEL[key]]

    out = _remap_experts(key)

    if out.startswith("layers."):
        return [f"model.{out}"]

    if out.startswith("mtp."):
        parts = out.split(".", 2)
        if len(parts) == 3 and parts[1].isdigit():
            rest = parts[2]
            if any(rest.startswith(s) for s in MTP_BLOCK_SUBS):
                return [f"mtp.{parts[1]}.block.{rest}"]
        return [out] if out != key else []

    return [out] if out != key else []


def main() -> int:
    model_dir = Path(sys.argv[1])
    cfg_path = model_dir / "config.json"
    bak_path = model_dir / "config.json.bak"

    cfg = json.loads(cfg_path.read_text())
    quant = cfg.get("quantization")
    if not isinstance(quant, dict):
        print("no `quantization` block -- nothing to do")
        return 1

    original_keys = [k for k in quant if k not in SCALARS]
    added: dict[str, object] = {}
    collisions: list[str] = []

    for key in original_keys:
        value = quant[key]
        for mapped in expand(key):
            if mapped in quant and quant[mapped] != value:
                collisions.append(mapped)
                continue
            if mapped in added and added[mapped] != value:
                collisions.append(mapped)
                continue
            added[mapped] = value

    if collisions:
        print(f"ERROR: {len(collisions)} conflicting remaps, e.g. {collisions[:5]}")
        return 2

    new_keys = {k: v for k, v in added.items() if k not in quant}
    if not new_keys:
        print("already remapped -- config.json unchanged")
        return 0

    if not bak_path.exists():
        shutil.copy2(cfg_path, bak_path)
        print(f"backed up -> {bak_path.name}")

    quant.update(new_keys)
    cfg_path.write_text(json.dumps(cfg, indent=2))

    by_mode: dict[str, int] = {}
    for v in new_keys.values():
        label = v["mode"] if isinstance(v, dict) else repr(v)
        by_mode[label] = by_mode.get(label, 0) + 1

    print(f"original keys : {len(original_keys)}")
    print(f"added keys    : {len(new_keys)}  {by_mode}")
    print(f"total keys    : {len([k for k in quant if k not in SCALARS])}")
    for k in list(new_keys)[:6]:
        print(f"  + {k}  ->  {new_keys[k]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
