#!/usr/bin/env python3
"""Derive ground-truth per-module quantization mode for the DeepSeek-V4-Flash
MXFP4/MXFP8 MLX checkpoint, straight from the safetensors headers.

No tensors are read -- only the JSON header of each shard -- so this runs in
about a second against a 156 GB checkpoint.

Discriminator (shapes are post-`sanitize`, derived arithmetically here):

    weight is uint32-packed, so one lane holds 32/bits values
      mxfp4 -> lane holds 8 values -> in_dim = 8 * W
      mxfp8 -> lane holds 4 values -> in_dim = 4 * W

    scales are one e8m0 byte per group of 32
      mxfp4 -> scales_last = in_dim/32 = W/4
      mxfp8 -> scales_last = in_dim/32 = W/8

`sanitize` reaches those post-shapes two different ways:

  * routed experts keep their per-32 scales as shipped (uint8 weight viewed as
    uint32, so W_u32 = W_u8 / 4) -- ratio W/4
  * everything else ships a per-128x128 block scale that gets broadcast with
    repeat(4, -1) then repeat(128, 0) -- ratio W/8

So we replicate that arithmetic instead of loading anything.
"""

import json
import struct
import sys
from collections import Counter, defaultdict
from pathlib import Path

MODEL_DIR = Path(sys.argv[1] if len(sys.argv) > 1 else ".")


def read_header(path: Path) -> dict:
    with open(path, "rb") as f:
        (header_len,) = struct.unpack("<Q", f.read(8))
        return json.loads(f.read(header_len))


def main() -> int:
    tensors: dict[str, dict] = {}
    for shard in sorted(MODEL_DIR.glob("model*.safetensors")):
        for name, info in read_header(shard).items():
            if name == "__metadata__":
                continue
            tensors[name] = info

    print(f"tensors in header: {len(tensors)}")

    # Group into modules that carry both .weight and .scales.
    modules: dict[str, dict] = defaultdict(dict)
    for name, info in tensors.items():
        if name.endswith(".weight"):
            modules[name[: -len(".weight")]]["weight"] = info
        elif name.endswith(".scales"):
            modules[name[: -len(".scales")]]["scales"] = info

    quantized = {k: v for k, v in modules.items() if "weight" in v and "scales" in v}
    passthrough = {k: v for k, v in modules.items() if "scales" not in v}
    print(f"quantized modules: {len(quantized)}")
    print(f"passthrough (no scales): {len(passthrough)}")

    dtypes = Counter(
        (v["weight"]["dtype"], v["scales"]["dtype"]) for v in quantized.values()
    )
    print(f"(weight,scales) dtypes: {dict(dtypes)}")

    inferred: dict[str, str] = {}
    unknown: list[tuple[str, list, list, str]] = []

    for mod, v in quantized.items():
        w_shape = v["weight"]["shape"]
        s_shape = v["scales"]["shape"]
        w_dtype = v["weight"]["dtype"]

        # Post-sanitize uint32 lane count along the input dim.
        if w_dtype in ("U8", "I8"):
            w_u32 = w_shape[-1] / 4  # uint8 bytes viewed as uint32
        elif w_dtype == "U32":
            w_u32 = w_shape[-1]
        else:
            unknown.append((mod, w_shape, s_shape, f"weight dtype {w_dtype}"))
            continue

        # This repo already ships MLX-native layout: every `.scales` tensor is
        # at per-32 granularity (that broadcast is the README's "+0.12%"), and
        # `sanitize` only renames -- it rewrites `.scale`, never `.scales`.
        s_last = s_shape[-1]

        if abs(s_last - w_u32 / 4) < 1e-9:
            inferred[mod] = "mxfp4"
        elif abs(s_last - w_u32 / 8) < 1e-9:
            inferred[mod] = "mxfp8"
        else:
            unknown.append(
                (mod, w_shape, s_shape, f"s_last={s_last} w_u32={w_u32} no match")
            )

    print(f"\ninferred: {dict(Counter(inferred.values()))}")
    if unknown:
        print(f"UNRESOLVED: {len(unknown)}")
        for row in unknown[:10]:
            print("  ", row)

    # Collapse to distinct structural patterns (expert/layer indices -> N/E).
    def pattern(name: str) -> str:
        out = []
        for part in name.split("."):
            out.append("N" if part.isdigit() else part)
        return ".".join(out)

    by_pattern: dict[str, Counter] = defaultdict(Counter)
    for mod, mode in inferred.items():
        by_pattern[pattern(mod)][mode] += 1

    print("\n=== distinct module patterns (ground truth from shapes) ===")
    for pat in sorted(by_pattern):
        modes = by_pattern[pat]
        flag = "  <-- MIXED!" if len(modes) > 1 else ""
        print(f"  {pat:<52} {dict(modes)}{flag}")

    # Cross-check against config.json's declared overrides.
    cfg = json.loads((MODEL_DIR / "config.json").read_text())
    q = cfg.get("quantization", {})
    default_mode = q.get("mode")
    overrides = {k: v for k, v in q.items() if isinstance(v, dict)}
    disabled = {k for k, v in q.items() if v is False}

    print(f"\nconfig default: mode={default_mode} bits={q.get('bits')}")
    print(f"config overrides: {len(overrides)}  disabled(False): {len(disabled)}")

    mismatch = []
    missing_override = []
    for mod, true_mode in inferred.items():
        declared = overrides.get(mod)
        declared_mode = declared["mode"] if declared else default_mode
        if declared_mode != true_mode:
            (mismatch if declared else missing_override).append(
                (mod, true_mode, declared_mode)
            )

    print(f"\nmodules where declared mode != true mode: {len(mismatch)}")
    for row in mismatch[:10]:
        print("  ", row)
    print(f"modules relying on default that is WRONG: {len(missing_override)}")
    for row in missing_override[:10]:
        print("  ", row)

    stray = [k for k in overrides if k not in inferred]
    print(f"\noverride keys with no matching quantized module: {len(stray)}")
    for k in stray[:10]:
        print("  ", k)

    print("\n=== passthrough modules (must NOT be quantized) ===")
    pt_pat = Counter(pattern(k) for k in passthrough)
    for pat, n in sorted(pt_pat.items()):
        print(f"  {pat:<52} x{n}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
