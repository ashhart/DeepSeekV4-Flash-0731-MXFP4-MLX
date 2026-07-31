# SPDX-License-Identifier: MIT
"""Build a `DSparkDrafter` and load the checkpoint's `mtp.*` tensors into it.

The checkpoint keeps DSpark under `mtp.<stage>.*` in source naming. This maps it
onto the module tree in `dspark_mlx.py`, applying the same transforms the main
model's `sanitize` does (per-expert -> stacked SwitchGLU, `w1/w2/w3` ->
`gate/down/up_proj`, `wo_a` -> 3D MultiLinear, `hc_*` -> `*_hc.*`).
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

W_TO_PROJ = {"w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"}


def _load_mtp_tensors(model_path: Path) -> dict:
    from omlx.patches.deepseek_v4.utils_patch import _load_safetensors

    out = {}
    for shard in sorted(glob.glob(str(model_path / "model*.safetensors"))):
        for k, v in _load_safetensors(shard).items():
            if k.startswith("mtp."):
                out[k] = v
    return out


def remap(weights: dict, n_experts: int, o_groups: int, o_lora_rank: int) -> dict:
    """`mtp.<i>.<rest>` -> DSparkDrafter parameter paths."""
    out: dict = {}

    for k, v in weights.items():
        _, idx, rest = k.split(".", 2)

        # Heads that live on the drafter itself, not on a stage.
        if rest.startswith("main_proj.") or rest.startswith("main_norm."):
            out[rest] = v
            continue
        if rest.startswith(("markov_head.", "confidence_head.")) or rest == "norm.weight":
            out[rest] = v
            continue
        if rest.startswith("hc_head_"):
            out[f"hc_head.{rest[len('hc_head_'):]}"] = v
            continue

        nk = f"stages.{idx}.{rest}"
        nk = nk.replace(".ffn.gate.bias", ".ffn.gate.e_score_correction_bias")
        for sub in ("attn", "ffn"):
            for param in ("fn", "base", "scale"):
                nk = nk.replace(f".hc_{sub}_{param}", f".{sub}_hc.{param}")
        for src, dst in W_TO_PROJ.items():
            nk = nk.replace(f".shared_experts.{src}.", f".shared_experts.{dst}.")
        out[nk] = v

    # Stack the 256 per-expert tensors into one SwitchGLU weight per projection.
    for idx in {k.split(".")[1] for k in out if k.startswith("stages.")}:
        prefix = f"stages.{idx}.ffn.experts"
        for src, dst in W_TO_PROJ.items():
            for suffix in ("weight", "scales"):
                if f"{prefix}.0.{src}.{suffix}" not in out:
                    continue
                stacked = [
                    out.pop(f"{prefix}.{e}.{src}.{suffix}") for e in range(n_experts)
                ]
                out[f"stages.{idx}.ffn.switch_mlp.{dst}.{suffix}"] = mx.stack(stacked)

    # wo_a is a MultiLinear: (o_groups, o_lora_rank, -1)
    for key in list(out):
        if ".attn.wo_a." in key and out[key].ndim == 2:
            out[key] = out[key].reshape(o_groups, o_lora_rank, -1)

    return out


def quant_spec(config_quant: dict, drafter) -> dict:
    """Per-module quantization for the drafter, derived from config.json.

    config.json keys the drafter's modules as `mtp.<i>.*`; translate onto the
    drafter's own paths so `class_predicate` can resolve them.
    """
    spec = {}
    for k, v in config_quant.items():
        if not isinstance(v, dict) or not k.startswith("mtp."):
            continue
        parts = k.split(".", 2)
        if len(parts) != 3:
            continue
        _, idx, rest = parts
        if rest.startswith("main_proj"):
            spec["main_proj"] = v
            continue
        nk = f"stages.{idx}.{rest}"
        for src, dst in W_TO_PROJ.items():
            nk = nk.replace(f".shared_experts.{src}", f".shared_experts.{dst}")
        if ".ffn.experts." in nk:
            head = nk.split(".ffn.experts.")[0]
            w = nk.rsplit(".", 1)[-1]
            nk = f"{head}.ffn.switch_mlp.{W_TO_PROJ.get(w, w)}"
        spec[nk] = v
    return spec


def build_drafter(model_path: Path, config: dict, main_args):
    from .dspark_mlx import DSparkDrafter

    ref_cfg = json.loads((model_path / "inference" / "config.json").read_text())
    dspark = {
        "dspark_block_size": config["dspark_block_size"],
        "dspark_noise_token_id": config["dspark_noise_token_id"],
        "dspark_target_layer_ids": config["dspark_target_layer_ids"],
        "dspark_markov_rank": config["dspark_markov_rank"],
        # config.json says num_nextn_predict_layers: 1, which is wrong for this
        # checkpoint -- inference/config.json has the real stage count.
        "n_mtp_layers": ref_cfg["n_mtp_layers"],
    }

    drafter = DSparkDrafter(main_args, dspark)

    weights = remap(
        _load_mtp_tensors(model_path),
        main_args.n_routed_experts,
        main_args.o_groups,
        main_args.o_lora_rank,
    )

    spec = quant_spec(config.get("quantization", {}), drafter)
    default = {
        "group_size": config["quantization"]["group_size"],
        "bits": config["quantization"]["bits"],
        "mode": config["quantization"]["mode"],
    }

    def class_predicate(path, module):
        if path in spec:
            return spec[path]
        if not hasattr(module, "to_quantized"):
            return False
        return f"{path}.scales" in weights

    nn.quantize(
        drafter,
        group_size=default["group_size"],
        bits=default["bits"],
        mode=default["mode"],
        class_predicate=class_predicate,
    )

    return drafter, weights, dspark
