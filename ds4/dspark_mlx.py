# SPDX-License-Identifier: MIT
"""DSpark speculative decoder for DeepSeek-V4-Flash, in MLX.

Ported from the reference `inference/model.py` that ships inside the checkpoint
(`DSparkBlock`, `DSparkMarkovHead`, `DSparkConfidenceHead`,
`Transformer.forward_spec`).

Why this exists
---------------
oMLX's `MTPBlock` implements the DeepSeek-*V3* MTP head (`e_proj` + `h_proj`,
`enorm` + `hnorm`). V4-Flash is a different design and will not load against it:

  * a single `main_proj` of `dim*len(target_layer_ids) -> dim` (12288 -> 4096),
    fusing the hidden states of layers 40/41/42, plus one `main_norm`
  * three *heterogeneous* stages, not N copies of one block:
      stage 0  main_proj + main_norm, then a block
      stage 1  block only
      stage 2  block + norm + hc_head + markov_head + confidence_head
  * `config.json` says `num_nextn_predict_layers: 1`; the checkpoint ships 3
    (`inference/config.json` has it right as `n_mtp_layers: 3`)

The trick that makes it fast
----------------------------
The drafter is *not* autoregressive. All `block_size` (=5) draft positions run
in a single forward pass: position 0 holds the real token and positions 1..n are
a fixed noise token, with the real context arriving through `main_x` in the KV
cache. `get_dspark_topk_idxs` hands every query position the same index set, so
there is no causal mask inside the block. Token-to-token dependency is then
injected cheaply in `forward_head` by a low-rank Markov bigram prior.

Consequence for rollback: the drafter keeps **no persistent state except the
rotating window of main-model KV**. A rejected draft costs nothing to undo.
"""

from __future__ import annotations

from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.base import scaled_dot_product_attention
from mlx_lm.models.deepseek_v4 import (
    DeepseekV4MoE,
    DeepseekV4RoPE,
    ModelArgs,
)
from mlx_lm.models.hyper_connection import HyperConnection, HyperHead, hc_expand
from mlx_lm.models.mla import MultiLinear


class DSparkAttention(nn.Module):
    """Window attention whose cache is fed by the *main* model's hidden state.

    Mirrors `DSparkAttention.forward` in the reference. Two distinct sources:

      * `main_x` -> KV written into a rotating `window_size` cache (this is the
        only thing that persists between cycles)
      * `x`      -> Q and KV for the `block_size` draft positions

    RoPE is applied at insert time, so each cached entry already carries its
    absolute position and the rotating buffer's physical order is irrelevant.
    """

    def __init__(self, config: ModelArgs, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.compress_ratio = 0  # the reference asserts this for DSpark stages
        self.n_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.o_groups = config.o_groups
        self.o_lora_rank = config.o_lora_rank
        self.window_size = config.sliding_window
        self.scale = self.head_dim**-0.5

        self.wq_a = nn.Linear(config.hidden_size, config.q_lora_rank, bias=False)
        self.q_norm = nn.RMSNorm(config.q_lora_rank, eps=config.rms_norm_eps)
        self.wq_b = nn.Linear(
            config.q_lora_rank, self.n_heads * self.head_dim, bias=False
        )
        self.wkv = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.kv_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.wo_a = MultiLinear(
            self.n_heads * self.head_dim // config.o_groups,
            config.o_lora_rank,
            config.o_groups,
        )
        self.wo_b = nn.Linear(
            config.o_groups * config.o_lora_rank,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.attn_sink = mx.zeros((self.n_heads,), dtype=mx.float32)
        self.rope = DeepseekV4RoPE(
            config.qk_rope_head_dim,
            config.rope_theta,
            None,
            config.max_position_embeddings,
        )

    def main_kv(self, main_x: mx.array, offset: int) -> mx.array:
        """KV contribution of the main model's hidden state, RoPE'd in place."""
        B, S, _ = main_x.shape
        kv = self.kv_norm(self.wkv(main_x)).reshape(B, 1, S, self.head_dim)
        return self.rope(kv, offset)

    def __call__(
        self,
        x: mx.array,
        window_kv: mx.array,
        offset: int,
    ) -> mx.array:
        """`window_kv` is the rotating cache *including* this step's main KV."""
        B, L, _ = x.shape

        q = self.wq_b(self.q_norm(self.wq_a(x)))
        q = q.reshape(B, L, self.n_heads, self.head_dim)
        q = mx.fast.rms_norm(q, None, self.config.rms_norm_eps)
        q = q.transpose(0, 2, 1, 3)
        q = self.rope(q, offset)

        kv = self.kv_norm(self.wkv(x)).reshape(B, 1, L, self.head_dim)
        kv = self.rope(kv, offset)

        # Every draft position sees the whole window plus every draft position:
        # no causal mask inside the block (get_dspark_topk_idxs is query-invariant).
        kv = mx.concatenate([window_kv, kv], axis=2)

        out = scaled_dot_product_attention(
            q,
            kv,
            kv,
            cache=None,
            scale=self.scale,
            mask=None,
            sinks=self.attn_sink.astype(q.dtype),
        )
        out = self.rope(out, offset, inverse=True)

        out = out.reshape(B, self.o_groups, -1, L, self.head_dim)
        out = out.transpose(0, 1, 3, 2, 4).flatten(-2)
        out = self.wo_a(out)
        out = out.transpose(0, 2, 1, 3).flatten(-2)
        return self.wo_b(out)


class DSparkStage(nn.Module):
    """One DSpark stage: the reference `Block` with `DSparkAttention`."""

    def __init__(self, config: ModelArgs, layer_idx: int):
        super().__init__()
        self.attn = DSparkAttention(config, layer_idx)
        self.ffn = DeepseekV4MoE(config, layer_idx)
        self.attn_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.ffn_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn_hc = HyperConnection(config)
        self.ffn_hc = HyperConnection(config)

    def __call__(
        self,
        h: mx.array,
        window_kv: mx.array,
        offset: int,
        input_ids: mx.array,
    ) -> mx.array:
        residual = h
        x, post, comb = self.attn_hc(h)
        x = self.attn(self.attn_norm(x), window_kv, offset)
        h = hc_expand(x, residual, post, comb)

        residual = h
        x, post, comb = self.ffn_hc(h)
        x = self.ffn(self.ffn_norm(x), input_ids)
        return hc_expand(x, residual, post, comb)


class DSparkMarkovHead(nn.Module):
    """Low-rank bigram prior: token -> rank-256 embedding -> vocab logit bias."""

    def __init__(self, vocab_size: int, rank: int):
        super().__init__()
        self.markov_w1 = nn.Embedding(vocab_size, rank)
        self.markov_w2 = nn.Linear(rank, vocab_size, bias=False)

    def __call__(self, token_ids: mx.array):
        embed = self.markov_w1(token_ids)
        return self.markov_w2(embed), embed


class DSparkConfidenceHead(nn.Module):
    """concat(hidden, markov_embed) -> scalar confidence, computed in fp32."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.proj = nn.Linear(input_dim, 1, bias=False)

    def __call__(self, hidden: mx.array, markov_embed: mx.array) -> mx.array:
        x = mx.concatenate(
            [hidden.astype(mx.float32), markov_embed.astype(mx.float32)], axis=-1
        )
        return (x @ self.proj.weight.astype(mx.float32).T).squeeze(-1)


class DSparkDrafter(nn.Module):
    """The full three-stage drafter.

    `window_kv` is owned by the caller (one array per stage) so that a rejected
    cycle is undone by simply not committing it.
    """

    def __init__(self, config: ModelArgs, dspark: dict):
        super().__init__()
        self.config = config
        self.hc_mult = config.hc_mult
        self.block_size = dspark["dspark_block_size"]
        self.noise_token_id = dspark["dspark_noise_token_id"]
        self.target_layer_ids = list(dspark["dspark_target_layer_ids"])
        self.window_size = config.sliding_window
        n_stages = dspark["n_mtp_layers"]
        self.n_stages = n_stages

        n_main = config.num_hidden_layers
        self.stages = [DSparkStage(config, n_main + i) for i in range(n_stages)]

        # stage 0 only
        self.main_proj = nn.Linear(
            config.hidden_size * len(self.target_layer_ids),
            config.hidden_size,
            bias=False,
        )
        self.main_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # last stage only
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hc_head = HyperHead(config)
        self.markov_head = DSparkMarkovHead(
            config.vocab_size, dspark["dspark_markov_rank"]
        )
        self.confidence_head = DSparkConfidenceHead(
            config.hidden_size + dspark["dspark_markov_rank"]
        )

    def new_window(self, batch: int, dtype=mx.bfloat16) -> list:
        head_dim = self.config.head_dim
        return [mx.zeros((batch, 1, 0, head_dim), dtype=dtype) for _ in self.stages]

    def fuse(self, main_hidden: mx.array) -> mx.array:
        """(B, S, 3D) hidden from layers 40/41/42 -> (B, S, D)."""
        return self.main_norm(self.main_proj(main_hidden))

    def push_window(self, window: list, main_hidden: mx.array, offset: int) -> list:
        """Append main-model positions [offset, offset+S) to each stage's window.

        Must be called exactly once per position the main model consumes -- the
        reference writes one slot per step (`kv_cache[:, start_pos % win]`), so a
        duplicated position silently corrupts the drafter's context.
        """
        main_x = self.fuse(main_hidden)
        out = []
        for stage, w in zip(self.stages, window):
            kv = stage.attn.main_kv(main_x, offset)
            w = mx.concatenate([w, kv.astype(w.dtype)], axis=2)
            if w.shape[2] > self.window_size:
                w = w[:, :, -self.window_size :, :]
            out.append(w)
        return out

    def __call__(
        self,
        last_token: mx.array,
        embed_tokens: nn.Module,
        lm_head: nn.Module,
        window: list,
        offset: int,
    ):
        """One draft cycle. `window` must already include position `offset`.

        last_token : (B, 1) the token the main model just produced, which sits at
                     position `offset + 1`
        returns (draft_ids (B, block_size), confidence (B, block_size))
        """
        B = last_token.shape[0]
        k = self.block_size

        # position 0 = the real token, 1..k-1 = noise
        noise = mx.full((B, k - 1), self.noise_token_id, dtype=last_token.dtype)
        draft_input_ids = mx.concatenate([last_token, noise], axis=1)

        h = embed_tokens(draft_input_ids)
        h = mx.contiguous(
            mx.broadcast_to(h[:, :, None, :], (B, k, self.hc_mult, h.shape[-1]))
        )

        for stage, w in zip(self.stages, window):
            h = stage(h, w, offset + 1, draft_input_ids)

        x = self.hc_head(h)  # (B, k, D)
        logits = lm_head(self.norm(x))  # (B, k, V)

        # Sequential only here: a bigram bias chains the otherwise-parallel block.
        ids = [last_token[:, 0]]
        embeds = []
        for i in range(k):
            bias, embed = self.markov_head(ids[-1])
            embeds.append(embed)
            ids.append(mx.argmax(logits[:, i] + bias.astype(logits.dtype), axis=-1))

        draft_ids = mx.stack(ids[1:], axis=1)
        markov_embed = mx.stack(embeds, axis=1)
        confidence = self.confidence_head(x, markov_embed)
        return draft_ids, confidence
