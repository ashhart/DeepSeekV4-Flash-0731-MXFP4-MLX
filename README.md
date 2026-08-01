# DeepSeek-V4-Flash-0731 MXFP4 — fast on Apple Silicon

Fixes and speed patches for running
[**Vontra/DeepSeek-V4-Flash-0731-MXFP4-MLX**](https://huggingface.co/Vontra/DeepSeek-V4-Flash-0731-MXFP4-MLX)
(305B total / 13B active) under [oMLX](https://omlx.app) on a Mac.

Three things live here:

1. **A loading fix.** The published `config.json` keyed its quantization
   overrides by checkpoint tensor name, but MLX matches on module path. Every
   lookup missed, all 390 MXFP8 attention modules silently fell back to MXFP4,
   and the model did not load at all. *(Now also fixed upstream on the Hub — the
   tool here is for older revisions and for verification.)*
2. **A windowed prefill kernel.** Default attention builds a dense `(L, L)` mask
   even though `sliding_window` is 128, making prefill quadratic. Blocking the
   queries removes that. **Prefill stops degrading with context length.**
3. **A DSpark speculative decoder.** The checkpoint ships a 3-stage drafter that
   no MLX runtime implements. Ported here, and its acceptance is excellent —
   but the KV rollback is not yet sound, so **no decode speedup is claimed**.
   See *Status* below before using it.

---

## Measured

Mac Studio, **M3 Ultra** (80-core GPU, 256 GB, 819 GB/s), macOS 26.6,
mlx 0.32.0 / mlx-lm 0.31.3, oMLX. Weights resident 145.5 GiB.

### Prefill, through the oMLX API

| prompt tokens | stock | **patched** |
|---|---|---|
| 7,209 | — | 518 tok/s |
| 14,409 | — | **937 tok/s** |
| 28,809 | — | **938 tok/s** |

Flat from 14K to 28K. For comparison, measured directly against the dense path:

| L | dense | windowed | speedup |
|---|---|---|---|
| 4,096 | 425 | 536 | 1.26x |
| 8,192 | 345 | 603 | 1.74x |
| 16,384 | 183 | 595 | **3.25x** |

The dense path halves from 8K to 16K; the windowed path does not move. That
shape matters more than the ratio for agentic use.

### Decode

| | tok/s |
|---|---|
| baseline greedy | 28-30 |
| + DSpark speculative decoding (k=3) | ~40 (1.41x) |
| **+ DSpark + 8-bit `lm_head`** | **43-48** (1.5-1.6x) |

#### `lm_head` — the biggest lever, and nothing to do with speculation

The checkpoint deliberately leaves the output head unquantized (`config.json`
has `"head": false`), so it is 129280x4096 in bf16 = **1.06 GB read per
forward** — roughly 10% of everything a decode step moves. Speculation reads it
**twice** per cycle (once in the drafter, once in the verify), so ~17% of the
cycle's bytes. Quantizing it halves that:

| `lm_head` | bytes/read | speculative tok/s |
|---|---|---|
| bf16 (stock) | 1.06 GB | 38.9 |
| **8-bit** | **0.56 GB** | **46.4-48.2** |
| 6-bit | 0.43 GB | 43.3 |

6-bit moves *fewer* bytes than 8-bit and is *slower* — MLX's 8-bit quantized
matmul is better optimised, so do not assume narrower is faster. Opt-in, since
it does perturb the logits:

```sh
echo 8 > ~/.omlx/ds4_head_bits     # or DS4_QUANT_HEAD=8
```

#### Draft width

Measured over 200 tokens, after the rollback was fixed:

| verify k | speedup | accepted prefix | clamped | restore+replay |
|---|---|---|---|---|
| 2 | 1.28x | 1.49/2 | 0% | 0% |
| **3** | **1.41-1.43x** | **2.01/3** | **0%** | **0%** |
| 5 | 0.94-1.09x | 2.41/5 | 26% | 21% |
| 7 | 0.49x | 1.65/7 | 50% | 32% |

**Draft width and verify width are separate knobs** (`DS4_SPEC_DRAFT_WIDTH` vs
`DS4_SPEC_BLOCK`). The drafter's block is non-causal — every draft position
attends to every other — so its width is part of its input distribution, not a
free parameter. It is left at the trained value while only a prefix is verified.
Drafting wider than you verify buys a little acceptance (1.67 -> 1.82 going
3 -> 7) for a little more drafter compute.

Verifying wider is worse, for two compounding reasons:

1. Padding the draft block past the trained width puts out-of-distribution noise
   positions in view of the real ones, so the accepted prefix actually *falls*
   (2.41 at k=5 -> 1.65 at k=7) despite drafting more tokens.
2. Above k=3 the `PoolingCache` rollback starts refusing, and each refusal costs
   a full cache restore plus a replay forward.

> An earlier revision of this README claimed 47.1 / 1.72x, and briefly 2.06x at
> k=7. **Those are withdrawn** — they were measured against a corrupt KV cache
> (see *The rollback bug*). Output looked plausible because a rejected draft is
> by construction a plausible continuation.

#### Standalone drafter acceptance

Measured against ground-truth autoregressive decoding with no rollback involved,
at the trained width on clean code — so unaffected by any of the above:

| content | pos 1 | pos 2 | pos 3 | pos 4 | pos 5 | E[prefix] |
|---|---|---|---|---|---|---|
| structured code | 100% | 96% | 96% | 92% | 96% | **4.75** |
| open-ended prose | 71% | 21% | 8% | 17% | 8% | 1.00 |

A 6-token verify costs ~1.80x one decode step. Acceptance is strongly
content-dependent; in a real generation loop against mixed chat traffic it runs
nearer 2.0/3, which is what the tables above reflect.

---

## Install

Needs oMLX at `/Applications/oMLX.app` and the model already downloaded.

```sh
git clone https://github.com/ashhart/DeepSeekV4-Flash-0731-MXFP4-MLX.git
cd DeepSeekV4-Flash-0731-MXFP4-MLX
./install.sh
```

Then restart oMLX:

```sh
pkill -f 'oMLX|omlx-server'; open -a oMLX
```

Windowed prefill is now active on every model load. Confirm:

```sh
./install.sh --verify
```

### What install.sh actually does

- copies `ds4/` to `~/ds4`
- writes one `.pth` file to `~/.local/lib/python3.11/site-packages/`

**It does not modify anything inside `/Applications/oMLX.app`.** The `.pth` runs
at interpreter startup and registers a `sys.meta_path` hook that waits for
oMLX's own `deepseek_v4` patch module and appends ours after it. An oMLX update
can remove the hook but can never conflict with it — just re-run `install.sh`.

Remove with `./install.sh --uninstall`, or disable temporarily with
`DS4_PATCHES=0`.

### Recommended oMLX settings

In `~/.omlx/settings.json`:

```json
{ "scheduler": { "chunked_prefill": false } }
```

Chunked prefill is a *different* fix for the same problem and the two work
against each other: chunking shrinks `L` for the whole layer, and this model's
MoE is ~2x less efficient at small `L` (1812 vs 857 ms per 1k tokens at L=1024
vs 4096). Windowed prefill blocks *inside* attention, keeping the MoE at full
width, and measured strictly better (3.25x vs 1.47x at 8K).

---

## The loading fix

If you pinned a revision of the model from before 2026-07-31, you will hit:

```
ValueError: Expected shape (1024, 512) but received shape (1024, 1024)
             for parameter model.layers.0.attn.wq_a.weight
```

`wq_a` is 4096 -> 1024. Packed into `uint32` lanes that is `4096*bits/32`: 512
lanes at 4 bits, 1024 at 8. So the loader built the layer as MXFP4 when the
checkpoint is MXFP8.

`config.json` carried 390 per-module MXFP8 overrides, but keyed them as
`layers.0.attn.wq_a` (checkpoint tensor name) while `nn.quantize` hands
`class_predicate` a module path (`model.layers.0.attn.wq_a`) — `sanitize`
renames tensors on load and nothing applied the same renaming to the
quantization keys. Every lookup missed and fell through to the MXFP4 default.

Fix an affected checkout in place (additive, idempotent, backs up):

```sh
python3 tools/fix_quant_keys.py ~/.omlx/models/Vontra/DeepSeek-V4-Flash-0731-MXFP4-MLX
```

Verify against the safetensors headers — derives each module's true mode from
the packing ratio (`scales_last == lanes/4 -> mxfp4`, `lanes/8 -> mxfp8`)
without reading a single tensor:

```sh
python3 tools/audit_quant.py ~/.omlx/models/Vontra/DeepSeek-V4-Flash-0731-MXFP4-MLX
# expect: 390 mxfp8 + 35328 mxfp4, zero mismatches, zero stray keys
```

> **Note for oMLX maintainers:** `omlx/utils/model_loading.py` already has
> `expand_per_layer_quant_keys()`, which does exactly this — but it is only
> called from the VLM path (`engine/vlm.py`), never from the batched LLM path.
> Calling it there would fix this class of bug generally.

---

## DSpark speculative decoding

**Working and opt-in.** Enable with a marker file, then restart oMLX:

```sh
touch ~/.omlx/ds4_spec_enabled     # on
rm    ~/.omlx/ds4_spec_enabled     # off
pkill -f 'oMLX|omlx-server'; open -a oMLX
```

Knobs: `DS4_SPEC_BLOCK` (verify width, default 3), `DS4_SPEC_DRAFT_WIDTH`
(default = trained width), `DS4_QUANT_HEAD` / `~/.omlx/ds4_head_bits`.
Confirm it engaged:

```sh
grep omlx.ds4 ~/.omlx/logs/server.log
```

> Do **not** use oMLX's own `mtp_enabled` toggle for this model. That builds
> oMLX's DeepSeek-**V3**-shaped MTP heads (`e_proj`/`h_proj`) and fails the weight
> load outright with 3140 unmatched tensors. `dflash_enabled` needs a trained
> draft checkpoint that does not exist for this target.

### What the checkpoint actually ships

DSpark lives under `mtp.*` as three *heterogeneous* stages, not the V3 MTP head:

- `mtp.0` — `main_proj` (`3*dim -> dim`, fusing hidden states from layers
  40/41/42 per `dspark_target_layer_ids`) + `main_norm`, then a block
- `mtp.1` — block only
- `mtp.2` — block + `norm` + `hc_head` + `markov_head` (low-rank bigram prior) +
  `confidence_head`

`config.json` says `num_nextn_predict_layers: 1`; `inference/config.json` in the
checkpoint has it right as `n_mtp_layers: 3`.

The reason it is fast: the drafter is **not autoregressive**. All draft positions
run in one forward — position 0 holds the real token, the rest a fixed noise
token, with real context arriving via `main_x` in the KV cache. Every query
position gets the same index set, so there is no causal mask inside the block.
Token-to-token dependency is added afterwards by the cheap Markov bigram prior.
That same non-causality is why drafting wider than the trained width backfires.

### The rollback bug (worth reading if you build on this)

Rejected drafts must come back out of the KV cache. Getting that wrong does not
crash — it produces fluent repetition ("The user id. The user id. ...") while
throughput *looks* great, because a rejected draft is a plausible continuation.
Three distinct traps, all hit here:

1. **`CacheList` is not a `list` subclass** — it wraps `.caches`. An
   `isinstance(x, (list, tuple))` check silently skips every compressed layer.
2. **Snapshots must be detached.** The server uses `BatchRotatingKVCache`, whose
   `offset` is an `mx.array` **mutated in place**; a plain reference reads back
   the *post*-update value, making the snapshot a no-op. oMLX's own
   `cache_rollback` does `v = v + 0` for exactly this. Snapshot the whole
   instance dict, not a hand-listed set of fields.
3. **oMLX's MTP patch is self-healing** — it reinstalls its own
   `DeepseekV4Model.__call__` whenever it sees a foreign one, during model load.
   A class-level "already patched" flag therefore never re-wraps; mark the
   function instead. oMLX also re-registers `mlx_lm.models.deepseek_v4` from
   source per load, so class patches must be re-applied every time.

Rollback itself goes through oMLX's own tested helpers — `mtp_clamp_accept`
(reduce accepted until every layer can undo), `mtp_partial_rollback`, and the
armed undo log (`set_undo_armed`, covering updates of 2..8 tokens). Do not
hand-roll these; they already exist and are correct.

`tools/cache_invariants.py` and `tools/pooling_rollback_test.py` reproduce the
cache behaviour with no model load.

### Standalone scripts

```sh
# acceptance on your own content
python3 ds4/acceptance.py <model_dir> --prompt "$(cat some_file.py)"

# through the real BatchGenerator, hook off vs on
python3 ds4/bench_engine.py <model_dir> --max-tokens 200
```

See [`RUNNING.md`](RUNNING.md).

### On exactness

Speculative output is **not** token-identical to greedy, and that is not a
rollback bug. `ds4/check_exactness.py` feeds identical tokens down both paths
(teacher-forced, no drift):

```
max |logit diff| : 1.89     argmax agreement : 5/6
```

Layer by layer the divergence is exactly `0.00391` (= 2^-8, one bf16 ULP) through
layers 0-4, then jumps 10x at the first layer where a top-k decision flips. A
k-token forward reassociates the same matmuls differently than k single-token
steps, and 1 ULP is enough to flip which of 256 experts the router picks.
Batching the verify *is* speculative decoding, so this is unavoidable — vLLM on
DGX Spark has the same property. Every committed token still comes from the
verify pass's own argmax, so the output is a sound greedy decode, just not the
same greedy path.

## Layout

```
ds4/windowed_prefill.py   blocked prefill attention (auto-applied)
ds4/engine_hook.py        speculative decoding wired into GenerationBatch._step
ds4/boot.py               sys.meta_path hook installed by the .pth
ds4/dspark_mlx.py         DSpark drafter: 3 stages, markov + confidence heads
ds4/load_dspark.py        maps mtp.* checkpoint tensors onto the drafter
ds4/spec_decode.py        speculative decode loop + rollback
ds4/acceptance.py         acceptance measurement
ds4/check_exactness.py    multi-token vs single-token numerics
tools/fix_quant_keys.py   config.json quantization key remap
tools/audit_quant.py      ground truth from safetensors headers
tools/bench_windowed.py   dense vs windowed prefill
tools/server_prefill.py   prefill through the oMLX HTTP API
tools/multitoken_cost.py  cost of a k-token verify
tools/cache_invariants.py RotatingKVCache trim behaviour, no model load
tools/pooling_rollback_test.py  PoolingCache rollback exactness, no model load
```

## Credits

Model conversion: [Vontra](https://huggingface.co/Vontra). Base model:
[deepseek-ai/DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731).
DSpark ported from the reference `inference/model.py` shipped in the checkpoint.

MIT.
