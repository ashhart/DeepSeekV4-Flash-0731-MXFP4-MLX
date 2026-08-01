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

> **On methodology.** Decode figures below are measured by *slope* — generate
> N1 and N2 tokens from the same prompt and take `(N2-N1)/(t2-t1)` — so prefill
> and per-request overhead cancel. Earlier revisions of this README timed
> `max_tokens=1` and subtracted it from `max_tokens=200`; at long context
> prefill dominates and varies run to run, so that method produced 34.4 and 77.4
> tok/s from *identical* requests whose total time was ~6.9s both times. Those
> numbers were withdrawn. Do not trust a decode figure that was not taken by
> slope or by an equivalent prefill-independent method.

### The biggest win is a config value, not a kernel

oMLX's prompt cache ships **disabled**: `hot_cache_max_size` defaults to `"0"`,
which the config comments mark as off. Combined with **Hot Cache Only** (RAM
only, no SSD spill) that means nothing is cached and every turn re-prefills the
whole prompt.

It is easy to miss because the admin UI's **CACHE** panel shows only *Cache
Enabled*, *Hot Cache Only* and *SSD Cache Directory* — all of which look
correctly configured. The size lives in a different section, **Memory
Management → "Memory Limit (In-Memory Hot Cache)"**.

Setting it (e.g. 24GB, in `~/.omlx/settings.json` or that UI field), on a
repeated 23.7K-token prompt:

| run | prompt tokens | cached | wall |
|---|---|---|---|
| 1 | 23,716 | 0 | **51.0s** |
| 2 | 23,716 | 23,552 | **4.6s** |

**~11x on the turn.** For agent traffic that resends a growing context every
turn, prefill is the overwhelming majority of wall-clock time and this dominates
every other optimisation here. A full agent turn (23.6K cached prompt + 200
tokens) went from ~59s to **~6.9s**.

### Decode

**2026-08-01 production stack** — speculation + batched rollback bookkeeping +
fused router + 8-bit `lm_head` + fused Q/KV norm/RoPE + async cache
materialisation. Measured by slope through the API:

| | spec off | production stack |
|---|---|---|
| 23K cached context | 26.1 | **40.7 tok/s** (+56%) |
| short prompt | ~29 | **50.1 tok/s** |

Phase attribution: short 55.5 ms/cycle vs 23K 63.1 — **97% of the context cost
is the target forward's attention** (indexer top-512 over long context +
pooled-KV reads); the drafter is context-immune by design. Next levers for long
context: the sparse-attention/indexer path, then a fused L=4 MoE.

#### Determinism (read before comparing hashes)

Temp-0 warm runs are **not bit-reproducible** here, and an exhaustive ladder
(every feature isolated and in production combination, degraded and healthy
machines, byte-reverting recent edits) found no configuration that is. The
divergence matches the documented 1-ULP × top-k amplification; the source sits
below the application. Gate: same-class cache offset-integrity (zero tolerance)
+ logit tolerance in the measured envelope + output-quality sweeps. Everything
exactness-testable in isolation passes: rollback 16/16, router sets 244/244,
GPU-accept 244/244.

#### Fused router (`ds4/router_fused.py`)

One Metal kernel replaces sqrtsoftplus→bias→argpartition(256)→gather→normalize.
Unit-proven 244/244; **measured in-pipeline −1.2 ms/cycle** — far below the
isolated ledger's 5.13 ms, which mostly overlaps away in the real pipeline.
Promoted: consistent, zero-risk. Harness lesson: levers must prove ENGAGEMENT
in-log or the A/B voids itself. `ds4/gpu_accept.py` (244/244) ships shelved —
the engine already single-barriers per-cycle readbacks, so its projected win
evaporated on reading the code it targeted.

#### For comparison: dual DGX Spark

A published dual-Spark benchmark of the same model reports **72.8 tok/s
single-stream** — genuinely faster, on two nodes with TP=2 halving per-node
weight traffic. The rest of that run is worth reading though:

| concurrency | aggregate | per-stream | TTFT |
|---|---|---|---|
| x1 | 72.8 | 72.8 | 237ms |
| x2 | 102.0 | 53.1 | 1.55s |
| x4 | 120.0 | 35.4 | 7.68s |
| x8 | 147.0 | 23.8 | 4.91s |

Per-stream falls to 35.4 at x4 and 23.8 at x8, so at real concurrency a single
M3 Ultra is level or ahead — and its cached prefill at 23K is ~1.2s against
their 7.68s TTFT at x4. Single-stream x1 is where two nodes win.

### Draft block size

Relative comparison, hook off vs hook on in the same process (short prompt):

| verify k | speedup | accepted prefix | clamped | restore+replay |
|---|---|---|---|---|
| 2 | 1.28x | 1.49/2 | 0% | 0% |
| **3** | **1.41-1.43x** | **2.01/3** | **0%** | **0%** |
| 5 | 0.94-1.09x | 2.41/5 | 26% | 21% |
| 7 | 0.49x | 1.65/7 | 50% | 32% |

**Draft width and verify width are separate knobs** (`DS4_SPEC_DRAFT_WIDTH` vs
`DS4_SPEC_BLOCK`). The drafter's block is non-causal — every draft position
attends to every other — so its width is part of its input distribution, not a
free parameter. It stays at the trained value while only a prefix is verified.

Verifying wider is worse, for two compounding reasons:

1. Padding the draft block past the trained width puts out-of-distribution noise
   positions in view of the real ones, so the accepted prefix actually *falls*
   (2.41 at k=5 -> 1.65 at k=7) despite drafting more tokens.
2. Above k=3 the `PoolingCache` rollback starts refusing, and each refusal costs
   a full cache restore plus a replay forward.

> Earlier revisions claimed 47.1 / 1.72x, and briefly 2.06x at k=7. **Withdrawn**
> — measured against a corrupt KV cache (see *The rollback bug*), whose output
> was fluent repetition.

### `lm_head` quantization

The checkpoint leaves the output head unquantized (`"head": false`), so it is
129280x4096 bf16 = **1.06 GB read per forward**, and speculation reads it twice
per cycle. Quantizing it halves that. Quality measured teacher-forced over 2047
positions against the untouched bf16 head:

| `lm_head` | bytes | perplexity | vs bf16 | top-1 agree | KL (nats) |
|---|---|---|---|---|---|
| bf16 (stock) | 1.06 GB | 8.3103 | — | — | — |
| **8-bit g64** | **0.56 GB** | **8.3019** | **-0.10%** | **98.78%** | 0.00073 |
| 8-bit g32 | 0.60 GB | 8.2909 | -0.23% | 98.93% | 0.00072 |
| 6-bit g64 | 0.43 GB | 8.3088 | -0.02% | 97.07% | 0.00256 |
| 4-bit g64 | 0.30 GB | 8.5392 | **+2.75%** | 91.26% | 0.02625 |

8-bit is effectively lossless (the -0.10% is noise, not an improvement). **4-bit
is clearly degraded — do not use it.** 6-bit moves fewer bytes than 8-bit and
measured *slower*; MLX's 8-bit quantized matmul is better optimised.

```sh
echo 8 > ~/.omlx/ds4_head_bits     # or DS4_QUANT_HEAD=8
```

### Prefill

> **The previously published 938 tok/s figure is withdrawn.** It was measured on
> the same three-line function repeated 800 times, which collapses MoE routing
> onto a handful of experts and is not representative. On real content, prefill
> at 25K measures **~430 tok/s** through the server.

The windowed-prefill kernel below removes a genuine `O(L^2)` term — the model
builds a dense `(L, L)` mask despite a 128-token sliding window. The A/B ratios
were measured on **random tokens**, which is the worst case for the dense path
(every MoE expert activated), so treat the ratios as an upper bound and the
shape as the real finding:

| L | dense | windowed |
|---|---|---|
| 4096 | 425 | 536 |
| 8192 | 345 | 603 |
| 16384 | 183 | 595 |

The dense path halves from 8K to 16K; the windowed path does not move. The
equivalent A/B on real text has **not** been measured.

With a working prompt cache, prefill is paid once per unique prefix rather than
once per turn, which matters far more than either number above.

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
{
  "cache":     { "hot_cache_max_size": "24GB" },
  "scheduler": { "chunked_prefill": false }
}
```

**`hot_cache_max_size` is the important one** — it ships as `"0"`, which means
*disabled*, so nothing is cached and every request re-prefills its whole prompt.
See *The biggest win* above. Size it to taste; 24GB leaves plenty of headroom
next to a 146 GiB resident model on a 256 GB machine. The same value is editable
in the admin UI under **Memory Management → Memory Limit (In-Memory Hot Cache)**
— note it is *not* in the CACHE panel, which is why it is easy to miss.

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

# Restart. NOTE the lowercase pattern: the server process is `omlx-server`,
# and `pkill -f "oMLX"` is case-sensitive so it will NOT kill it -- `omlx start`
# then finds port 8000 in use and silently keeps the OLD code running.
pkill -9 -f 'omlx-server'; pkill -9 -f 'oMLX'; sleep 5
~/.omlx/bin/omlx start          # or launch oMLX.app from the Dock
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

Rejected drafts must come back out of the KV cache. Getting it wrong does not
crash — it produces fluent repetition while throughput *looks* fine, because a
rejected draft is by construction a plausible continuation. Four distinct traps,
all hit here:

1. **`CacheList` is not a `list` subclass** — it wraps `.caches`. An
   `isinstance(x, (list, tuple))` check silently skips every compressed layer.

2. **`CacheList.trim` masks refusals.** It is:

   ```python
   def trim(self, n):
       for c in self.caches:
           m = c.trim(n)
       return m          # only the LAST sub-cache's result
   ```

   Each layer holds `[RotatingKVCache, PoolingCache]`. If the rotating one
   refuses (returns 0) while the pooling one succeeds, the wrapper reports
   success and the rotating cache never gets trimmed. **Trim the concrete
   caches, never the wrapper.**

3. **Snapshots must be detached.** `BatchRotatingKVCache.offset` is an
   `mx.array` *mutated in place*, so a plain reference reads back the
   post-update value and the snapshot is a silent no-op. oMLX's own
   `cache_rollback` does `v = v + 0` for exactly this reason.

4. **There are TWO rotating cache classes, and their `offset` fields are not
   comparable.** Layers 0-1 use `PrefillReadyRotatingKVCache`; layers 2-42 use
   `BatchRotatingKVCache` (batched, with `_offset` and `left_padding`). They sit
   a constant 12 apart, forever, with perfectly clean output. An integrity check
   that compares *across* classes will declare corruption that is not there —
   this cost ~20% throughput before it was understood. **Compare within a class;
   same-class layers drifting apart is the real corruption signal.**

`tools/batch_cache_rollback.py` reproduces the rollback for both classes with no
model load (16/16 land exactly where a plain decode would).
`tools/cache_invariants.py` and `tools/pooling_rollback_test.py` cover the
`RotatingKVCache` and `PoolingCache` state machines the same way.

Rollback itself goes through oMLX's `cache_rollback` armed undo log
(`set_undo_armed`, covering updates of 2..8 tokens). Do not hand-roll it.

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
tools/cache_check.py      does the prompt cache actually engage?
tools/decode_rate.py      decode by slope (prefill-independent)
tools/agent_turn.py       end-to-end timing of a realistic agent turn
tools/head_quality.py     perplexity/top-1/KL cost of quantizing lm_head
tools/cache_invariants.py RotatingKVCache trim behaviour, no model load
tools/pooling_rollback_test.py  PoolingCache rollback exactness, no model load
```

## Credits

Model conversion: [Vontra](https://huggingface.co/Vontra). Base model:
[deepseek-ai/DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731).
DSpark ported from the reference `inference/model.py` shipped in the checkpoint.

MIT.
