# SPDX-License-Identifier: MIT
"""Wire DSpark speculative decoding into oMLX's generation loop.

Hook point
----------
`mlx_lm.generate.GenerationBatch._step` feeds one token and returns one token,
pipelined by one: the token it *returns* is the token it just fed into the KV
cache, while the next one is sampled asynchronously. So `self.tokens` always
mirrors the cache exactly.

That maps cleanly onto speculation. One DSpark cycle feeds `[t, d1..dk]` and
keeps `[t, d1..dn]` after rollback, so it can emit `n+1` tokens. We queue them
and let `_step` pop one per call; the surrounding machinery (stop sequences,
max_tokens, filtering, streaming) is untouched.

Correctness
-----------
Acceptance is **match-based**: sample `s_i` from the target's own logprobs with
the batch's own sampler, accept `d_{i+1}` only if `s_i == d_{i+1}`, otherwise
emit `s_i` and end the cycle. Either way the emitted token is drawn from the
target distribution conditioned on the accepted prefix, so the output
distribution is exact **for any sampler** — greedy or not. No `p_target/p_draft`
ratio is needed (that rule buys higher acceptance, not correctness).

A degraded drafter therefore costs throughput, never correctness: the verify
pass is always authoritative. That is what makes this safe to switch on.

Guards
------
Engages only for a single-sequence batch with no logits processors, on a
deepseek_v4 model whose drafter loaded. Anything else falls straight through to
the stock path. Off unless DS4_SPEC=1.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any, Optional

import mlx.core as mx

_PATCHED = False
_DRAFTER: Optional[Any] = None
_DRAFTER_FAILED = False


class _NotReady(Exception):
    """Raised before any model call, so falling back is safe."""


# Cycle accounting, so a disappointing speedup can be attributed: low draft
# acceptance is a different problem from rollback clamping it away.
STATS = {"cycles": 0, "raw_accepted": 0, "clamped_accepted": 0, "clamp_hits": 0,
         "restore_replay": 0}


def stats_summary() -> str:
    c = STATS["cycles"]
    if not c:
        return "no speculative cycles"
    return (
        f"cycles={c} "
        f"raw_accept={STATS['raw_accepted'] / c:.2f} "
        f"after_clamp={STATS['clamped_accepted'] / c:.2f} "
        f"tokens/cycle={(STATS['clamped_accepted'] + c) / c:.2f} "
        f"clamped_on={STATS['clamp_hits'] / c * 100:.0f}% "
        f"restore_replay={STATS['restore_replay'] / c * 100:.0f}%"
    )


_logger = logging.getLogger("omlx.ds4")


def _log(msg: str) -> None:
    # Both: stderr for CLI runs, and the logging module so messages reach
    # ~/.omlx/logs/server.log when oMLX is launched from the GUI (whose stderr
    # goes nowhere visible).
    print(f"[ds4] {msg}", file=sys.stderr)
    try:
        _logger.info(msg)
    except Exception:  # noqa: BLE001
        pass


def enabled() -> bool:
    """On when DS4_SPEC=1, or when the marker file exists.

    The file matters because oMLX is normally launched from the GUI, which does
    not inherit a shell's environment. Deliberately NOT gated on oMLX's own
    `mtp_enabled`: that flag makes `is_mtp_active()` true, which builds oMLX's
    V3-shaped MTP heads and fails the weight load outright (3140 unmatched
    tensors) on this DSpark checkpoint.

        enable :  touch ~/.omlx/ds4_spec_enabled   (then restart oMLX)
        disable:  rm    ~/.omlx/ds4_spec_enabled
    """
    if os.environ.get("DS4_SPEC") == "1":
        return True
    if os.environ.get("DS4_SPEC") == "0":
        return False
    return (Path.home() / ".omlx" / "ds4_spec_enabled").exists()


def _block_size() -> int:
    """Draft tokens per cycle. Measured on M3 Ultra, 200 tokens, after the
    rollback was fixed (earlier numbers were taken against a corrupt cache and
    were meaningless -- see `_snapshot`):

        k=2  1.28x   raw_accept 1.49/2   clamped  0%   restore  0%
        k=3  1.43x   raw_accept 2.01/3   clamped  0%   restore  0%   <- default
        k=5  1.09x   raw_accept 2.41/5   clamped 25%   restore 20%
        k=7  0.49x   raw_accept 1.65/7   clamped 50%   restore 32%

    Two effects push the optimum down to 3:

    1. The drafter is trained at `dspark_block_size` = 5, and its block is
       *non-causal* -- every draft position attends to every other. Padding past
       the trained width puts out-of-distribution noise positions in view of all
       the real ones, so accepted-prefix actually FALLS (2.41 at k=5 -> 1.65 at
       k=7) even though there are more drafts to accept.
    2. Above k=3 the PoolingCache rollback starts refusing, and each refusal
       costs a full restore plus a replay forward. At k=5 that is a fifth of all
       cycles.
    """
    try:
        return int(os.environ.get("DS4_SPEC_BLOCK", "3"))
    except ValueError:
        return 3


def _draft_width(trained: int) -> int:
    """How many tokens the drafter is asked to produce.

    Deliberately separate from `_block_size()`, which is how many of them we
    actually verify. The drafter's block is non-causal -- every draft position
    attends to every other -- so its width is part of its input distribution,
    not a free knob. Running it at its trained `dspark_block_size` and then
    verifying a prefix keeps it in distribution while keeping the verify narrow
    enough that the PoolingCache rollback never refuses.

    DS4_SPEC_DRAFT_WIDTH overrides; default is the checkpoint's trained width.
    """
    try:
        return max(1, int(os.environ.get("DS4_SPEC_DRAFT_WIDTH", trained)))
    except ValueError:
        return trained


def _get_drafter(model) -> Optional[Any]:
    """Build the DSpark drafter once, lazily, from the model's own directory."""
    global _DRAFTER, _DRAFTER_FAILED
    if _DRAFTER is not None or _DRAFTER_FAILED:
        return _DRAFTER

    path = getattr(model, "_ds4_model_path", None)
    if path is None:
        _DRAFTER_FAILED = True
        _log("no model path recorded; speculative decoding disabled")
        return None

    try:
        import json

        from ds4.load_dspark import build_drafter

        config = json.loads((Path(path) / "config.json").read_text())

        # _apply_hidden_capture assumes the target layers are the last three.
        targets = config.get("dspark_target_layer_ids")
        n_layers = config.get("num_hidden_layers")
        if targets != list(range(n_layers - 3, n_layers)):
            _DRAFTER_FAILED = True
            _log(
                f"dspark_target_layer_ids {targets} are not the last 3 of "
                f"{n_layers}; hidden capture would be wrong. Disabled."
            )
            return None

        drafter, weights, dspark = build_drafter(Path(path), config, model.args)
        drafter.load_weights(list(weights.items()), strict=True)
        mx.eval(drafter.parameters())
        del weights
        drafter.block_size = _draft_width(dspark["dspark_block_size"])
        _DRAFTER = drafter
        _log(
            f"DSpark drafter loaded ({dspark['n_mtp_layers']} stages, "
            f"draft_width={drafter.block_size}, "
            f"verify_width={min(_block_size(), drafter.block_size)})"
        )
    except Exception as e:  # noqa: BLE001
        _DRAFTER_FAILED = True
        _log(f"DSpark drafter NOT loaded: {type(e).__name__}: {e}")
    return _DRAFTER


_REPORTED = set()


def _eligible(gb) -> bool:
    if not enabled():
        return False

    model = gb.model
    checks = {
        "single_sequence": len(getattr(gb, "uids", []) or []) == 1,
        "no_logits_processors": not any(getattr(gb, "logits_processors", None) or []),
        "is_deepseek_v4": getattr(model, "model_type", None) == "deepseek_v4",
        # Rollback goes through oMLX's own helpers; without them there is no
        # sound way to undo a rejected draft, so stay on the stock path.
        "has_main_hidden": getattr(model.model, "main_hidden", None) is not None,
    }
    bad = [k for k, v in checks.items() if not v]

    # oMLX's MTP patch is self-healing: it reinstalls its own
    # DeepseekV4Model.__call__ whenever it sees ours, and in the server that
    # happens during model load, i.e. after our boot hook has run. So heal back:
    # if the capture is the only thing missing, re-wrap now and pick it up on
    # the next forward. Costs one non-speculative step, then engages.
    if bad == ["has_main_hidden"]:
        _apply_hidden_capture()
        return False

    if bad:
        key = tuple(bad)
        if key not in _REPORTED:
            _REPORTED.add(key)
            _log(f"speculation not engaging, failed: {', '.join(bad)}")
        return False
    return True


def _trim_rotating(c, n: int) -> None:
    """Drop the last `n` entries after a multi-token update.

    `is_trimmable()` is False once `offset >= max_size` because the *in-place*
    (S==1) path leaves the ring out of temporal order. A verify has S>1, which
    always takes `_update_concat` -- that calls `_temporal_order` first, then
    rebinds to fresh arrays with the new entries last. So right after a verify
    the buffer is ordered and the rejected tail can simply be sliced off.
    """
    if n <= 0 or getattr(c, "keys", None) is None:
        return
    c.keys = c.keys[..., :-n, :]
    c.values = c.values[..., :-n, :]
    c.offset -= n
    c._idx = c.keys.shape[2]


def _check_offsets(cache, expected: int, tag: str) -> None:
    """Assert every rotating cache agrees on how many tokens are committed.

    Layer desync is the mechanism behind the repetition garble: if one layer
    holds more tokens than another, the attention mask is built for one length
    and the keys are another, and the model decodes against phantom context.
    Enable with DS4_SPEC_DEBUG=1.
    """
    if os.environ.get("DS4_SPEC_DEBUG") != "1":
        return
    seen = {}
    for i, layer_cache in enumerate(cache):
        for c in _unwrap(layer_cache):
            if type(c).__name__.endswith("RotatingKVCache"):
                seen.setdefault(int(c.offset), []).append(i)
    if len(seen) > 1 or (seen and expected not in seen):
        _log(
            f"CACHE DESYNC after {tag}: expected offset {expected}, "
            f"found {[(off, f'{len(ls)} layers') for off, ls in seen.items()]}"
        )


def _unwrap(layer_cache) -> list:
    """Flatten a layer's cache into its concrete cache objects.

    `CacheList` is NOT a list subclass -- it wraps `.caches` -- so an
    `isinstance(x, (list, tuple))` check silently misses it and leaves the real
    caches untouched. Its own `trim()` delegates to `RotatingKVCache.trim()`,
    which no-ops once the ring has rotated (`is_trimmable()` is False), so
    rejected drafts would survive as phantom tokens on 41 of 43 layers.
    """
    if layer_cache is None:
        return []
    caches = getattr(layer_cache, "caches", None)
    if caches is not None:
        return [c for c in caches if c is not None]
    if isinstance(layer_cache, (list, tuple)):
        return [c for c in layer_cache if c is not None]
    return [layer_cache]


def _set_armed(flag: bool) -> None:
    """Arm/disarm oMLX's rotating-cache undo stash around a verify forward."""
    try:
        from omlx.patches.mlx_lm_mtp.cache_rollback import set_undo_armed

        set_undo_armed(flag)
    except Exception:  # noqa: BLE001 -- absence just means no undo coverage
        pass


def _ensure_rollback_patch() -> None:
    """Attach oMLX's rotating-cache undo log.

    NOTE: deliberately does NOT apply oMLX's MTP *model* patch. That patch
    replaces `DeepseekV4Model.__call__` and `make_cache`, and measured **-21% on
    prefill** at 25K context (396 -> 311 tok/s). For prompt-heavy agent traffic
    prefill dominates -- a 25K prompt is ~58s of prefill against ~2.5s of decode
    -- so paying 21% there to win 1.5x on decode is a net loss. The three
    helpers we actually needed from it (`_can_trim`, `clamp_accept`,
    `partial_rollback`) are reimplemented below instead.

    `cache_rollback` itself is cheap: it only wraps `update_and_fetch` with a
    stash that is skipped unless armed.
    """
    try:
        from omlx.patches.mlx_lm_mtp import cache_rollback

        cache_rollback.apply()
    except Exception as e:  # noqa: BLE001
        _log(f"cache_rollback not available: {type(e).__name__}: {e}")


def _can_trim(c, n: int) -> bool:
    """Non-mutating check that `c.trim(n)` will succeed.

    Mirrors oMLX's `_cache_can_trim`. Must be checked for every layer *before*
    trimming any of them: a trim that fails partway leaves per-layer lengths
    desynchronised, which is the phantom-token corruption.
    """
    caches = getattr(c, "caches", None)
    if caches is not None:
        return all(_can_trim(sub, n) for sub in caches)

    remainder = getattr(c, "remainder", None)
    if remainder is not None:  # PoolingCache / BatchPoolingCache
        rem_min = remainder if isinstance(remainder, int) else min(remainder)
        if n <= rem_min:
            return True
        can_undo = getattr(c, "_can_undo", None)
        return bool(can_undo and can_undo(n))

    is_trimmable = getattr(c, "is_trimmable", None)
    if callable(is_trimmable) and is_trimmable():
        return True
    # A rotated RotatingKVCache is only trimmable via cache_rollback's armed
    # undo stash, which covers the last multi-token write.
    undo = getattr(c, "_mtp_undo", None)
    if undo is not None:
        return undo[1].shape[2] >= n
    return False


def clamp_accept(cache, accepted: int, num_drafts: int) -> int:
    """Largest m <= accepted whose rollback every layer supports.

    Emitting fewer verified drafts than acceptance allowed is always correct --
    the skipped ones are re-derived next cycle -- so this keeps the cycle alive
    when a PoolingCache cannot replay a longer confirmed prefix.
    """
    for m in range(accepted, -1, -1):
        n = num_drafts - m
        if n <= 0 or all(_can_trim(c, n) for c in cache):
            return m
    return 0


def partial_rollback(cache, accepted: int, num_drafts: int) -> bool:
    """Trim the verify window back to `accepted` drafts on every layer."""
    n = num_drafts - accepted
    if n <= 0:
        return True
    if not all(_can_trim(c, n) for c in cache):
        return False
    for c in cache:
        if c.trim(n) != n:
            return False
    return True


def _snapshot(cache) -> list:
    """Capture enough state to undo a verify forward exactly.

    A multi-token update takes the *concat* path on both cache types, which
    rebinds `keys`/`values`/`pooled` to fresh arrays rather than mutating them,
    so holding the old references is a valid snapshot. `buf_kv`/`buf_gate` are
    the exception -- `accumulate_windows` writes into them in place -- so those
    get a real copy. They are only (B, ratio, D), so it is cheap.
    """
    snaps = []
    for layer_cache in cache:
        for c in _unwrap(layer_cache):
            state = {}
            for key, value in vars(c).items():
                if isinstance(value, mx.array):
                    # DETACH. Several caches mutate arrays in place -- notably
                    # BatchRotatingKVCache, whose `offset` is an mx.array, and
                    # PoolingCache's `buf_kv` via setitem. A plain reference
                    # would read back the post-update value, making the whole
                    # snapshot a silent no-op. `+ 0` is lazy, so this costs a
                    # graph node, not a copy, unless it is actually used.
                    value = value + 0
                state[key] = value
            snaps.append((c, state))
    return snaps


def _restore(snaps) -> None:
    """Put every cache back exactly as it was before the verify forward.

    Snapshots the full instance dict rather than a hand-listed set of fields:
    the cache classes differ in what they carry (`RotatingKVCache` has
    keys/values/offset/_idx; `BatchRotatingKVCache` adds _offset/rotated/
    left_padding; `PoolingCache` has pooled/remainder/buf_*/prev_win_*), and
    missing one silently corrupts the rollback.
    """
    for c, state in snaps:
        for key, value in state.items():
            setattr(c, key, value)


def _rollback(cache, n: int) -> bool:
    """Trim `n` rejected positions. False means nothing was changed for some
    cache and the caller must restore from a snapshot instead.

    `PoolingCache.trim` refuses (returns 0, without mutating) when the replayed
    prefix would complete a pool window, because it discards what
    `accumulate_windows` hands back -- the window the Compressor still has to
    compress into `pooled`. Measured: 36/40 rollbacks trim exactly, the other 4
    are window-boundary crossings. Detect and fall back rather than corrupt.
    """
    if n <= 0:
        return True
    pending = []
    for layer_cache in cache:
        for c in _unwrap(layer_cache):
            name = type(c).__name__
            if name.endswith("RotatingKVCache"):
                pending.append(("rot", c))
            elif hasattr(c, "trim"):
                if not c.is_trimmable() or not c._can_undo(n):
                    return False  # refuse before touching anything
                pending.append(("other", c))
            else:
                return False

    for kind, c in pending:
        if kind == "rot":
            _trim_rotating(c, n)
        elif c.trim(n) != n:
            return False
    return True


def _apply_hidden_capture(n_last: int = 3) -> None:
    """Make DeepseekV4Model stash the hidden states DSpark fuses.

    The reference takes `h.mean(dim=2)` (the mean over the hc_mult
    Hyper-Connection copies) at each of `dspark_target_layer_ids` and
    concatenates along the feature axis. For this checkpoint those ids are
    [40, 41, 42] -- the last three of 43 -- which is what `n_last` encodes;
    `_get_drafter` re-checks that against config.json and disables itself if it
    ever stops holding.
    """
    import mlx_lm.models.deepseek_v4 as dsv4

    cls = dsv4.DeepseekV4Model
    # Check the *function*, not a class flag. oMLX's MTP patch is self-healing:
    # it treats our wrapper as drift and reinstalls its own __call__, so a
    # class-level "already patched" flag would stop us ever re-wrapping.
    if getattr(cls.__call__, "_ds4_capture", False):
        return
    original = cls.__call__

    # *args/**kwargs, because oMLX's MTP patch gives this a wider signature
    # (return_hidden, n_confirmed) than stock. Apply this AFTER that patch so we
    # wrap it rather than replace it.
    def __call__(self, inputs, *args, **kwargs):
        targets = set(
            range(self.args.num_hidden_layers - n_last, self.args.num_hidden_layers)
        )
        captured = []
        layer_cls = type(self.layers[0])
        inner = layer_cls.__call__
        index_of = {id(layer): i for i, layer in enumerate(self.layers)}

        def wrapper(layer_self, *a, **kw):
            out = inner(layer_self, *a, **kw)
            if index_of.get(id(layer_self)) in targets:
                captured.append(out.mean(axis=2))
            return out

        layer_cls.__call__ = wrapper
        try:
            out = original(self, inputs, *args, **kwargs)
        finally:
            layer_cls.__call__ = inner

        if len(captured) == n_last:
            self.main_hidden = mx.concatenate(captured, axis=-1)
            # Keep the prompt's hidden separately. The drafter's window holds up
            # to `sliding_window` main-model positions and is what gives its
            # drafts context; seeding it from a single decode position instead
            # of the whole prompt costs a lot of acceptance until it refills one
            # token at a time. A verify is at most 8 wide, so anything wider is
            # a prompt.
            if inputs.shape[1] > 16:
                self.main_hidden_prefill = self.main_hidden
        else:
            self.main_hidden = None
            if not getattr(type(self), "_ds4_capture_warned", False):
                type(self)._ds4_capture_warned = True
                _log(
                    f"hidden capture got {len(captured)} of {n_last} target layers "
                    f"(num_hidden_layers={self.args.num_hidden_layers})"
                )
        return out

    __call__._ds4_capture = True
    cls.__call__ = __call__
    cls._ds4_hidden_patched = True


def _maybe_quantize_head(model) -> None:
    """Optionally quantize `lm_head` in place after load. Off by default.

    The checkpoint deliberately leaves `head` unquantized (`config.json` has
    `"head": false`), so it is 129280 x 4096 in bf16 = **1.06 GB read per
    forward** -- about 10% of everything a decode step moves, and speculation
    reads it TWICE per cycle (once in the drafter, once in the verify), so ~17%
    of the cycle's bytes.

    Quantizing it to 8-bit halves that. It is not free: the logits change
    slightly, which can flip an argmax on a near-tie. Opt-in, and the bit width
    is explicit:

        DS4_QUANT_HEAD=8    # or 6, or 4
    """
    bits = os.environ.get("DS4_QUANT_HEAD")
    if not bits:
        # File fallback: oMLX is launched from the GUI, which inherits no shell
        # environment.  echo 8 > ~/.omlx/ds4_head_bits
        marker = Path.home() / ".omlx" / "ds4_head_bits"
        if marker.exists():
            bits = marker.read_text().strip()
    if not bits or str(bits).strip() in ("0", "off", "none", "bf16"):
        return  # explicit off, for A/B measurement
    try:
        bits = int(bits)
        import mlx.nn as nn

        head = getattr(model, "lm_head", None)
        if head is None or isinstance(head, nn.QuantizedLinear):
            return
        before = head.weight.size * head.weight.dtype.size
        model.lm_head = nn.QuantizedLinear.from_linear(head, group_size=64, bits=bits)
        after = sum(
            v.size * v.dtype.size
            for v in (model.lm_head.weight, model.lm_head.scales, model.lm_head.biases)
        )
        mx.eval(model.lm_head.parameters())
        _log(
            f"lm_head quantized to {bits}-bit: "
            f"{before / 1e9:.2f} GB -> {after / 1e9:.2f} GB per read"
        )
    except Exception as e:  # noqa: BLE001
        _log(f"lm_head quantization skipped: {type(e).__name__}: {e}")


def _apply_path_capture() -> None:
    """Record each model's directory so the drafter can find its `mtp.*` weights."""
    import mlx_lm.utils as _utils

    original = _utils.load_model
    if getattr(original, "_ds4_path_wrapped", False):
        return

    def wrapped(model_path, *a, **kw):
        model, config = original(model_path, *a, **kw)
        try:
            model._ds4_model_path = str(model_path)
        except Exception:  # noqa: BLE001
            pass
        _maybe_quantize_head(model)
        return model, config

    wrapped._ds4_path_wrapped = True
    _utils.load_model = wrapped
    for name, mod in list(sys.modules.items()):
        if name.startswith("mlx_lm") and name != "mlx_lm.utils":
            if getattr(mod, "load_model", None) is original:
                try:
                    mod.load_model = wrapped
                except Exception:  # noqa: BLE001
                    pass


def apply() -> bool:
    """Patch `GenerationBatch._step` with the speculative path.

    The class-level patches run on EVERY call, not just the first: oMLX
    re-registers `mlx_lm.models.deepseek_v4` from source on each model load, so
    the classes are fresh objects and any patch applied to the previous ones is
    gone. Each helper carries its own per-class idempotency guard, so repeating
    them is free. Only the `GenerationBatch._step` patch is once-only —
    `mlx_lm.generate` is never re-registered.
    """
    global _PATCHED

    # Order matters: the MTP patch replaces DeepseekV4Model.__call__, so install
    # it first and let the hidden capture wrap the result.
    _ensure_rollback_patch()
    _apply_hidden_capture()
    _apply_path_capture()

    if _PATCHED:
        return False

    from mlx_lm.generate import GenerationBatch

    original = GenerationBatch._step

    def _spec_step(self):
        queue = getattr(self, "_ds4_queue", None)
        if queue:
            tok, lp = queue.pop(0)
            self.tokens[0].append(tok)
            return [tok], [lp]

        if not _eligible(self):
            return original(self)

        drafter = _get_drafter(self.model)
        if drafter is None:
            return original(self)

        global _DRAFTER_FAILED
        try:
            return _cycle(self, drafter)
        except _NotReady:
            # Raised before any model call -- the cache is untouched, so the
            # stock path can take over cleanly. Expected on the first step.
            return original(self)
        except Exception as e:  # noqa: BLE001 -- never break generation
            # Past the verify forward the cache has already advanced, so this
            # fallback is best-effort only. Disable speculation rather than
            # risk decoding against inconsistent state on every later step.
            _log(
                f"DSpark cycle failed after the forward ({type(e).__name__}: {e}); "
                "speculation disabled for this process"
            )
            _DRAFTER_FAILED = True
            self._ds4_queue = []
            self._ds4_window = None
            return original(self)

    GenerationBatch._step = _spec_step
    _PATCHED = True
    return True


def _cycle(self, drafter):
    """One draft -> verify -> accept -> rollback cycle. Returns the first token."""
    model = self.model
    # k is the VERIFY width -- how many drafts we check and may commit.
    # drafter.block_size is the DRAFT width, kept at the trained value so the
    # non-causal draft block stays in distribution. They are decoupled on
    # purpose; see `_draft_width`.
    k = min(_block_size(), drafter.block_size)
    cache = self.prompt_cache

    # Everything that can fail must happen BEFORE the verify forward: once the
    # model has run, the cache has advanced and falling back to the stock path
    # would decode against inconsistent state.
    tok_arr = self._next_tokens
    if tok_arr is None or not self._next_logprobs:
        raise _NotReady("pipeline not primed yet")
    lp_prev = self._next_logprobs[0]
    mx.eval(tok_arr)
    t = int(tok_arr[0])

    offset = int(cache[0][0].offset if isinstance(cache[0], (list, tuple)) else cache[0].offset)
    pos = offset - 1  # absolute position of the last token already in the cache

    window = getattr(self, "_ds4_window", None)
    if window is None:
        # Seed from the *prompt's* hidden states where possible -- the drafter
        # needs main-model context to draft well, and the last forward before
        # the first cycle is a single decode step. A short window only costs
        # acceptance, never correctness.
        window = drafter.new_window(1)
        mh = getattr(model.model, "main_hidden_prefill", None)
        if mh is None:
            mh = model.model.main_hidden
        mh = mh[:, -drafter.window_size :]
        start = max(0, offset - mh.shape[1])
        window = drafter.push_window(window, mh, start)
        if os.environ.get("DS4_SPEC_DEBUG") == "1":
            _log(f"window seeded with {mh.shape[1]} positions from offset {start}")

    draft_ids, _conf = drafter(
        mx.array([[t]]),
        model.model.embed_tokens,
        model.lm_head,
        window,
        pos,
    )
    mx.eval(draft_ids)
    # The drafter may have produced more than we intend to verify (see
    # `_draft_width`); verify a prefix of them.
    drafts = [int(x) for x in draft_ids[0].tolist()][:k]

    # Verify [t, d1..dk] in one forward, with oMLX's undo log armed.
    #
    # cache_rollback wraps RotatingKVCache.update_and_fetch to stash a DETACHED
    # snapshot (`v + 0` -- a plain reference would see the post-update value,
    # which is the trap I fell into rolling my own) for any armed update of
    # 2..8 tokens, and makes trim() replay the confirmed prefix from it. It is
    # only armed around a verify forward, so stock decode keeps stock semantics.
    # Snapshot before the forward. `mtp_clamp_accept` can return an accepted
    # count that `mtp_partial_rollback` then refuses, and by that point the
    # verify has already advanced the cache -- which is what turns output into
    # repetition. This gives an unconditional way back.
    snaps = _snapshot(cache)
    _check_offsets(cache, offset, "pre-verify")

    _set_armed(True)
    try:
        logits = model(cand_arr := mx.array([[t] + drafts]), cache=cache)[0]
    finally:
        _set_armed(False)
    del cand_arr
    logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)

    sampler = (self.samplers and self.samplers[0]) or self.fallback_sampler
    samples = [int(sampler(logprobs[i : i + 1])[0]) for i in range(k + 1)]
    mx.eval(logprobs)

    # Longest prefix where the target's own sample reproduces the draft.
    n = 0
    while n < k and samples[n] == drafts[n]:
        n += 1

    # Not every accepted length is rollback-able: a PoolingCache can only replay
    # a confirmed prefix that stays inside its window. `mtp_clamp_accept` finds
    # the largest m <= n that every layer can undo. Emitting fewer drafts than
    # were verified is always correct -- the rest are re-derived next cycle.
    n_raw = n
    n = int(clamp_accept(cache, n, k))
    STATS["cycles"] += 1
    STATS["raw_accepted"] += n_raw
    STATS["clamped_accepted"] += n
    STATS["clamp_hits"] += int(n != n_raw)

    if n != n_raw and STATS["clamp_hits"] == 1:
        # Leading indicator of the unsafe regime: once the clamp starts firing,
        # `mtp_partial_rollback` can still refuse the value the clamp returned,
        # and that refusal lands AFTER the verify forward -- rejected drafts
        # stay in the cache and output degenerates into repetition. Measured
        # clean at block_size <= 3; garbled at 7.
        _log(
            f"clamp fired (accept {n_raw}->{n}, block_size={k}). Output is only "
            "verified clean where the clamp never fires; lower DS4_SPEC_BLOCK "
            "if you see repetition."
        )

    if not partial_rollback(cache, n, k):
        # Refused (possibly after partially trimming some layers). Restore the
        # pre-verify state wholesale and re-run only the accepted tokens. The
        # restore overwrites whatever the partial trim did, and the replay is
        # the model itself, so the cache cannot disagree with the model.
        STATS["restore_replay"] += 1
        _restore(snaps)
        _check_offsets(cache, offset, "restore")
        model(mx.array([[t] + drafts[:n]]), cache=cache)

    # Emitted: t (already fed) then the n accepted drafts. `_next_tokens`
    # becomes the target's sample at the first unaccepted position.
    emitted = [(t, lp_prev)] + [(drafts[i], logprobs[i]) for i in range(n)]
    self._next_tokens = mx.array([samples[n]])
    self._next_logprobs = [logprobs[n]]

    _check_offsets(cache, offset + n + 1, f"commit(n={n}/{k})")

    mh = model.model.main_hidden[:, : n + 1]
    for j in range(n + 1):
        pos += 1
        window = drafter.push_window(window, mh[:, j : j + 1], pos)
    self._ds4_window = window
    mx.eval(window, self._next_tokens)

    self._ds4_queue = emitted[1:]
    first_tok, first_lp = emitted[0]
    self.tokens[0].append(first_tok)
    return [first_tok], [first_lp]
