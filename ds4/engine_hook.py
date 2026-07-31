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


def _log(msg: str) -> None:
    print(f"[ds4] {msg}", file=sys.stderr)


def enabled() -> bool:
    return os.environ.get("DS4_SPEC", "0") == "1"


def _block_size() -> int:
    try:
        return int(os.environ.get("DS4_SPEC_BLOCK", "5"))
    except ValueError:
        return 5


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
        drafter.block_size = _block_size()
        _DRAFTER = drafter
        _log(
            f"DSpark drafter loaded ({dspark['n_mtp_layers']} stages, "
            f"block_size={drafter.block_size})"
        )
    except Exception as e:  # noqa: BLE001
        _DRAFTER_FAILED = True
        _log(f"DSpark drafter NOT loaded: {type(e).__name__}: {e}")
    return _DRAFTER


def _eligible(gb) -> bool:
    if not enabled():
        return False
    if len(getattr(gb, "uids", []) or []) != 1:
        return False
    if any(getattr(gb, "logits_processors", None) or []):
        return False
    model = gb.model
    if getattr(model, "model_type", None) != "deepseek_v4":
        return False
    return getattr(model.model, "main_hidden", None) is not None


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


def _rollback(cache, n: int) -> bool:
    if n <= 0:
        return True
    for layer_cache in cache:
        for c in _unwrap(layer_cache):
            if type(c).__name__.endswith("RotatingKVCache"):
                _trim_rotating(c, n)
            elif hasattr(c, "trim"):
                if c.trim(n) != n:
                    return False
            else:
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
    if getattr(cls, "_ds4_hidden_patched", False):
        return
    original = cls.__call__

    def __call__(self, inputs, cache=None):
        targets = set(
            range(self.args.num_hidden_layers - n_last, self.args.num_hidden_layers)
        )
        captured = []
        layer_cls = type(self.layers[0])
        inner = layer_cls.__call__
        index_of = {id(layer): i for i, layer in enumerate(self.layers)}

        def wrapper(layer_self, h, mask, c, ids):
            out = inner(layer_self, h, mask, c, ids)
            if index_of.get(id(layer_self)) in targets:
                captured.append(out.mean(axis=2))
            return out

        layer_cls.__call__ = wrapper
        try:
            out = original(self, inputs, cache)
        finally:
            layer_cls.__call__ = inner
        self.main_hidden = (
            mx.concatenate(captured, axis=-1) if len(captured) == n_last else None
        )
        return out

    cls.__call__ = __call__
    cls._ds4_hidden_patched = True


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
    """Patch `GenerationBatch._step` with the speculative path."""
    global _PATCHED
    if _PATCHED:
        return False

    _apply_hidden_capture()
    _apply_path_capture()

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
    k = drafter.block_size
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
        # Seed from the prefill's hidden states; a short window only costs
        # acceptance, never correctness.
        window = drafter.new_window(1)
        mh = model.model.main_hidden
        start = max(0, offset - mh.shape[1])
        window = drafter.push_window(window, mh, start)

    draft_ids, _conf = drafter(
        mx.array([[t]]),
        model.model.embed_tokens,
        model.lm_head,
        window,
        pos,
    )
    mx.eval(draft_ids)
    drafts = [int(x) for x in draft_ids[0].tolist()]

    # Verify [t, d1..dk] in one forward.
    cand = mx.array([[t] + drafts])
    logits = model(cand, cache=cache)[0]  # (k+1, V)
    logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)

    sampler = (self.samplers and self.samplers[0]) or self.fallback_sampler
    samples = [int(sampler(logprobs[i : i + 1])[0]) for i in range(k + 1)]
    mx.eval(logprobs)

    # Longest prefix where the target's own sample reproduces the draft.
    n = 0
    while n < k and samples[n] == drafts[n]:
        n += 1

    if not _rollback(cache, k - n):
        raise RuntimeError("cache rollback unsupported")

    # Emitted: t (already fed) then the n accepted drafts. `_next_tokens`
    # becomes the target's sample at the first unaccepted position.
    emitted = [(t, lp_prev)] + [(drafts[i], logprobs[i]) for i in range(n)]
    self._next_tokens = mx.array([samples[n]])
    self._next_logprobs = [logprobs[n]]

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
