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
import math
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional

import mlx.core as mx

_PATCHED = False
_DRAFTER_FAILED = False
_METAL_LABELER = None
_METAL_RECORDER = None
_METAL_STATE = threading.local()


def set_metal_profiler(labeler, recorder) -> None:
    """Install test-only command-buffer labels and per-cycle host records."""
    global _METAL_LABELER, _METAL_RECORDER
    _METAL_LABELER = labeler
    _METAL_RECORDER = recorder


def _metal_label(value: str) -> None:
    if _METAL_LABELER is not None:
        _METAL_LABELER(value)


def _metal_phase(value: str) -> None:
    _METAL_STATE.phase = value
    _metal_label(value)


def _metal_record(value: dict) -> None:
    if _METAL_RECORDER is not None:
        _METAL_RECORDER(value)


class _NotReady(Exception):
    """Raised before any model call, so falling back is safe."""


# Cycle accounting, so a disappointing speedup can be attributed: low draft
# acceptance is a different problem from rollback clamping it away. Timings are
# opt-in because even the clock reads should not be in the normal hot path.
# They surround mx.eval calls that already exist, so enabling them adds no GPU
# synchronization and preserves the shape of the workload being measured.
STATS = {
    "cycles": 0,
    "raw_accepted": 0,
    "clamped_accepted": 0,
    "clamp_hits": 0,
    "cache_clamp_hits": 0,
    "stop_clamp_hits": 0,
    "restore_replay": 0,
    "timed_cycles": 0,
    "draft_s": 0.0,
    "snapshot_s": 0.0,
    "verify_sample_s": 0.0,
    "commit_s": 0.0,
    # Commit sub-phases (attack #1: split the ~9 ms partial-rollback host gap).
    "commit_clamp_s": 0.0,
    "commit_roll_s": 0.0,
    "commit_consist_s": 0.0,
    "commit_emit_s": 0.0,
    "commit_window_s": 0.0,
    "partial_timed_cycles": 0,
    "partial_commit_s": 0.0,
    "cycle_s": 0.0,
    "verify_width_counts": [0] * 9,
    "verify_width_timed": [0] * 9,
    "verify_width_s": [0.0] * 9,
    "verify_width_accepted": [0] * 9,
    "confidence_survival_sum": [0.0] * 8,
    "confidence_survival_count": [0] * 8,
    "actual_survival_count": [0] * 8,
}


def stats_summary() -> str:
    c = STATS["cycles"]
    if not c:
        return "no speculative cycles"
    summary = (
        f"cycles={c} "
        f"raw_accept={STATS['raw_accepted'] / c:.2f} "
        f"after_clamp={STATS['clamped_accepted'] / c:.2f} "
        f"tokens/cycle={(STATS['clamped_accepted'] + c) / c:.2f} "
        f"clamped_on={STATS['clamp_hits'] / c * 100:.0f}% "
        f"cache_clamped_on={STATS['cache_clamp_hits'] / c * 100:.0f}% "
        f"stop_clamped_on={STATS['stop_clamp_hits'] / c * 100:.0f}% "
        f"restore_replay={STATS['restore_replay'] / c * 100:.0f}%"
    )
    tc = STATS["timed_cycles"]
    if tc:
        summary += (
            " | ms/cycle "
            f"draft={STATS['draft_s'] / tc * 1000:.2f} "
            f"snapshot={STATS['snapshot_s'] / tc * 1000:.2f} "
            f"verify+sample={STATS['verify_sample_s'] / tc * 1000:.2f} "
            f"commit={STATS['commit_s'] / tc * 1000:.2f} "
            f"[clamp={STATS['commit_clamp_s'] / tc * 1000:.2f} "
            f"roll={STATS['commit_roll_s'] / tc * 1000:.2f} "
            f"consist={STATS['commit_consist_s'] / tc * 1000:.2f} "
            f"emit={STATS['commit_emit_s'] / tc * 1000:.2f} "
            f"window={STATS['commit_window_s'] / tc * 1000:.2f}] "
            f"partial_commit={0 if not STATS['partial_timed_cycles'] else STATS['partial_commit_s'] / STATS['partial_timed_cycles'] * 1000:.2f}x{STATS['partial_timed_cycles']} "
            f"total={STATS['cycle_s'] / tc * 1000:.2f}"
        )
    widths = []
    for width, count in enumerate(STATS["verify_width_counts"]):
        if not count:
            continue
        accepted = STATS["verify_width_accepted"][width] / count
        timing_count = STATS["verify_width_timed"][width]
        latency = (
            f"/{STATS['verify_width_s'][width] / timing_count * 1000:.1f}ms"
            if timing_count
            else ""
        )
        widths.append(f"{width}:{count}@{accepted:.2f}{latency}")
    if widths:
        summary += " | widths=" + ",".join(widths)
    calibration = []
    for i, count in enumerate(STATS["confidence_survival_count"]):
        if not count:
            continue
        predicted = STATS["confidence_survival_sum"][i] / count
        observed = STATS["actual_survival_count"][i] / count
        calibration.append(f"p{i + 1}={predicted:.2f}/{observed:.2f}")
    if calibration:
        summary += " | pred/actual=" + ",".join(calibration)
    return summary


def _report_every() -> int:
    """Optional cumulative server-log interval, in speculative cycles."""
    try:
        configured = os.environ.get("DS4_SPEC_REPORT_EVERY")
        if configured is not None:
            return max(0, int(configured))
        # The GUI-launched server does not inherit shell variables.  Creating
        # the timing marker should therefore make the instrumentation useful on
        # its own rather than silently collecting data that is never reported.
        if (Path.home() / ".omlx" / "ds4_spec_timing").exists():
            return 25
        return 0
    except ValueError:
        return 0


def _timing_enabled() -> bool:
    value = os.environ.get("DS4_SPEC_TIMING")
    if value is not None:
        return value == "1"
    return (Path.home() / ".omlx" / "ds4_spec_timing").exists()


def _profile_sync_enabled() -> bool:
    """Test-only phase isolation with a real Metal barrier after drafting.

    Normal timing intentionally preserves draft/verify overlap, so its draft
    number is only host enqueue time and verify includes the pending drafter.
    This mode sacrifices throughput to attribute GPU work correctly. It must
    never be enabled as a production optimization.
    """
    value = os.environ.get("DS4_SPEC_PROFILE_SYNC")
    if value is not None:
        return value == "1"
    return (Path.home() / ".omlx" / "ds4_spec_profile_sync").exists()


def _maybe_reset_stats() -> None:
    """Consume the profiling reset marker without restarting the server."""
    marker = Path.home() / ".omlx" / "ds4_spec_reset_stats"
    if not marker.exists():
        return
    for key, value in STATS.items():
        if isinstance(value, list):
            value[:] = [0] * len(value)
        elif isinstance(value, float):
            STATS[key] = 0.0
        else:
            STATS[key] = 0
    marker.unlink(missing_ok=True)
    _log("speculative profiling counters reset")


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
    value = os.environ.get("DS4_SPEC_BLOCK")
    if value is None:
        marker = Path.home() / ".omlx" / "ds4_block_size"
        if marker.exists():
            try:
                value = marker.read_text().strip()
            except OSError:
                value = None
    try:
        return max(0, min(7, int(value if value is not None else "3")))
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
    # Measured at 23K cached context, verify width 3, correct rollback:
    #     draft_width 3 -> 35.0   5 -> 35.0   7 -> 38.2 tok/s
    # Wider drafting than we verify raises acceptance: the drafter sees more of
    # its own block, and we only commit a prefix. 7 beat the checkpoint's
    # trained width of 5, so the default is 7 rather than `trained`.
    default = 7
    value = os.environ.get("DS4_SPEC_DRAFT_WIDTH")
    if value is None:
        marker = Path.home() / ".omlx" / "ds4_draft_width"
        if marker.exists():
            try:
                value = marker.read_text().strip()
            except OSError:
                value = None
    try:
        return max(1, min(8, int(value if value is not None else default)))
    except ValueError:
        return default


_SCHEDULE_LOADED = False
_SCHEDULE_CONFIG: Optional[dict] = None


def _schedule_config() -> Optional[dict]:
    """Load the profiled single-request DSpark cost curve once per process.

    The scheduler is intentionally inactive without an explicit profile.  A
    CUDA SPS table would be wrong for Metal, and even old measurements become
    stale when a kernel changes.  The GUI-friendly file format is:

        ~/.omlx/ds4_schedule.json
        {
          "verify_ms": [ms_for_1_token, ..., ms_for_8_tokens],
          "fixed_ms": draft_and_commit_overhead,
          "temperatures": [optional STS temperature per draft position],
          "max_drafts": optional safety cap
        }

    ``verify_ms[k]`` is the target-forward latency for physical input length
    k+1, i.e. anchor plus k draft tokens.
    """
    global _SCHEDULE_LOADED, _SCHEDULE_CONFIG
    if _SCHEDULE_LOADED:
        return _SCHEDULE_CONFIG
    _SCHEDULE_LOADED = True

    path_value = os.environ.get("DS4_SPEC_SCHEDULE")
    if path_value is not None and path_value.strip().lower() in ("", "0", "off"):
        return None
    path = Path(path_value) if path_value else Path.home() / ".omlx" / "ds4_schedule.json"
    if not path.exists():
        return None
    try:
        import json

        config = json.loads(path.read_text())
        verify_ms = [float(value) for value in config["verify_ms"]]
        if not verify_ms or any(value <= 0 for value in verify_ms):
            raise ValueError("verify_ms must contain positive values")
        fixed_ms = max(0.0, float(config.get("fixed_ms", 0.0)))
        temperatures = [
            max(1e-4, float(value)) for value in config.get("temperatures", [])
        ]
        _SCHEDULE_CONFIG = {
            "verify_ms": verify_ms,
            "fixed_ms": fixed_ms,
            "temperatures": temperatures,
            "max_drafts": max(
                0, int(config.get("max_drafts", len(verify_ms) - 1))
            ),
        }
        _log(
            "confidence scheduler enabled from profiled Metal curve "
            f"({len(verify_ms)} widths, fixed={fixed_ms:.2f}ms)"
        )
    except Exception as exc:  # noqa: BLE001
        _log(f"confidence scheduler ignored: {type(exc).__name__}: {exc}")
        _SCHEDULE_CONFIG = None
    return _SCHEDULE_CONFIG


def _confidence_probabilities(raw: list[float], config: Optional[dict]) -> list[float]:
    temperatures = config.get("temperatures", []) if config else []
    out = []
    for i, value in enumerate(raw):
        temperature = temperatures[i] if i < len(temperatures) else 1.0
        z = max(-30.0, min(30.0, value / temperature))
        out.append(1.0 / (1.0 + math.exp(-z)))
    return out


def _scheduled_width(raw: list[float], maximum: int) -> tuple[int, list[float]]:
    """Select the prefix maximizing expected output tokens per millisecond.

    For one request, expected emitted tokens at width k are
    ``1 + sum_j prod_{i<=j} confidence_i``.  The denominator is the measured
    fixed draft/commit cost plus the measured target pass for physical width
    k+1.  This is Algorithm 1 from the DSpark paper specialized to R=1.
    """
    config = _schedule_config()
    probabilities = _confidence_probabilities(raw, config)
    if config is None:
        return maximum, probabilities

    verify_ms = config["verify_ms"]
    maximum = min(maximum, len(verify_ms) - 1, len(probabilities))
    fixed_ms = config["fixed_ms"]
    expected = 1.0
    survival = 1.0
    best_width = 0
    best_rate = expected / (fixed_ms + verify_ms[0])
    # Early stopping is both the paper's causal rule and the correct choice for
    # the smooth single-request curves we profile on this Studio.
    for width in range(1, maximum + 1):
        survival *= probabilities[width - 1]
        expected += survival
        rate = expected / (fixed_ms + verify_ms[width])
        if rate > best_rate:
            best_rate = rate
            best_width = width
        else:
            break
    return best_width, probabilities


def _get_drafter(model) -> Optional[Any]:
    """Build the DSpark drafter once, lazily, from the model's own directory."""
    global _DRAFTER_FAILED
    existing = getattr(model, "_ds4_drafter", None)
    if existing is not None or _DRAFTER_FAILED:
        return existing

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
        # Ownership belongs to the loaded model, not this module. A module-level
        # strong reference kept the 10 GiB drafter alive after oMLX unloaded the
        # target, defeating emergency memory reclaim and making the next load an
        # OOM risk. Model unload now releases target and drafter together.
        model._ds4_drafter = drafter
        _log(
            f"DSpark drafter loaded ({dspark['n_mtp_layers']} stages, "
            f"draft_width={drafter.block_size}, "
            f"verify_width={min(_block_size(), drafter.block_size)})"
        )
    except Exception as e:  # noqa: BLE001
        _DRAFTER_FAILED = True
        _log(f"DSpark drafter NOT loaded: {type(e).__name__}: {e}")
    return getattr(model, "_ds4_drafter", None)


_REPORTED = set()
_EXPECT_WARNED = False
_DISAGREE_DETAILED = False


def _limit_accept_for_terminal(gb, anchor: int, drafts: list[int], accepted: int) -> int:
    """Do not commit queued tokens beyond a terminal state-machine match.

    ``GenerationBatch.next()`` advances the matcher only when each queued token
    is returned to the caller.  The KV cache, however, is committed for the
    whole accepted prefix inside this cycle.  If a stop token occurs in the
    middle of that prefix, ``next()`` immediately extracts the cache at the
    stop and any later committed draft would make it longer than ``all_tokens``.

    Simulate the exact immutable matcher transition from the live request state
    over ``[anchor, accepted drafts]`` and retain the stop token itself, but no
    token after it.  The live matcher is deliberately not changed here; the
    ordinary response path will advance it once per emitted token as usual.
    """
    if accepted <= 0:
        return 0
    machines = getattr(gb, "state_machines", None) or []
    states = getattr(gb, "_matcher_states", None) or []
    if not machines or not states:
        return accepted

    machine = machines[0]
    state = states[0]
    for index, token in enumerate([anchor, *drafts[:accepted]]):
        state, match_sequence, current_state = machine.match(state, int(token))
        if match_sequence is not None and current_state is None:
            # index 0 is the anchor (zero drafts); index N is draft N.
            return min(accepted, index)
    return accepted


def _cap_verify_for_request(gb, maximum_k: int) -> int:
    """Cap drafts so cache commitment cannot pass the request token limit."""
    remaining = int(gb.max_tokens[0]) - int(gb._num_tokens[0])
    return min(maximum_k, max(0, remaining - 1))


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


def _offsets_consistent(cache, expected: int) -> bool:
    """Do caches of the SAME class agree with each other?

    Comparing across classes is meaningless here. This model's layers use two
    different rotating caches:

        layers 0-1   PrefillReadyRotatingKVCache  (single-sequence)
        layers 2-42  BatchRotatingKVCache         (batched: `offset` is an
                                                   mx.array, plus `_offset`
                                                   and `left_padding`)

    Their `offset` fields do not count the same thing, so they sit a constant
    distance apart (measured: exactly 12, never growing, with clean output over
    450-token generations). An earlier version of this check compared them to
    each other and to a single `expected`, declared corruption, and disabled
    speculation — costing ~20% throughput for nothing.

    Real corruption is layers of the *same* class drifting apart, which is what
    a failed rollback produces. That is what this now tests.
    """
    # One eval, not one per cache. `BatchRotatingKVCache.offset` is an
    # mx.array; right after a trim each holds a fresh lazy `offset - n` node,
    # so calling `.item()` per cache forces up to ~41 separate host-device
    # round trips. Concatenating them and materialising once turns that into a
    # single sync. Marker-gated for A/B: absent -> original per-item path.
    batched = (Path.home() / ".omlx" / "ds4_consist_batched").exists()

    by_cls: dict = {}
    if batched:
        names, arrs, ints = [], [], []
        for layer_cache in cache:
            for c in _unwrap(layer_cache):
                name = type(c).__name__
                if not name.endswith("RotatingKVCache"):
                    continue
                o = c.offset
                if isinstance(o, mx.array):
                    names.append(name)
                    arrs.append(o.reshape(-1)[:1])
                else:
                    try:
                        ints.append((name, int(o)))
                    except Exception:  # noqa: BLE001
                        pass
        if arrs:
            try:
                values = mx.concatenate(arrs).tolist()  # ONE sync
            except Exception:  # noqa: BLE001
                values = []
                names = []
        else:
            values = []
        for name, value in zip(names, values):
            by_cls.setdefault(name, set()).add(int(value))
        for name, value in ints:
            by_cls.setdefault(name, set()).add(value)
    else:
        for layer_cache in cache:
            for c in _unwrap(layer_cache):
                name = type(c).__name__
                if not name.endswith("RotatingKVCache"):
                    continue
                o = c.offset
                try:
                    o = int(o.item() if hasattr(o, "item") else o)
                except Exception:  # noqa: BLE001
                    continue
                by_cls.setdefault(name, set()).add(o)

    for name, offsets in by_cls.items():
        if len(offsets) > 1:
            _log(f"{name} layers disagree: {sorted(offsets)[:6]} — rollback failed")
            return False
    return True


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


def clamp_accept(cache, accepted: int, num_drafts: int) -> Optional[int]:
    """Largest m <= accepted whose rollback every layer supports.

    Emitting fewer verified drafts than acceptance allowed is always correct --
    the skipped ones are re-derived next cycle -- so this keeps the cycle alive
    when a PoolingCache cannot replay a longer confirmed prefix. ``None`` means
    no accepted prefix is directly trimmable; the caller must undo the complete
    verify update and replay the accepted prefix instead.
    """
    entries = [c for layer_cache in cache for c in _unwrap(layer_cache)]
    for m in range(accepted, -1, -1):
        n = num_drafts - m
        if n <= 0 or all(_can_trim(c, n) for c in entries):
            return m
    return None


def partial_rollback(cache, accepted: int, num_drafts: int) -> bool:
    """Trim the verify window back to `accepted` drafts on every layer.

    Trims each **concrete** cache, not the `CacheList` wrapper. `CacheList.trim`
    is:

        def trim(self, n):
            for c in self.caches:
                m = c.trim(n)
            return m          # only the LAST sub-cache's result

    so if the rotating sub-cache refuses (returns 0) while the pooling one
    succeeds, the wrapper reports success and the rotating cache silently never
    gets trimmed. Each layer then drifts by `n` per cycle, layers end up at
    different lengths, and generation degrades into fluent repetition. Observed
    as `layers disagree on offset: [24298, 24310]`.
    """
    n = num_drafts - accepted
    if n <= 0:
        return True

    entries = [c for layer_cache in cache for c in _unwrap(layer_cache)]
    if not all(_can_trim(c, n) for c in entries):
        return False
    for c in entries:
        if c.trim(n) != n:
            return False
    return True


def _builtin_verify_undo_supported(cache) -> bool:
    """Does every concrete cache carry a complete one-update verify undo?

    Current oMLX PoolingCaches save ``_undo`` for updates up to eight tokens,
    while its rotating-cache patch saves ``_mtp_undo`` when the verifier is
    armed. In that known layout a full pre-verify snapshot merely duplicates
    the cache-owned transaction logs. Unknown cache types stay on the generic
    snapshot path so a future oMLX layout cannot silently weaken correctness.
    """
    entries = [c for layer_cache in cache for c in _unwrap(layer_cache)]
    if not entries:
        return False
    for c in entries:
        if getattr(c, "remainder", None) is not None:
            if not hasattr(c, "_undo") or not callable(getattr(c, "_can_undo", None)):
                return False
            continue
        if type(c).__name__.endswith("RotatingKVCache"):
            if not getattr(type(c), "_omlx_mtp_undo_attached", False):
                return False
            continue
        return False
    return True


def _undo_verify(cache, verify_tokens: int) -> bool:
    """Undo the complete armed verify update on every concrete cache.

    Unlike a partial rollback this never needs to reconstruct a completed pool
    window: replay length is zero. It is therefore the safe boundary fallback
    when no ``m <= accepted`` is directly trimmable. The caller then replays
    only ``[t, accepted drafts]`` through the target model.
    """
    entries = [c for layer_cache in cache for c in _unwrap(layer_cache)]
    if not entries or not all(_can_trim(c, verify_tokens) for c in entries):
        return False
    for c in entries:
        if c.trim(verify_tokens) != verify_tokens:
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
    """Make DeepseekV4Model stash hidden states and stage layer graphs.

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
        from ds4 import layer_async

        targets = set(
            range(self.args.num_hidden_layers - n_last, self.args.num_hidden_layers)
        )
        fire_after = layer_async.boundaries(inputs.shape[1], len(self.layers))
        captured = []
        layer_cls = type(self.layers[0])
        inner = layer_cls.__call__
        index_of = {id(layer): i for i, layer in enumerate(self.layers)}
        staged_from = 0

        def wrapper(layer_self, *a, **kw):
            nonlocal staged_from
            out = inner(layer_self, *a, **kw)
            index = index_of.get(id(layer_self))
            if index in targets:
                captured.append(out.mean(axis=2))
            if index in fire_after:
                phase = getattr(_METAL_STATE, "phase", "model")
                _metal_label(
                    f"{phase}/layers_{staged_from:02d}_{index:02d}"
                )
                mx.async_eval(out)
                staged_from = index + 1
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
        # oMLX may re-register the DeepSeek-V4 module during model load, so
        # reapply the opt-in cache patch here before the first forward.
        try:
            from ds4 import cache_async

            cache_async.apply()
        except Exception as exc:  # noqa: BLE001
            _log(f"async cache patch skipped: {type(exc).__name__}: {exc}")
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
    global _DRAFTER_FAILED
    model = self.model
    # maximum_k is the VERIFY budget -- the confidence scheduler may choose a
    # shorter prefix after the drafter has produced its fixed-width block.
    # drafter.block_size is the DRAFT width, kept at the trained value so the
    # non-causal draft block stays in distribution. They are decoupled on
    # purpose; see `_draft_width`.
    schedule = _schedule_config()
    if schedule is None:
        maximum_k = min(_block_size(), drafter.block_size)
    else:
        maximum_k = min(
            schedule["max_drafts"], len(schedule["verify_ms"]) - 1,
            drafter.block_size,
        )

    # GenerationBatch increments _num_tokens only after _step returns.  Limit
    # the accepted queue to what the request can still emit, otherwise a final
    # speculative cycle can commit tokens past max_tokens.  oMLX then extracts
    # that too-long cache under the shorter visible token list, contaminating
    # future prefix-cache hits.  Falling back at one token remaining is safe
    # because _NotReady is raised before any model/drafter call.
    maximum_k = _cap_verify_for_request(self, maximum_k)
    if maximum_k == 0:
        raise _NotReady("final request token must use the stock step")
    cache = self.prompt_cache

    # Everything that can fail must happen BEFORE the verify forward: once the
    # model has run, the cache has advanced and falling back to the stock path
    # would decode against inconsistent state.
    tok_arr = self._next_tokens
    if tok_arr is None or not self._next_logprobs:
        raise _NotReady("pipeline not primed yet")
    lp_prev = self._next_logprobs[0]
    profile_sync = _profile_sync_enabled()
    timing = _timing_enabled() or profile_sync
    if timing:
        _maybe_reset_stats()
    cycle_number = STATS["cycles"] + 1
    cycle_prefix = f"cycle/{cycle_number:04d}"
    _metal_phase(f"{cycle_prefix}/draft/build")
    cycle_started = time.perf_counter() if timing else 0.0
    draft_started = cycle_started

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

    draft_ids, raw_confidence = drafter(
        tok_arr.reshape(1, 1),
        model.model.embed_tokens,
        model.lm_head,
        window,
        pos,
    )
    if profile_sync:
        # Attribute the drafter's queued Metal work to the draft phase instead
        # of the target's first blocking eval. Production deliberately omits
        # this barrier so both graphs can share one command-buffer pipeline.
        _metal_phase(f"{cycle_prefix}/draft/eval")
        mx.eval(draft_ids, raw_confidence)
    drafts = None
    raw_confidence_values = None
    if schedule is None:
        # Keep draft -> target entirely on the GPU.  Pulling draft_ids to the
        # host just to rebuild this array introduced a hard command-buffer
        # boundary between the two largest pieces of every cycle.  The fixed
        # width is shape-known, so the target can consume the lazy draft array
        # directly and all host-visible values resolve at the post-verify sync.
        k = maximum_k
        cand_arr = mx.concatenate(
            [tok_arr.reshape(1, 1), draft_ids[:, :k].astype(tok_arr.dtype)],
            axis=1,
        )
        confidence_probabilities = None
    else:
        # The optional confidence scheduler makes a genuinely data-dependent
        # shape choice, so this mode alone retains the mid-cycle sync.
        mx.eval(draft_ids, raw_confidence)
        raw_confidence_values = [float(x) for x in raw_confidence[0].tolist()]
        k, confidence_probabilities = _scheduled_width(
            raw_confidence_values, maximum_k
        )
        drafts = [int(x) for x in draft_ids[0].tolist()][:k]
        cand_arr = mx.array([[int(tok_arr[0])] + drafts])
    draft_finished = time.perf_counter() if timing else 0.0

    # Verify [t, d1..dk] in one forward, with oMLX's undo log armed.
    #
    # cache_rollback wraps RotatingKVCache.update_and_fetch to stash a DETACHED
    # snapshot (`v + 0` -- a plain reference would see the post-update value,
    # which is the trap I fell into rolling my own) for any armed update of
    # 2..8 tokens, and makes trim() replay the confirmed prefix from it. It is
    # only armed around a verify forward, so stock decode keeps stock semantics.
    # Current oMLX caches already snapshot the same verify update in `_undo` /
    # `_mtp_undo`. Duplicating every array in all ~105 concrete caches here cost
    # ~11% of decode (32.5 -> 36.2 tok/s at 23K). Use the built-in transaction
    # logs for the known layout; keep the generic snapshot for an unfamiliar
    # cache type and as an explicit diagnostic override.
    snapshot_started = time.perf_counter() if timing else 0.0
    needs_snapshot = (
        os.environ.get("DS4_FORCE_SNAPSHOT") == "1"
        or not _builtin_verify_undo_supported(cache)
    )
    snaps = _snapshot(cache) if needs_snapshot else None
    _check_offsets(cache, offset, "pre-verify")
    snapshot_finished = time.perf_counter() if timing else 0.0
    verify_started = snapshot_finished

    _metal_phase(f"{cycle_prefix}/verify/target")
    _set_armed(True)
    try:
        logits = model(cand_arr, cache=cache)[0]
    finally:
        _set_armed(False)
    del cand_arr
    _metal_phase(f"{cycle_prefix}/verify/tail_sample")
    logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)

    sampler = (self.samplers and self.samplers[0]) or self.fallback_sampler
    if float(getattr(sampler, "temp", -1.0) or 0.0) == 0.0:
        # oMLX's temp=0 sampler is exactly argmax. One batched reduction is
        # cheaper than k+1 callable invocations and has identical semantics.
        target_ids = mx.argmax(logits, axis=-1).astype(mx.int32)
        mx.eval(logprobs, target_ids, draft_ids, raw_confidence, tok_arr)
        samples = [int(value) for value in target_ids.tolist()]
    else:
        # Preserve stochastic sampler call/RNG order while resolving every row
        # at the same barrier.
        sample_arrays = [sampler(logprobs[i : i + 1]) for i in range(k + 1)]
        mx.eval(
            logprobs,
            *sample_arrays,
            draft_ids,
            raw_confidence,
            tok_arr,
        )
        samples = [int(sample[0]) for sample in sample_arrays]
    t = int(tok_arr[0])
    if drafts is None:
        drafts = [int(x) for x in draft_ids[0].tolist()][:k]
    if raw_confidence_values is None:
        raw_confidence_values = [float(x) for x in raw_confidence[0].tolist()]
    if confidence_probabilities is None:
        confidence_probabilities = _confidence_probabilities(
            raw_confidence_values, None
        )
    verify_finished = time.perf_counter() if timing else 0.0

    # Longest prefix where the target's own sample reproduces the draft.
    n = 0
    while n < k and samples[n] == drafts[n]:
        n += 1

    # Clamp at an actual terminal transition rather than disabling speculation
    # for every ordinary chat/EOS state machine.  This must happen before cache
    # rollback feasibility is chosen because it changes the committed prefix.
    n_raw = n
    n_stop = _limit_accept_for_terminal(self, t, drafts, n_raw)

    # Not every accepted length is rollback-able: a PoolingCache can only replay
    # a confirmed prefix that stays inside its window. `mtp_clamp_accept` finds
    # the largest m <= n that every layer can undo. Emitting fewer drafts than
    # were verified is always correct -- the rest are re-derived next cycle.
    clamp_started = time.perf_counter() if timing else 0.0
    feasible = clamp_accept(cache, n_stop, k)
    clamp_finished = time.perf_counter() if timing else 0.0
    replay_from_start = feasible is None
    n = n_stop if replay_from_start else int(feasible)
    STATS["cycles"] += 1
    STATS["raw_accepted"] += n_raw
    STATS["clamped_accepted"] += n
    STATS["clamp_hits"] += int(n != n_raw)
    STATS["cache_clamp_hits"] += int(n != n_stop)
    STATS["stop_clamp_hits"] += int(n_stop != n_raw)
    STATS["verify_width_counts"][k] += 1
    STATS["verify_width_accepted"][k] += n_raw
    _metal_phase(f"{cycle_prefix}/commit/cache")
    predicted_survival = 1.0
    for i in range(k):
        predicted_survival *= confidence_probabilities[i]
        STATS["confidence_survival_sum"][i] += predicted_survival
        STATS["confidence_survival_count"][i] += 1
        STATS["actual_survival_count"][i] += int(n_raw > i)

    if n_stop != n_raw and STATS["stop_clamp_hits"] == 1:
        _log(
            f"terminal match clamped accepted prefix {n_raw}->{n_stop}; "
            "the stop token remains committed and no queued token follows it"
        )

    if n != n_stop and STATS["cache_clamp_hits"] == 1:
        # Correctness is unchanged: skipped accepted drafts are re-derived next
        # cycle. This is a throughput signal that the verify width is crossing
        # pool boundaries often enough to waste target work.
        _log(
            f"pooling-cache clamp fired (accept {n_stop}->{n}, block_size={k}); "
            "correctness is "
            "preserved, but a lower DS4_SPEC_BLOCK may be faster"
        )

    roll_started = time.perf_counter() if timing else 0.0
    if replay_from_start:
        # Ratio-4 PoolingCache can hit a phase where every m <= n_raw would
        # replay across a pool boundary. Its one-update log can still undo the
        # entire verify (zero-token replay), after which a normal target forward
        # commits the accepted prefix exactly. This replaces the old every-cycle
        # full snapshot with work only on the rare boundary cycle.
        STATS["restore_replay"] += 1
        if not _undo_verify(cache, k + 1):
            if snaps is None:
                _DRAFTER_FAILED = True
                _log("complete verify undo refused; speculation disabled")
                raise RuntimeError("complete verify undo refused")
            _restore(snaps)
        _check_offsets(cache, offset, "full-undo")
        model(mx.array([[t] + drafts[:n]]), cache=cache)
    elif not partial_rollback(cache, n, k):
        # Refused (possibly after partially trimming some layers). Restore the
        # pre-verify state wholesale and re-run only the accepted tokens. The
        # restore overwrites whatever the partial trim did, and the replay is
        # the model itself, so the cache cannot disagree with the model.
        STATS["restore_replay"] += 1
        if snaps is None:
            _DRAFTER_FAILED = True
            _log("rollback contradicted its pre-check; speculation disabled")
            raise RuntimeError("rollback contradicted its pre-check")
        _restore(snaps)
        _check_offsets(cache, offset, "restore")
        model(mx.array([[t] + drafts[:n]]), cache=cache)

    roll_finished = time.perf_counter() if timing else 0.0

    # Emitted: t (already fed) then the n accepted drafts. `_next_tokens`
    # becomes the target's sample at the first unaccepted position.
    emitted = [(t, lp_prev)] + [(drafts[i], logprobs[i]) for i in range(n)]
    self._next_tokens = mx.array([samples[n]])
    self._next_logprobs = [logprobs[n]]

    _check_offsets(cache, offset + n + 1, f"commit(n={n}/{k})")

    # Diagnostic only -- deliberately does NOT raise.
    #
    # Raising here was a mistake: by this point the verify has already advanced
    # the cache, so the fallback decodes against that state and throughput
    # collapsed to 8.5 tok/s. Worse, the check itself was unsound -- isolation
    # testing (tools/batch_cache_rollback.py) shows the rollback lands exactly
    # where a plain decode would, 16/16, on both RotatingKVCache and
    # BatchRotatingKVCache, so the mismatch was in the computed `expected`, not
    # the caches. Log it and keep going; DS4_SPEC_STRICT=1 restores the abort.
    consist_started = time.perf_counter() if timing else 0.0
    consistent = _offsets_consistent(cache, offset + n + 1)
    consist_finished = time.perf_counter() if timing else 0.0
    if not consistent:
        if os.environ.get("DS4_SPEC_STRICT") == "1":
            _DRAFTER_FAILED = True
            _log("layers disagree on offset -- speculation disabled (STRICT)")
            raise RuntimeError("speculative rollback left caches inconsistent")

    emit_finished = time.perf_counter() if timing else 0.0
    mh = model.model.main_hidden[:, : n + 1]
    # `push_window` accepts a whole [offset, offset + S) sequence.  Calling it
    # once per committed token rereads the 3D->D main projection (about 100 MB)
    # and launches each stage's KV projection n+1 times.  The target verify has
    # already produced these hidden states as one sequence, so commit them the
    # same way: one fused projection and one KV projection per stage.  RoPE's
    # offset is the first position, exactly matching the old loop's first
    # `pos += 1`; concatenation and final window truncation are equivalent.
    window = drafter.push_window(window, mh, pos + 1)
    pos += n + 1
    self._ds4_window = window

    # The verify result above is the cycle's one unavoidable host barrier: the
    # accepted-prefix length controls cache rollback. Materialising the draft
    # window here added a second barrier even though no host value depends on
    # it. Queue it asynchronously instead; accepted tokens are emitted while
    # Metal prepares the next cycle, and MLX's dependency graph still makes a
    # following drafter call wait for the completed window. Keep the blocking
    # form only when timing is explicitly enabled so phase measurements remain
    # honest.
    if timing:
        _metal_phase(f"{cycle_prefix}/commit/window_eval")
        mx.eval(window, self._next_tokens)
    else:
        _metal_phase(f"{cycle_prefix}/commit/window_async")
        mx.async_eval(window, self._next_tokens)

    if timing:
        cycle_finished = time.perf_counter()
        STATS["timed_cycles"] += 1
        STATS["draft_s"] += draft_finished - draft_started
        STATS["snapshot_s"] += snapshot_finished - snapshot_started
        STATS["verify_sample_s"] += verify_finished - verify_started
        STATS["commit_s"] += cycle_finished - verify_finished
        STATS["commit_clamp_s"] += clamp_finished - clamp_started
        STATS["commit_roll_s"] += roll_finished - roll_started
        STATS["commit_consist_s"] += consist_finished - consist_started
        STATS["commit_emit_s"] += emit_finished - consist_finished
        STATS["commit_window_s"] += cycle_finished - emit_finished
        if n < k:
            STATS["partial_timed_cycles"] += 1
            STATS["partial_commit_s"] += cycle_finished - verify_finished
        STATS["cycle_s"] += cycle_finished - cycle_started
        STATS["verify_width_timed"][k] += 1
        STATS["verify_width_s"][k] += cycle_finished - cycle_started
        _metal_record(
            {
                "cycle": cycle_number,
                # On macOS perf_counter() and Metal's GPUStartTime/GPUEndTime
                # share the mach-absolute clock domain. Persist both ends so a
                # report can reject command buffers that were labelled during
                # a cycle but actually belong to later request cleanup.
                "host_start_s": cycle_started,
                "host_end_s": cycle_finished,
                "verify_width": k,
                "raw_accepted": n_raw,
                "committed_drafts": n,
                "draft_ms": (draft_finished - draft_started) * 1000,
                "snapshot_ms": (snapshot_finished - snapshot_started) * 1000,
                "verify_sample_ms": (verify_finished - verify_started) * 1000,
                "commit_ms": (cycle_finished - verify_finished) * 1000,
                "total_ms": (cycle_finished - cycle_started) * 1000,
                "active_gib": mx.get_active_memory() / 2**30,
                "peak_gib": mx.get_peak_memory() / 2**30,
            }
        )

    report_every = _report_every()
    if report_every and STATS["cycles"] % report_every == 0:
        _log(stats_summary())

    # Labels are sampled when a command buffer is committed. Do not leave the
    # last cycle's label live: response finalisation, prefix-cache extraction,
    # or an unrelated request would otherwise be charged to that cycle.
    _metal_phase("server/outside_cycle")

    self._ds4_queue = emitted[1:]
    first_tok, first_lp = emitted[0]
    self.tokens[0].append(first_tok)
    return [first_tok], [first_lp]
