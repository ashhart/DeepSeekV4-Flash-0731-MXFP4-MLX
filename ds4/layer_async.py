# SPDX-License-Identifier: MIT
"""Process-once scheduling boundaries for DeepSeek-V4 layer graphs.

MLX builds a lazy graph on the CPU.  With a deep decoder the GPU can otherwise
sit idle until the complete transformer stack reaches its final evaluation.
``mx.async_eval`` at selected layer outputs starts already-built work without
changing any operation, cache update, dtype, or token.

This is deliberately opt-in.  The GUI-launched oMLX server does not inherit a
shell environment, so ``~/.omlx/ds4_layer_async`` is the normal control.  An
empty marker uses the schedule adapted from the accepted mlxfast Laguna work::

    decode=at:0,1,7,15,23,31,39,last
    multi=1

The first line is used for one-token decode; ``multi`` is a layer stride for
prompt prefill and speculative verification.  ``off`` disables either side.
Environment variables ``DS4_LAYER_ASYNC_DECODE`` and
``DS4_LAYER_ASYNC_MULTI`` override the marker fields.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import FrozenSet, Optional, Tuple


_DEFAULT_DECODE = "at:0,1,7,15,23,31,39,last"
_DEFAULT_MULTI = "1"
_UNSET = object()
_CACHED_CONFIGURATION: object = _UNSET


def _marker_values() -> Optional[Tuple[str, str]]:
    path = Path.home() / ".omlx" / "ds4_layer_async"
    if not path.exists():
        return None

    decode = _DEFAULT_DECODE
    multi = _DEFAULT_MULTI
    try:
        raw = path.read_text().strip()
    except OSError:
        return None
    if not raw:
        return decode, multi

    # A bare value is a decode schedule.  Key/value lines let experiments tune
    # decode and multi-token graph staging independently.
    bare = []
    for field in raw.replace(",\n", "\n").splitlines():
        field = field.strip()
        if not field or field.startswith("#"):
            continue
        if "=" not in field:
            bare.append(field)
            continue
        key, value = (part.strip().lower() for part in field.split("=", 1))
        if key == "decode":
            decode = value
        elif key in {"multi", "prefill", "verify"}:
            multi = value
    if bare:
        decode = bare[-1].lower()
    return decode, multi


def _load_configuration() -> Optional[Tuple[str, str]]:
    marker = _marker_values()
    decode_env = os.environ.get("DS4_LAYER_ASYNC_DECODE")
    multi_env = os.environ.get("DS4_LAYER_ASYNC_MULTI")
    if marker is None and decode_env is None and multi_env is None:
        return None
    decode, multi = marker or (_DEFAULT_DECODE, _DEFAULT_MULTI)
    return (decode_env or decode).lower(), (multi_env or multi).lower()


def _configuration() -> Optional[Tuple[str, str]]:
    # File I/O (even a stat) in every decode step is measurable.  Configuration
    # is process-once just like the accepted implementation; changing a GUI
    # marker therefore takes effect on the next server process.
    global _CACHED_CONFIGURATION
    if _CACHED_CONFIGURATION is _UNSET:
        _CACHED_CONFIGURATION = _load_configuration()
    return _CACHED_CONFIGURATION  # type: ignore[return-value]


def _reset_for_tests() -> None:
    global _CACHED_CONFIGURATION
    _CACHED_CONFIGURATION = _UNSET


def _decode_boundaries(raw: str, n_layers: int) -> FrozenSet[int]:
    if n_layers <= 0 or raw in {"", "0", "off"}:
        return frozenset()
    if raw in {"1", "on", "accepted", "default"}:
        raw = _DEFAULT_DECODE

    if raw.startswith("ladder"):
        try:
            stride = int(raw[len("ladder") :])
        except ValueError:
            return frozenset()
        if not 1 <= stride <= n_layers:
            return frozenset()
        return frozenset(i for i in range(n_layers) if (i + 1) % stride == 0)

    if raw.startswith("at:"):
        result = set()
        for field in raw[3:].split(","):
            field = field.strip()
            if field == "last":
                result.add(n_layers - 1)
                continue
            try:
                index = int(field)
            except ValueError:
                return frozenset()
            if not 0 <= index < n_layers:
                return frozenset()
            result.add(index)
        return frozenset(result)

    try:
        index = int(raw)
    except ValueError:
        return frozenset()
    return frozenset({index}) if 0 <= index < n_layers else frozenset()


def _multi_boundaries(raw: str, n_layers: int) -> FrozenSet[int]:
    if n_layers <= 0 or raw in {"", "0", "off"}:
        return frozenset()
    try:
        stride = int(raw)
    except ValueError:
        return frozenset()
    if not 1 <= stride <= n_layers:
        return frozenset()
    return frozenset(i for i in range(n_layers) if (i + 1) % stride == 0)


def boundaries(sequence_length: int, n_layers: int) -> FrozenSet[int]:
    """Return layer indices after which the current graph should be enqueued."""
    configured = _configuration()
    if configured is None:
        return frozenset()
    decode, multi = configured
    if sequence_length == 1:
        return _decode_boundaries(decode, n_layers)
    return _multi_boundaries(multi, n_layers)


def describe(n_layers: int = 43) -> str:
    configured = _configuration()
    if configured is None:
        return "off"
    decode, multi = configured
    one = sorted(_decode_boundaries(decode, n_layers))
    many = sorted(_multi_boundaries(multi, n_layers))
    return f"decode={one}, multi={many if len(many) <= 12 else 'every ' + multi}"
