# SPDX-License-Identifier: MIT
"""Marker-gated bridge to the lightweight Metal command-buffer timing probe."""

from __future__ import annotations

import ctypes
import json
from pathlib import Path
import threading
from typing import Any


MARKER = Path.home() / ".omlx" / "ds4_metal_probe"
LIBRARY = Path.home() / "ds4/tools/libds4_metal_probe.dylib"
COMMAND_LOG = Path("/tmp/ds4-server-metal.jsonl")
HOST_LOG = Path("/tmp/ds4-server-host.jsonl")

_LIB = None
_HOST_FILE = None
_HOST_LOCK = threading.Lock()


def enabled() -> bool:
    return MARKER.exists()


def label(value: str) -> None:
    if _LIB is not None:
        _LIB.ds4_metal_probe_set_label(value.encode())


def record_host(value: dict[str, Any]) -> None:
    if _HOST_FILE is None:
        return
    with _HOST_LOCK:
        _HOST_FILE.write(json.dumps(value, separators=(",", ":")) + "\n")
        _HOST_FILE.flush()


def apply() -> bool:
    global _LIB, _HOST_FILE
    if not enabled():
        return False
    if _LIB is not None:
        return True
    if not LIBRARY.exists():
        raise FileNotFoundError(LIBRARY)

    COMMAND_LOG.unlink(missing_ok=True)
    HOST_LOG.unlink(missing_ok=True)
    library = ctypes.CDLL(str(LIBRARY))
    library.ds4_metal_probe_install.argtypes = [ctypes.c_char_p]
    library.ds4_metal_probe_install.restype = ctypes.c_int
    library.ds4_metal_probe_set_label.argtypes = [ctypes.c_char_p]
    result = library.ds4_metal_probe_install(str(COMMAND_LOG).encode())
    if result != 0:
        raise RuntimeError(f"Metal probe install failed: {result}")

    _LIB = library
    _HOST_FILE = HOST_LOG.open("w", buffering=1)
    from ds4 import engine_hook

    engine_hook.set_metal_profiler(label, record_host)
    label("server/setup")
    return True
