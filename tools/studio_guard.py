#!/usr/bin/env python3
"""Refuse to start a Studio benchmark unless the machine is demonstrably idle.

Run this with oMLX's bundled Python so the authenticated model registry can be
queried without copying API keys or admin secrets into scripts.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import urllib.request


DEFAULT_PORT = 8000
MODEL_FREE_SERVER_RSS_GIB = 8.0
MIN_MEMORY_PRESSURE_FREE_PERCENT = 15.0
MAX_SWAP_USED_GIB = 4.0
SUSPICIOUS_PROCESS = re.compile(
    r"(?:python|mlx|llama|vllm|ollama|benchmark|bench[_-]|inference|generate|"
    r"chat/completions|v1/completions|dgx_parity|spec_rollback)",
    re.IGNORECASE,
)


def _ancestors() -> set[int]:
    rows = subprocess.check_output(
        ["ps", "-axo", "pid=,ppid="], text=True
    ).splitlines()
    parents: dict[int, int] = {}
    for row in rows:
        fields = row.split()
        if len(fields) == 2:
            parents[int(fields[0])] = int(fields[1])
    out: set[int] = set()
    pid = os.getpid()
    while pid > 1 and pid not in out:
        out.add(pid)
        pid = parents.get(pid, 1)
    return out


def _is_omlx_server_command(command: str) -> bool:
    lower = command.lower()
    return (
        "omlx-server" in lower
        or ("omlx" in lower and (" serve" in lower or "/macos/omlx" in lower))
    )


def _suspicious_processes() -> list[str]:
    own = _ancestors()
    rows = subprocess.check_output(
        ["ps", "-axo", "pid=,ppid=,%cpu=,rss=,etime=,command="], text=True
    ).splitlines()
    found: list[str] = []
    for row in rows:
        fields = row.strip().split(None, 5)
        if len(fields) < 6:
            continue
        pid = int(fields[0])
        command = fields[5]
        if (
            pid in own
            or _is_omlx_server_command(command)
            or command.startswith("/System/Library/")
        ):
            continue
        if SUSPICIOUS_PROCESS.search(command):
            found.append(row.strip())
    return found


def _omlx_process_rss() -> tuple[float, list[dict]]:
    """Return total oMLX RSS and a key-free per-process summary.

    The admin registry can report no loaded model while a stale Python strong
    reference still owns Metal-backed weights.  A model-free preflight must
    therefore inspect the serving processes too.  Do not include full command
    lines in the report: launch arguments are not needed and could contain
    credentials in other deployments.
    """
    rows = subprocess.check_output(
        ["ps", "-axo", "pid=,rss=,comm=,command="], text=True
    ).splitlines()
    summaries = []
    rss_kib = 0
    for row in rows:
        fields = row.strip().split(None, 3)
        if len(fields) < 4:
            continue
        pid, rss, comm, command = fields
        if not _is_omlx_server_command(command):
            continue
        value = int(rss)
        rss_kib += value
        summaries.append(
            {
                "pid": int(pid),
                "rss_gib": round(value / 2**20, 2),
                "executable": Path(comm).name,
            }
        )
    return rss_kib / 2**20, summaries


def _established_connections(port: int) -> list[str]:
    proc = subprocess.run(
        ["lsof", "-nP", f"-iTCP:{port}", "-sTCP:ESTABLISHED"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return proc.stdout.splitlines()[1:] if proc.stdout else []


def _loaded_models(port: int) -> list[str]:
    from omlx.admin.auth import (  # pylint: disable=import-outside-toplevel
        SESSION_COOKIE_NAME,
        create_session_token,
        init_auth,
    )

    settings = json.loads((Path.home() / ".omlx/settings.json").read_text())
    init_auth(settings["auth"]["secret_key"])
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/admin/api/models",
        headers={
            "Cookie": f"{SESSION_COOKIE_NAME}={create_session_token()}"
        },
    )
    with urllib.request.urlopen(request, timeout=15) as response:
        models = json.load(response)["models"]
    return [str(model.get("id")) for model in models if model.get("loaded")]


def _memory_headroom_gib() -> tuple[float, float, float]:
    """Return conservative available, immediately-free, and total memory.

    macOS deliberately keeps RAM in its inactive queue, so the raw ``Pages
    free`` value can be tiny on a perfectly healthy machine.  Free + inactive
    is the same conservative base used by oMLX's guard.  Speculative and
    purgeable pages are not added because their accounting can overlap these
    queues.
    Memory pressure remains a separate requirement below.
    """
    output = subprocess.check_output(["vm_stat"], text=True)
    page_match = re.search(r"page size of (\d+) bytes", output)
    if not page_match:
        raise RuntimeError("could not parse vm_stat")
    pages = {
        name: int(value)
        for name, value in re.findall(r"^([^:]+):\s+(\d+)\.", output, re.MULTILINE)
    }
    required = ("Pages free", "Pages inactive")
    if any(name not in pages for name in required):
        raise RuntimeError("vm_stat is missing memory queue counters")
    page_size = int(page_match.group(1))
    free_gib = pages["Pages free"] * page_size / 2**30
    available_gib = sum(pages[name] for name in required) * page_size / 2**30
    total_bytes = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True))
    return available_gib, free_gib, total_bytes / 2**30


def _memory_pressure_free_percent() -> float | None:
    proc = subprocess.run(
        ["memory_pressure", "-Q"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    match = re.search(
        r"System-wide memory free percentage:\s*([0-9.]+)%", proc.stdout
    )
    return float(match.group(1)) if match else None


def _swap_used_gib() -> float:
    output = subprocess.check_output(["sysctl", "-n", "vm.swapusage"], text=True)
    match = re.search(r"used\s*=\s*([0-9.]+)([KMG])", output)
    if not match:
        raise RuntimeError("could not parse vm.swapusage")
    value = float(match.group(1))
    scale = {"K": 2**10, "M": 2**20, "G": 2**30}[match.group(2)]
    return value * scale / 2**30


def preflight(
    required_free_gib: float = 32.0,
    port: int = DEFAULT_PORT,
    expected_loaded: set[str] | None = None,
) -> tuple[dict, list[str]]:
    """Return the machine-idle report and every reason testing must stop.

    Benchmark scripts import this function so the user's "look before every
    test" rule is enforced at the test boundary, not merely by convention in a
    shell command that can be forgotten.
    """
    failures: list[str] = []
    omlx_rss_gib, omlx_processes = _omlx_process_rss()
    registry_reachable = False
    registry_error = None
    try:
        loaded = _loaded_models(port)
        registry_reachable = True
    except Exception as exc:  # fail closed if a serving process exists
        loaded = []
        registry_error = type(exc).__name__

    if not registry_reachable:
        if expected_loaded is not None:
            failures.append("oMLX registry is unavailable; cannot verify loaded model")
        elif omlx_processes:
            failures.append(
                "oMLX registry is unavailable while an oMLX process exists"
            )
    elif expected_loaded is None:
        if loaded:
            failures.append("loaded oMLX models: " + ", ".join(loaded))
    elif set(loaded) != expected_loaded:
        failures.append(
            "loaded model set does not match the test plan: expected "
            f"{sorted(expected_loaded)}, found {sorted(loaded)}"
        )

    connections = _established_connections(port)
    if connections:
        failures.append("active oMLX client connections: " + " | ".join(connections))

    processes = _suspicious_processes()
    if processes:
        failures.append("possible test/inference processes: " + " | ".join(processes))

    if expected_loaded is None and omlx_rss_gib > MODEL_FREE_SERVER_RSS_GIB:
        failures.append(
            "oMLX claims model-free but its processes retain "
            f"{omlx_rss_gib:.1f} GiB RSS (limit {MODEL_FREE_SERVER_RSS_GIB:.1f} GiB)"
        )

    available_gib, immediately_free_gib, total_gib = _memory_headroom_gib()
    if available_gib < required_free_gib:
        failures.append(
            f"only {available_gib:.1f} GiB conservatively available; "
            f"need {required_free_gib:.1f} GiB"
        )
    pressure_free = _memory_pressure_free_percent()
    if (
        pressure_free is not None
        and pressure_free < MIN_MEMORY_PRESSURE_FREE_PERCENT
    ):
        failures.append(
            "system memory pressure reports only "
            f"{pressure_free:.1f}% free; need at least "
            f"{MIN_MEMORY_PRESSURE_FREE_PERCENT:.1f}%"
        )
    swap_used_gib = _swap_used_gib()
    if swap_used_gib > MAX_SWAP_USED_GIB:
        failures.append(
            f"swap use is {swap_used_gib:.1f} GiB; limit is "
            f"{MAX_SWAP_USED_GIB:.1f} GiB"
        )

    return (
        {
            "safe": not failures,
            "registry_reachable": registry_reachable,
            "registry_error_type": registry_error,
            "loaded_models": loaded,
            "expected_loaded_models": (
                sorted(expected_loaded) if expected_loaded is not None else []
            ),
            "active_connections": len(connections),
            "suspicious_processes": processes,
            "omlx_process_rss_gib": round(omlx_rss_gib, 1),
            "omlx_processes": omlx_processes,
            "available_gib": round(available_gib, 1),
            "immediately_free_gib": round(immediately_free_gib, 1),
            "total_gib": round(total_gib, 1),
            "required_free_gib": required_free_gib,
            "memory_pressure_free_percent": pressure_free,
            "minimum_memory_pressure_free_percent": (
                MIN_MEMORY_PRESSURE_FREE_PERCENT
            ),
            "swap_used_gib": round(swap_used_gib, 2),
            "maximum_swap_used_gib": MAX_SWAP_USED_GIB,
        },
        failures,
    )


def assert_safe(
    required_free_gib: float = 32.0,
    port: int = DEFAULT_PORT,
    expected_loaded: set[str] | None = None,
) -> dict:
    """Raise before any allocation or inference unless the Studio is idle."""
    report, failures = preflight(required_free_gib, port, expected_loaded)
    print(json.dumps({"studio_preflight": report}, indent=2))
    if failures:
        raise RuntimeError("REFUSING TEST: " + " ; ".join(failures))
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--required-free-gib", type=float, default=32.0)
    parser.add_argument(
        "--expect-loaded",
        action="append",
        default=None,
        help="model id expected to be resident; repeat for multiple models",
    )
    args = parser.parse_args()

    expected = set(args.expect_loaded) if args.expect_loaded is not None else None
    report, failures = preflight(args.required_free_gib, args.port, expected)
    print(json.dumps(report, indent=2))
    if failures:
        for failure in failures:
            print(f"REFUSING TEST: {failure}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
