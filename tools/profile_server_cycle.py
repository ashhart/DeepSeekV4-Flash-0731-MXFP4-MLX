#!/usr/bin/env python3
"""Profile one real oMLX DSpark request from host phases to Metal layer pairs."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics
import time
import urllib.request

import studio_guard


MODEL = "DeepSeek-V4-Flash-0731-MXFP4-MLX"
URL = "http://127.0.0.1:8000/v1/chat/completions"
COMMAND_LOG = Path("/tmp/ds4-server-metal.jsonl")
HOST_LOG = Path("/tmp/ds4-server-host.jsonl")
OUTPUT = Path.home() / "ds4/profiles/server_cycle_metal_ledger.json"
REQUIRED_MARKERS = (
    Path.home() / ".omlx/ds4_spec_enabled",
    Path.home() / ".omlx/ds4_spec_timing",
    Path.home() / ".omlx/ds4_spec_profile_sync",
    Path.home() / ".omlx/ds4_metal_probe",
)


def _union_ms(rows):
    intervals = sorted(
        (row["gpu_start_s"], row["gpu_end_s"])
        for row in rows
        if row["gpu_end_s"] > row["gpu_start_s"]
    )
    if not intervals:
        return 0.0
    start, end = intervals[0]
    total = 0.0
    for next_start, next_end in intervals[1:]:
        if next_start <= end:
            end = max(end, next_end)
        else:
            total += end - start
            start, end = next_start, next_end
    return (total + end - start) * 1000


def _summary(values):
    return {
        "count": len(values),
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "min": min(values),
        "max": max(values),
    }


def main() -> int:
    missing = [str(path) for path in REQUIRED_MARKERS if not path.exists()]
    if missing:
        raise RuntimeError(f"required profiling markers missing: {missing}")
    studio_guard.assert_safe(required_free_gib=64.0, expected_loaded={MODEL})
    if not COMMAND_LOG.exists() or not HOST_LOG.exists():
        raise RuntimeError("backend Metal probe logs do not exist")

    # Consumed by the first cycle before its cycle number is assigned.
    (Path.home() / ".omlx/ds4_spec_reset_stats").touch()
    key = json.loads((Path.home() / ".omlx/settings.json").read_text())["auth"][
        "api_key"
    ]
    body = json.dumps(
        {
            "model": MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Continue explaining in precise technical prose why a "
                        "high-bandwidth GPU can underperform optimized FP4 inference."
                    ),
                }
            ],
            "max_tokens": 96,
            "temperature": 0,
        }
    ).encode()
    request = urllib.request.Request(
        URL,
        data=body,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=180) as response:
        payload = json.load(response)
    elapsed = time.perf_counter() - started
    time.sleep(0.1)
    post_request_report, post_request_failures = studio_guard.preflight(
        required_free_gib=64.0,
        expected_loaded={MODEL},
    )

    command_rows = [
        json.loads(line)
        for line in COMMAND_LOG.read_text().splitlines()
        if '"label":"cycle/' in line
    ]
    host_rows = [json.loads(line) for line in HOST_LOG.read_text().splitlines()]
    if not command_rows or not host_rows:
        raise RuntimeError("profiling request produced no cycle records")

    host_by_cycle = {int(row["cycle"]): row for row in host_rows}
    metal_by_cycle = defaultdict(list)
    for row in command_rows:
        parts = row["label"].split("/")
        metal_by_cycle[int(parts[1])].append(row)
    cycle_ids = sorted(set(host_by_cycle) & set(metal_by_cycle))
    if not cycle_ids:
        raise RuntimeError("host and Metal cycle IDs did not intersect")

    categories = defaultdict(list)
    layer_pairs = defaultdict(list)
    per_cycle = []
    for cycle in cycle_ids:
        rows = metal_by_cycle[cycle]
        category_sums = defaultdict(float)
        pair_sums = defaultdict(float)
        for row in rows:
            parts = row["label"].split("/")[2:]
            label = "/".join(parts)
            if label.startswith("verify/target/layers_"):
                pair = label.removeprefix("verify/target/")
                category = "verify/target/layers"
                pair_sums[pair] += row["gpu_ms"]
            else:
                category = label
            category_sums[category] += row["gpu_ms"]
        for category, value in category_sums.items():
            categories[category].append(value)
        for pair, value in pair_sums.items():
            layer_pairs[pair].append(value)
        starts = [row["gpu_start_s"] for row in rows]
        ends = [row["gpu_end_s"] for row in rows]
        busy = _union_ms(rows)
        span = (max(ends) - min(starts)) * 1000
        per_cycle.append(
            {
                "cycle": cycle,
                "metal_command_buffers": len(rows),
                "gpu_busy_ms": busy,
                "gpu_span_ms": span,
                "gpu_idle_gap_ms": max(0.0, span - busy),
                **host_by_cycle[cycle],
            }
        )

    completion_tokens = int(payload["usage"]["completion_tokens"])
    message = payload["choices"][0]["message"]
    encoded = json.dumps(message, sort_keys=True, ensure_ascii=False).encode()
    result = {
        "measurement": {
            "mode": "profile-sync phase isolation",
            "production_throughput": False,
            "notes": [
                "The draft/target barrier deliberately removes production overlap.",
                "Use phase medians for attribution; do not quote whole_request_tok_s as production speed.",
                "The command-buffer probe records timestamps and labels only, never Metal resources or tensor contents.",
            ],
        },
        "request": {
            "completion_tokens": completion_tokens,
            "elapsed_s": elapsed,
            "whole_request_tok_s": completion_tokens / elapsed,
            "message_sha256": hashlib.sha256(encoded).hexdigest(),
        },
        "cycles": {
            "count": len(cycle_ids),
            "tokens_per_cycle": _summary(
                [row["committed_drafts"] + 1 for row in per_cycle]
            ),
            "raw_accepted": _summary([row["raw_accepted"] for row in per_cycle]),
            "host_total_ms": _summary([row["total_ms"] for row in per_cycle]),
            "host_draft_ms": _summary([row["draft_ms"] for row in per_cycle]),
            "host_verify_sample_ms": _summary(
                [row["verify_sample_ms"] for row in per_cycle]
            ),
            "host_commit_ms": _summary([row["commit_ms"] for row in per_cycle]),
            "metal_gpu_busy_ms": _summary(
                [row["gpu_busy_ms"] for row in per_cycle]
            ),
            "metal_gpu_span_ms": _summary(
                [row["gpu_span_ms"] for row in per_cycle]
            ),
            "metal_gpu_idle_gap_ms": _summary(
                [row["gpu_idle_gap_ms"] for row in per_cycle]
            ),
            "peak_gib": max(row["peak_gib"] for row in per_cycle),
        },
        "metal_categories_ms_per_cycle": {
            name: _summary(values)
            for name, values in sorted(
                categories.items(),
                key=lambda item: statistics.fmean(item[1]),
                reverse=True,
            )
        },
        "target_layer_pairs_ms": {
            name: _summary(values)
            for name, values in sorted(layer_pairs.items())
        },
        "per_cycle": per_cycle,
        "command_trace_bytes": COMMAND_LOG.stat().st_size,
        "host_trace_bytes": HOST_LOG.stat().st_size,
        "post_request_safety": {
            "report": post_request_report,
            "failures": post_request_failures,
        },
        "report_path": str(OUTPUT),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(result, indent=2) + "\n"
    OUTPUT.write_text(rendered)
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
