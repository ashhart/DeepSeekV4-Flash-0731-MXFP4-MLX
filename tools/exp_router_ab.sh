#!/bin/sh
# Attack #1: split and shrink the partial-rollback commit gap.
#
# One model load. Interleaved A/B/A/B in the same server process:
#   A = stock per-item _offsets_consistent (up to ~41 .item() syncs post-trim)
#   B = batched single-eval variant (marker ~/.omlx/ds4_router_fused)
#
# Exact same deterministic request each run; token content hashes must be
# identical (the check is diagnostic-only, so any divergence = abort finding).
# Commit sub-phase timers (this deploy) attribute the gap. Cleans up and
# unloads on every exit path per the handoff acceptance criteria.
set -u
LOG=/Users/ash/ds4/exp_router_ab.log
PYROOT=/Applications/oMLX.app/Contents/Resources
PY="$PYROOT/Python/cpython-3.11/bin/python3.11"
export PYTHONPATH="$PYROOT:$PYROOT/Python/framework-mlx-base/lib/python3.11/site-packages:/Users/ash/.local/lib/python3.11/site-packages:/Users/ash/ds4"

log() { echo "[exp] $(date '+%H:%M:%S') $*" >> "$LOG"; }

cleanup() {
    log "cleanup: removing markers, unloading model"
    # ds4_spec_enabled is removed too: the handoff's production state is
    # speculation OFF pending validation, and this script must restore it.
    rm -f "$HOME/.omlx/ds4_spec_timing" "$HOME/.omlx/ds4_router_fused" \
          "$HOME/.omlx/ds4_spec_reset_stats" "$HOME/.omlx/ds4_spec_enabled" \
          "$HOME/.omlx/ds4_consist_batched"
    "$PY" /Users/ash/ds4/tools/model_control.py unload >> "$LOG" 2>&1
    "$PY" /Users/ash/ds4/tools/studio_guard.py >> "$LOG" 2>&1
    log "cleanup done"
}
trap cleanup EXIT

: > "$LOG"
log "guard (expect clean)"
"$PY" /Users/ash/ds4/tools/studio_guard.py >> "$LOG" 2>&1 || { log "GUARD REFUSED"; exit 1; }

log "loading model (guarded)"
"$PY" /Users/ash/ds4/tools/model_control.py load >> "$LOG" 2>&1 || { log "LOAD FAILED"; exit 1; }

# The experiment measures the SPECULATIVE cycle, so speculation must be on.
# Run #2 of this experiment silently measured plain decode in both arms
# because this line was missing. Marker is removed again in cleanup.
touch "$HOME/.omlx/ds4_spec_enabled"
touch "$HOME/.omlx/ds4_spec_timing"
# Confirmed consist fix stays ON in both arms so the router A/B is clean.
touch "$HOME/.omlx/ds4_consist_batched"
SRVLOG="$HOME/.omlx/logs/server.log"
LOG_START=$(wc -l < "$SRVLOG" | tr -d " ")

run_one() {  # $1 = label, $2 = batched(0/1)
    if [ "$2" = "1" ]; then touch "$HOME/.omlx/ds4_router_fused"; else rm -f "$HOME/.omlx/ds4_router_fused"; fi
    touch "$HOME/.omlx/ds4_spec_reset_stats"
    "$PY" - "$1" >> "$LOG" 2>&1 <<'EOF'
import hashlib, json, pathlib, sys, time, urllib.request
label = sys.argv[1]
key = json.loads((pathlib.Path.home()/".omlx/settings.json").read_text())["auth"]["api_key"]
ctx_parts = sorted(pathlib.Path("/Users/ash/ds4/ds4").glob("*.py"))
ctx = "".join(p.read_text() for p in ctx_parts)[:85000]
prompt = ctx + "\n\nExplain, in detail, how a KV cache works in transformer inference."
body = json.dumps({"model": "DeepSeek-V4-Flash-0731-MXFP4-MLX",
    "messages": [{"role": "user", "content": prompt}],
    "max_tokens": 260, "temperature": 0}).encode()
req = urllib.request.Request("http://127.0.0.1:8000/v1/chat/completions", data=body,
    headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
t0 = time.time()
resp = json.load(urllib.request.urlopen(req, timeout=1800))
dt = time.time() - t0
content = resp["choices"][0]["message"].get("content") or ""
u = resp["usage"]
print(f"RESULT {label} wall={dt:.2f}s completion={u['completion_tokens']} "
      f"cached={u.get('prompt_tokens_details',{}).get('cached_tokens',0)} "
      f"sha={hashlib.sha256(content.encode()).hexdigest()[:16]}")
EOF
    sleep 2
    # Only lines emitted since this script started, and only the NEW format --
    # "[clamp=" exists solely in the sub-phase build. An old-format or absent
    # line means stale code or zero speculative cycles: the run proved nothing.
    tail -n +"$LOG_START" "$SRVLOG" | grep "\[clamp=" | tail -1 >> "$LOG"
}

check_valid() {
    if ! tail -n +"$LOG_START" "$SRVLOG" | grep -q "\[clamp="; then
        log "VOID: no new-format stats line -- stale code or speculation off"
        exit 1
    fi
}

log "run A1 (stock router)";   run_one A1 0
check_valid
log "run B1 (fused router)";   run_one B1 1
if ! tail -n +"$LOG_START" "$SRVLOG" | grep -q "fused router ENGAGED"; then
    log "VOID: fused router never engaged in B1 -- lever not wired"
    exit 1
fi
log "run A2 (stock router)";   run_one A2 0
log "run B2 (fused router)";   run_one B2 1

log "experiment complete"
grep "RESULT" "$LOG" | tail -4 >> "$LOG".summary 2>/dev/null || true
exit 0
