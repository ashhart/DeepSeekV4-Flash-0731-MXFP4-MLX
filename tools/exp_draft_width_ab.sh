#!/bin/sh
# Draft-width sweep 7 vs 8 (FINDINGS S37).
#
# layer_async (decode=off, multi=2) forces an async_eval every 2 layers -- 21
# extra command-buffer flushes per forward. The cycle is dispatch-bound
# (hyper-connections run at 0.4% of roofline, pure launch latency), so those
# flushes may cost more than the overlap they buy. It has been ON in production
# since the handoff on a +16.75% claim; the only later signal (C2, -3.5x) was
# confounded by the box degrading mid-run, so this has never been cleanly
# measured on a healthy machine.
#
# One model load, ON/OFF interleaved twice, arm-validity gated.
set -u
LOG=/Users/ash/ds4/exp_draft_width_ab.log
PYROOT=/Applications/oMLX.app/Contents/Resources
PY="$PYROOT/Python/cpython-3.11/bin/python3.11"
export PYTHONPATH="$PYROOT:$PYROOT/Python/framework-mlx-base/lib/python3.11/site-packages:/Users/ash/.local/lib/python3.11/site-packages:/Users/ash/ds4"

log() { echo "[exp] $(date '+%H:%M:%S') $*" >> "$LOG"; }

cleanup() {
    log "cleanup: removing markers, unloading model"
    # ds4_spec_enabled is removed too: the handoff's production state is
    # speculation OFF pending validation, and this script must restore it.
    rm -f "$HOME/.omlx/ds4_spec_timing" "$HOME/.omlx/ds4_spec_reset_stats" \
          "$HOME/.omlx/ds4_draft_width"
    touch "$HOME/.omlx/ds4_spec_enabled" "$HOME/.omlx/ds4_consist_batched" \
          "$HOME/.omlx/ds4_router_fused"
    "$PY" /Users/ash/ds4/tools/model_control.py unload >> "$LOG" 2>&1
    "$PY" /Users/ash/ds4/tools/studio_guard.py >> "$LOG" 2>&1
    log "cleanup done"
}
trap cleanup EXIT

: > "$LOG"
log "guard (expect clean)"
"$PY" /Users/ash/ds4/tools/studio_guard.py >> "$LOG" 2>&1 || { log "GUARD REFUSED"; exit 1; }

# Each arm loads and unloads its own drafter (width is fixed at load).

# The experiment measures the SPECULATIVE cycle, so speculation must be on.
# Run #2 of this experiment silently measured plain decode in both arms
# because this line was missing. Marker is removed again in cleanup.
touch "$HOME/.omlx/ds4_spec_enabled"
touch "$HOME/.omlx/ds4_spec_timing"
# Full promoted stack ON in every arm; only the verify width changes.
touch "$HOME/.omlx/ds4_consist_batched" "$HOME/.omlx/ds4_router_fused"
SRVLOG="$HOME/.omlx/logs/server.log"
LOG_START=$(wc -l < "$SRVLOG" | tr -d " ")

run_one() {  # $1 = label, $2 = draft width
    echo "$2" > "$HOME/.omlx/ds4_draft_width"
    "$PY" /Users/ash/ds4/tools/model_control.py load >> "$LOG" 2>&1 || {
        log "LOAD FAILED ($1)"; return 1; }
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
    # Per-arm proof AFTER the request: the drafter is built lazily on the first
    # speculative cycle, not at load, so checking earlier reads the PREVIOUS
    # arm's line.
    if ! grep "DSpark drafter loaded" /Users/ash/ds4/serve_headless.log \
         | tail -1 | grep -q "draft_width=$2"; then
        log "VOID: arm $1 did not build the drafter at width $2"
        "$PY" /Users/ash/ds4/tools/model_control.py unload >> "$LOG" 2>&1
        return 1
    fi
    "$PY" /Users/ash/ds4/tools/model_control.py unload >> "$LOG" 2>&1
}

check_valid() {
    if ! tail -n +"$LOG_START" "$SRVLOG" | grep -q "\[clamp="; then
        log "VOID: no new-format stats line -- stale code or speculation off"
        exit 1
    fi
}

for pass_no in 1 2; do
    for w in 7 8; do
        log "run w${w}-p${pass_no} (draft width $w)"
        run_one "w${w}-p${pass_no}" "$w" || exit 1
        check_valid
    done
done

log "experiment complete"
grep "RESULT" "$LOG" | tail -4 >> "$LOG".summary 2>/dev/null || true
exit 0
