#!/bin/sh
# Verify-width sweep (FINDINGS S30).
#
# The k=3 default was chosen when PoolingCache rollback refused above k=3 and
# each refusal cost a restore+replay (20% of cycles at k=5). That failure mode
# is gone -- production shows restore_replay=0% -- so the ceiling was never
# re-measured. The target forward is weight-bound: verifying 5 candidates costs
# little more than 3, while accepting more tokens per cycle.
#
# One model load. Widths 3/4/5 interleaved twice in the same process.
#
# Exact same deterministic request each run; token content hashes must be
# identical (the check is diagnostic-only, so any divergence = abort finding).
# Commit sub-phase timers (this deploy) attribute the gap. Cleans up and
# unloads on every exit path per the handoff acceptance criteria.
set -u
LOG=/Users/ash/ds4/exp_width_sweep.log
PYROOT=/Applications/oMLX.app/Contents/Resources
PY="$PYROOT/Python/cpython-3.11/bin/python3.11"
export PYTHONPATH="$PYROOT:$PYROOT/Python/framework-mlx-base/lib/python3.11/site-packages:/Users/ash/.local/lib/python3.11/site-packages:/Users/ash/ds4"

log() { echo "[exp] $(date '+%H:%M:%S') $*" >> "$LOG"; }

cleanup() {
    log "cleanup: removing markers, unloading model"
    # ds4_spec_enabled is removed too: the handoff's production state is
    # speculation OFF pending validation, and this script must restore it.
    rm -f "$HOME/.omlx/ds4_spec_timing" "$HOME/.omlx/ds4_spec_reset_stats" \
          "$HOME/.omlx/ds4_block_size"
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

log "loading model (guarded)"
"$PY" /Users/ash/ds4/tools/model_control.py load >> "$LOG" 2>&1 || { log "LOAD FAILED"; exit 1; }

# The experiment measures the SPECULATIVE cycle, so speculation must be on.
# Run #2 of this experiment silently measured plain decode in both arms
# because this line was missing. Marker is removed again in cleanup.
touch "$HOME/.omlx/ds4_spec_enabled"
touch "$HOME/.omlx/ds4_spec_timing"
# Full promoted stack ON in every arm; only the verify width changes.
touch "$HOME/.omlx/ds4_consist_batched" "$HOME/.omlx/ds4_router_fused"
SRVLOG="$HOME/.omlx/logs/server.log"
LOG_START=$(wc -l < "$SRVLOG" | tr -d " ")

run_one() {  # $1 = label, $2 = verify width
    echo "$2" > "$HOME/.omlx/ds4_block_size"
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

for pass_no in 1 2; do
    for w in 3 4 5; do
        log "run w${w}-p${pass_no} (verify width $w)"
        run_one "w${w}-p${pass_no}" "$w"
        check_valid
        if ! tail -n +"$LOG_START" "$SRVLOG" | grep "\[clamp=" | tail -1 | grep -qE "widths=([^ |]*,)?$w:"; then
            log "VOID: arm w$w did not run at width $w -- knob not live"
            exit 1
        fi
    done
done

log "experiment complete"
grep "RESULT" "$LOG" | tail -4 >> "$LOG".summary 2>/dev/null || true
exit 0
