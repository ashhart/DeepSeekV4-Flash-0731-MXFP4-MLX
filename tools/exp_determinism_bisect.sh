#!/bin/sh
# Bisect the warm-run nondeterminism found in exp_commit_gap take 3
# (three warm speculative runs at temperature 0 produced three different
# token hashes; same-config B1 != B2, so the A/B lever is not the cause).
#
# Ladder, one server restart + one guarded model load per config:
#   C0  cache_async OFF, layer_async OFF   -> must be deterministic
#   C1  cache_async ON,  layer_async OFF
#   C2  cache_async OFF, layer_async ON (decode=off, multi=2 = production)
#
# Per config: 1 priming run (cold prefill; RAM prompt cache dies with the
# server) then 3 warm runs whose hashes are compared. qkv_rope and decode_moe
# stay at production settings throughout; if C0 is already nondeterministic
# they are the next round's suspects.
#
# All production markers are saved on entry and restored on ANY exit.
set -u
LOG=/Users/ash/ds4/exp_determinism.log
PYROOT=/Applications/oMLX.app/Contents/Resources
PY="$PYROOT/Python/cpython-3.11/bin/python3.11"
export PYTHONPATH="$PYROOT:$PYROOT/Python/framework-mlx-base/lib/python3.11/site-packages:/Users/ash/.local/lib/python3.11/site-packages:/Users/ash/ds4"
OMLX="$HOME/.omlx"
SAVE=/Users/ash/ds4/marker_backup

log() { echo "[bisect] $(date '+%H:%M:%S') $*" >> "$LOG"; }

save_markers() {
    rm -rf "$SAVE"; mkdir -p "$SAVE"
    for m in ds4_async_cache ds4_layer_async ds4_qkv_rope ds4_spec_enabled \
             ds4_spec_timing ds4_consist_batched; do
        [ -e "$OMLX/$m" ] && cp "$OMLX/$m" "$SAVE/$m"
    done
    ls "$SAVE" >> "$LOG" 2>&1
}

restore_markers() {
    for m in ds4_async_cache ds4_layer_async ds4_qkv_rope ds4_spec_enabled \
             ds4_spec_timing ds4_consist_batched; do
        if [ -e "$SAVE/$m" ]; then cp "$SAVE/$m" "$OMLX/$m"; else rm -f "$OMLX/$m"; fi
    done
}

restart_server() {
    pkill -9 -f "omlx-server" 2>/dev/null
    pkill -9 -f "oMLX" 2>/dev/null
    sleep 8
    "$HOME/.omlx/bin/omlx" start >> "$LOG" 2>&1
    i=0
    until curl -s -m 5 -o /dev/null -H "Connection: close" http://127.0.0.1:8000/v1/models \
        -H "Authorization: Bearer $(python3 -c "import json,pathlib;print(json.loads((pathlib.Path.home()/'.omlx/settings.json').read_text())['auth']['api_key'])")"; do
        i=$((i+1)); [ $i -gt 24 ] && { log "SERVER FAILED TO START"; return 1; }
        sleep 5
    done
    # Let the health-check sockets drain: the model-load guard refuses on ANY
    # active connection, and our own keep-alive probe tripped it (C0, take 1).
    sleep 10
    return 0
}

cleanup() {
    log "cleanup: restoring production markers, unloading"
    restore_markers
    "$PY" /Users/ash/ds4/tools/model_control.py unload >> "$LOG" 2>&1
    restart_server
    "$PY" /Users/ash/ds4/tools/studio_guard.py >> "$LOG" 2>&1
    log "cleanup done"
}
trap cleanup EXIT

: > "$LOG"
save_markers
log "guard (expect clean)"
"$PY" /Users/ash/ds4/tools/studio_guard.py >> "$LOG" 2>&1 || { log "GUARD REFUSED"; exit 1; }

one_request() {  # $1 = label
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
print(f"HASH {label} wall={dt:.2f}s "
      f"cached={u.get('prompt_tokens_details',{}).get('cached_tokens',0)} "
      f"sha={hashlib.sha256(content.encode()).hexdigest()[:16]}")
EOF
}

run_config() {  # $1 = name, $2 = async_cache(0/1), $3 = layer_async(0/1)
    log "=== config $1 (async_cache=$2 layer_async=$3) ==="
    if [ "$2" = "1" ]; then touch "$OMLX/ds4_async_cache"; else rm -f "$OMLX/ds4_async_cache"; fi
    if [ "$3" = "1" ]; then printf 'decode=off\nmulti=2\n' > "$OMLX/ds4_layer_async"; else rm -f "$OMLX/ds4_layer_async"; fi
    touch "$OMLX/ds4_spec_enabled" "$OMLX/ds4_spec_timing"
    rm -f "$OMLX/ds4_consist_batched"

    restart_server || return 1
    tries=0
    until "$PY" /Users/ash/ds4/tools/model_control.py load >> "$LOG" 2>&1; do
        tries=$((tries+1))
        [ $tries -ge 3 ] && { log "LOAD FAILED ($1) after $tries tries"; return 1; }
        log "load refused ($1), retry $tries after drain"
        sleep 15
    done

    one_request "$1-prime"
    one_request "$1-w1"
    one_request "$1-w2"
    one_request "$1-w3"

    "$PY" /Users/ash/ds4/tools/model_control.py unload >> "$LOG" 2>&1
}

run_config C0 0 0 || exit 1
run_config C1 1 0 || exit 1
run_config C2 0 1 || exit 1

log "bisect complete"
grep "HASH" "$LOG" >> "$LOG".summary 2>/dev/null || true
exit 0
