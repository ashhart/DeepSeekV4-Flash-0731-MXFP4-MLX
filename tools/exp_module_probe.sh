#!/bin/sh
# Attention-vs-MoE attribution inside the target layer (FINDINGS S31 follow-up).
#
# The cycle is  42.4 ms base + 6.95 ms x verify_width.  The base is weight
# reads at ~19% of roofline and is the only term worth a kernel. MoE is 62% of
# active BYTES -- this measures whether it is 62% of the TIME.
#
# Diagnostic only: the probe forces ~86 extra evals per forward, so absolute
# times inflate. Only the attn:ffn RATIO is read.
set -u
LOG=/Users/ash/ds4/exp_module_probe.log
PYROOT=/Applications/oMLX.app/Contents/Resources
PY="$PYROOT/Python/cpython-3.11/bin/python3.11"
export PYTHONPATH="$PYROOT:$PYROOT/Python/framework-mlx-base/lib/python3.11/site-packages:/Users/ash/.local/lib/python3.11/site-packages:/Users/ash/ds4:/Users/ash/ds4/tools"

log() { echo "[probe] $(date '+%H:%M:%S') $*" >> "$LOG"; }

cleanup() {
    log "cleanup: restoring production markers, unloading"
    rm -f "$HOME/.omlx/ds4_module_probe" "$HOME/.omlx/ds4_metal_probe" \
          "$HOME/.omlx/ds4_spec_profile_sync" "$HOME/.omlx/ds4_spec_timing" \
          "$HOME/.omlx/ds4_spec_reset_stats"
    touch "$HOME/.omlx/ds4_spec_enabled" "$HOME/.omlx/ds4_consist_batched" \
          "$HOME/.omlx/ds4_router_fused"
    "$PY" /Users/ash/ds4/tools/model_control.py unload >> "$LOG" 2>&1
    log "cleanup done"
}
trap cleanup EXIT

: > "$LOG"
log "guard"
"$PY" /Users/ash/ds4/tools/studio_guard.py >> "$LOG" 2>&1 || { log "GUARD REFUSED"; exit 1; }

# Probe markers must be set BEFORE the load so the backend opens its trace logs.
touch "$HOME/.omlx/ds4_metal_probe" "$HOME/.omlx/ds4_spec_profile_sync" \
      "$HOME/.omlx/ds4_spec_timing" "$HOME/.omlx/ds4_spec_enabled" \
      "$HOME/.omlx/ds4_consist_batched" "$HOME/.omlx/ds4_router_fused" \
      "$HOME/.omlx/ds4_module_probe"

log "loading model (guarded)"
"$PY" /Users/ash/ds4/tools/model_control.py load >> "$LOG" 2>&1 || { log "LOAD FAILED"; exit 1; }

if ! grep -q "module probe installed" /Users/ash/ds4/serve_headless.log; then
    log "VOID: module probe not installed at boot"
    exit 1
fi

log "profiling"
"$PY" /Users/ash/ds4/tools/profile_server_cycle.py >> "$LOG" 2>&1 || { log "PROFILE FAILED"; exit 1; }

"$PY" - >> "$LOG" 2>&1 <<'EOF'
import json, pathlib
d = json.load(open(pathlib.Path.home() / "ds4/profiles/server_cycle_metal_ledger.json"))
pairs = d.get("target_layer_pairs_ms", {})
kinds = ["attn", "ffn", "hc", "idx"]
got = {k: pairs.get(f"layers_all_{k}", {}).get("median", 0.0) for k in kinds}
other = sum(v.get("median", 0.0) for k, v in pairs.items()
            if k not in [f"layers_all_{k}" for k in kinds])
total = sum(got.values()) + other
if total <= 0:
    print("VOID: no labelled work -- probe never fired")
else:
    parts = "  ".join(f"{k}={v:.2f}" for k, v in got.items())
    print(f"SPLIT {parts}  other={other:.2f}  total={total:.2f} ms")
    for k, v in sorted(got.items(), key=lambda kv: -kv[1]):
        print(f"SPLIT   {k:5s} {v:6.2f} ms  {v/total*100:4.0f}% of layer time")
EOF
grep -E "^SPLIT|^VOID" "$LOG" >> "$LOG".summary 2>/dev/null || true
log "experiment complete"
exit 0
