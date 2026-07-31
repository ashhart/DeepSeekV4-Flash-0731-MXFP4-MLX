#!/bin/sh
# Install the DS4 speed patches for DeepSeek-V4-Flash-0731-MXFP4-MLX under oMLX.
#
# Only ADDS files. Nothing inside /Applications/oMLX.app is modified, so an oMLX
# update can remove the hook but never conflict with it -- just re-run this.
#
#   ./install.sh              install
#   ./install.sh --verify     check the patches load and apply
#   ./install.sh --uninstall  remove
set -eu

OMLX="/Applications/oMLX.app/Contents/Resources"
PY="$OMLX/Python/cpython-3.11/bin/python3.11"
SITE="$OMLX/Python/framework-mlx-base/lib/python3.11/site-packages"
DEST="${DS4_HOME:-$HOME/ds4}"
# `.pth` files only execute in real site directories -- a PYTHONPATH entry will
# not do. oMLX's interpreter reports ENABLE_USER_SITE=True with this path, so
# hooking here stays entirely outside /Applications.
USER_SITE="$HOME/.local/lib/python3.11/site-packages"
PTH="$USER_SITE/zz-ds4-patches.pth"   # sorts late: runs after anything else

case "${1:-}" in
--uninstall)
    rm -f "$PTH"
    echo "removed $PTH"
    echo "(left $DEST in place; delete it manually if you want)"
    echo "restart oMLX:  pkill -f 'oMLX|omlx-server'; open -a oMLX"
    exit 0
    ;;
--verify)
    [ -f "$PTH" ] || { echo "NOT installed: $PTH missing" >&2; exit 1; }
    PYTHONPATH="$OMLX:$SITE" "$PY" - <<'EOF'
import sys
assert "ds4_boot" in sys.modules, "the .pth hook did not run"
import omlx.patches.deepseek_v4 as p
assert getattr(p.apply_deepseek_v4_patch, "_ds4_wrapped", False), "wrapper missing"
p.apply_deepseek_v4_patch()
import mlx_lm.models.deepseek_v4 as d
for cls in ("LocalAttention", "CompressedAttention"):
    got = getattr(d, cls).__call__.__qualname__
    assert "apply.<locals>" in got, f"{cls} not patched (got {got})"
    print(f"  {cls:<22} -> {got}")
print("OK: windowed prefill is active")
EOF
    exit 0
    ;;
esac

[ -d "$OMLX" ] || { echo "oMLX not found at /Applications/oMLX.app" >&2; exit 1; }

SRC="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$DEST" "$USER_SITE"
rm -rf "$DEST/ds4"
cp -R "$SRC/ds4" "$DEST/"
# boot.py is imported by name from the .pth, so it also sits at the top level.
cp "$SRC/ds4/boot.py" "$DEST/ds4_boot.py"
echo "installed code -> $DEST"

printf "import sys; sys.path.insert(0, '%s'); import ds4_boot\n" "$DEST" > "$PTH"
echo "installed hook -> $PTH"

cat <<EOF

Restart oMLX, then verify:
  pkill -f 'oMLX|omlx-server'; open -a oMLX
  $SRC/install.sh --verify

Recommended, in ~/.omlx/settings.json:
  "scheduler": { "chunked_prefill": false }
EOF
