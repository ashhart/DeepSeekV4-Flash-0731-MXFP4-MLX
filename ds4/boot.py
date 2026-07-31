# SPDX-License-Identifier: MIT
"""Install the DS4 patches into any oMLX process, without editing the app bundle.

Loaded at interpreter startup by `ds4_patches.pth` in site-packages. At that
point `omlx` is not imported yet and `mlx_lm.models.deepseek_v4` does not exist
(oMLX synthesises it at model-load time), so this registers a `sys.meta_path`
finder that waits for `omlx.patches.deepseek_v4` to be imported and wraps its
`apply_deepseek_v4_patch` -- which oMLX calls on every model load -- so our
patches land immediately afterwards, against the classes it just registered.

Adding a `.pth` file touches nothing oMLX ships, so an app update can only remove
it, never conflict with it. Re-run `integration/install.sh` after an update.

Disable with DS4_PATCHES=0.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import os
import sys

TARGET = "omlx.patches.deepseek_v4"
_log = lambda m: print(f"[ds4] {m}", file=sys.stderr)  # noqa: E731


def _install_patches(module) -> None:
    """Wrap `apply_deepseek_v4_patch` so our patches follow oMLX's."""
    original = getattr(module, "apply_deepseek_v4_patch", None)
    if original is None or getattr(original, "_ds4_wrapped", False):
        return

    def wrapped(*args, **kwargs):
        result = original(*args, **kwargs)
        try:
            from ds4 import windowed_prefill

            if windowed_prefill.apply():
                _log(
                    "windowed prefill enabled "
                    f"(block={windowed_prefill.DEFAULT_BLOCK}, "
                    f"min_len={windowed_prefill.MIN_PREFILL_LEN})"
                )
        except Exception as e:  # noqa: BLE001 -- never break model loading
            _log(f"windowed prefill NOT applied: {type(e).__name__}: {e}")

        try:
            from ds4 import engine_hook

            if engine_hook.apply():
                state = "ON" if engine_hook.enabled() else "off (set DS4_SPEC=1)"
                _log(f"DSpark speculative decoding hook installed, {state}")
        except Exception as e:  # noqa: BLE001
            _log(f"DSpark hook NOT installed: {type(e).__name__}: {e}")
        return result

    wrapped._ds4_wrapped = True
    module.apply_deepseek_v4_patch = wrapped


class _Finder(importlib.abc.MetaPathFinder):
    """Delegate to the normal finders, then post-process the target module."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname != TARGET:
            return None
        # Step aside so the real finders resolve it, then wrap the loader.
        sys.meta_path.remove(self)
        try:
            spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        finally:
            if self not in sys.meta_path:
                sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return None

        inner = spec.loader

        class _Loader(importlib.abc.Loader):
            def create_module(self, spec):
                return inner.create_module(spec)

            def exec_module(self, module):
                inner.exec_module(module)
                _install_patches(module)

        spec.loader = _Loader()
        return spec


def install() -> None:
    if os.environ.get("DS4_PATCHES") == "0":
        return
    if any(isinstance(f, _Finder) for f in sys.meta_path):
        return
    # Already imported (e.g. re-entrant import): patch in place.
    if TARGET in sys.modules:
        _install_patches(sys.modules[TARGET])
        return
    sys.meta_path.insert(0, _Finder())


install()
