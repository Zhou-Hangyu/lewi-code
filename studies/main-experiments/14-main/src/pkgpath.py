"""Put the benchmark package of this archive (src/) first on the path.

Every 14-main tool that imports ``csfrbench`` (directly or through the
bundle tools) imports this module first, so the campaign runs the v1.1
package (``versions/v1.1/implementation``) whether or not the repository
source has been synced yet (``versions/v1.1/src/sync_src.sh``); the path is
also exported into ``PYTHONPATH`` so subprocesses inherit it.

    import pkgpath  # noqa: F401
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
PKG = str(REPO / "src")

if sys.path[:1] != [PKG]:
    sys.path[:] = [PKG] + [p for p in sys.path if p != PKG]
_pp = os.environ.get("PYTHONPATH", "")
if PKG not in _pp.split(os.pathsep):
    os.environ["PYTHONPATH"] = PKG + (os.pathsep + _pp if _pp else "")
if "csfrbench" in sys.modules:
    _loaded = Path(sys.modules["csfrbench"].__file__).resolve()
    if PKG not in str(_loaded):
        raise ImportError(f"csfrbench was imported from {_loaded} before pkgpath; "
                          f"import pkgpath first (expected {PKG})")
