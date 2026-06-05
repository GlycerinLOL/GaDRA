"""Make the repo-only ``examples`` package importable when running the reproduction test suite.

These tests live outside the installed ``gadra`` wheel and import ``examples.processing`` /
``examples.evaluation`` / ``examples.convert`` from the repo checkout. ``pyproject.toml`` sets ``pythonpath = ["."]`` for the normal
rootdir-based run; this conftest guarantees the repo root is importable regardless of cwd or how pytest is
invoked (e.g. a single file by path), so collection-time ``from examples.* import ...`` always resolves.
"""

import pathlib
import sys

# examples/tests/conftest.py -> examples/ -> GaDRA/ (repo root).
_REPO_ROOT = str(pathlib.Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
