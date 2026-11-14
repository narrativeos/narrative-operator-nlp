"""Repo bootstrap for the example scripts in this directory.

Import this *before* ``core`` (top of file) when running a script as
``python examples/<name>.py``: the script dir is what Python puts on
sys.path, not the repo root, so the root has to be inserted first.
Doing it here keeps the callers' imports all at module top level
(no E402 between the bootstrap and ``from core.analyzer import ...``).
"""

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
