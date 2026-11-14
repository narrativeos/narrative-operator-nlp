"""pytest bootstrap: make the project root importable.

pytest imports this file before any test module, so the test files can
import ``core`` with plain top-level imports (keeping ruff E402 quiet)
instead of each repeating the sys.path.insert boilerplate.
"""

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
