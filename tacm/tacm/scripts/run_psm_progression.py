"""
Forwarding entry point so `cd tacm && python -m tacm.scripts.run_psm_progression` works.

The canonical implementation lives in tacm/scripts/run_psm_progression.py.
"""
import sys
from pathlib import Path

# Ensure the tacm/ directory is on sys.path so `scripts.*` imports resolve
_tacm_dir = Path(__file__).parent.parent.parent
if str(_tacm_dir) not in sys.path:
    sys.path.insert(0, str(_tacm_dir))

from scripts.run_psm_progression import main  # noqa: E402

if __name__ == "__main__":
    main()
