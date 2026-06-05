"""CLI entry point for the equity regime research pipeline."""

import sys
from pathlib import Path

# Ensure src is on path when running as a script (before pip install -e .)
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from equity_regime.pipeline import main_cli

if __name__ == "__main__":
    main_cli()
