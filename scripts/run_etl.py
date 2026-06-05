"""CLI entry point for the ETL pipeline (runnable before pip install -e .)."""

import sys
from pathlib import Path

# Ensure project root is on path when running as a plain script
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.etl import main_cli

if __name__ == "__main__":
    main_cli()
