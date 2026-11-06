import sys
from pathlib import Path

# Make `forecast` importable as a top-level package when running pytest from
# the repo root (Hermes imports it as part of the plugin package instead).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
