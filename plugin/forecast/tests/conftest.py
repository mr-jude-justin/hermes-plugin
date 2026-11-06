import sys
from pathlib import Path

# Make `forecast` and `contracts` importable as top-level modules when running
# pytest from the repo root (Hermes imports them as part of the plugin package).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
