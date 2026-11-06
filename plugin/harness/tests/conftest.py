import sys
from pathlib import Path

# Make `contracts` and `harness` importable as top-level modules when running pytest
# from the repo root, the same way plugin/forecast/tests does.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
