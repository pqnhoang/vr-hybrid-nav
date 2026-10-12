import sys
from pathlib import Path

for folder in ("planner", "exploration"):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / folder))
