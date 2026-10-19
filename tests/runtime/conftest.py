import sys
from pathlib import Path

for folder in ("planner", "exploration", "move"):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / folder))
