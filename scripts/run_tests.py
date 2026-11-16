from __future__ import annotations
import os
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
env = os.environ.copy()
src = str(root / "src")
env["PYTHONPATH"] = src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
raise SystemExit(subprocess.call([sys.executable, "-m", "pytest", "-q", "tests"], cwd=root, env=env))
