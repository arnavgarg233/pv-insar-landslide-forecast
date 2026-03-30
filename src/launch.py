"""
Default training launcher: run a script under `Models/` from the repository root.

Usage (from repo root):
  python src/launch.py
  python src/launch.py --script Models/model_e_lda.py
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch a training or evaluation script under Models/.")
    parser.add_argument(
        "--script",
        default="Models/model_b_simple.py",
        help="Path relative to repository root (default: main RF baseline).",
    )
    args = parser.parse_args()
    target = (_REPO / args.script).resolve()
    if not target.is_file():
        print(f"Script not found: {target}", file=sys.stderr)
        return 1
    return subprocess.call([sys.executable, str(target)], cwd=str(_REPO))


if __name__ == "__main__":
    raise SystemExit(main())
