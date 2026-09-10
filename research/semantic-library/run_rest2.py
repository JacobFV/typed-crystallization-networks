"""The gradient, held-out and check stages, after the modules are published."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import _paths  # noqa: F401

HERE = Path(__file__).resolve().parent
PY = sys.executable


def sh(*args):
    print(">>", " ".join(args), flush=True)
    subprocess.run([PY, str(HERE / args[0])] + list(args[1:]), check=True)


if __name__ == "__main__":
    sh("run_grad.py", "tight", "24")
    sh("run_grad.py", "wide", "8")
    sh("run_heldout.py", "all")
    sh("run_checks.py")
