"""Load a saved episode in a fresh interpreter, continue it, report the result.

Used by `replay_check.py` for the cross-process arm of the replay contract.
"""
from __future__ import annotations
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

from tcn.generation import Host  # noqa: E402
from replay_check import torque  # noqa: E402  (same directory, shared action builder)


def main(path):
    host = Host.load(path)
    for _ in range(8):
        host.step((torque(0.4),), dt=0.05)
    print(json.dumps({"digest": host.digest, "integration": host.state["integration"]}))


if __name__ == "__main__":
    main(sys.argv[1])
