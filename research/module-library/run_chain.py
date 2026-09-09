"""Run the three-stage chain through the shipped curriculum orchestrator."""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))

from runner import stage_runner  # noqa: E402
from tcn.curriculum import Curriculum  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default=str(HERE / "chain.json"))
    ap.add_argument("--out", default=str(HERE / "out" / "chain"))
    ap.add_argument("--library", default=str(HERE / "library"))
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()
    records = Curriculum.load(args.spec).run(stage_runner, args.out, workers=1,
                                             resume=not args.fresh, library=args.library)
    summary = {k: {"status": v["status"], "failures": v.get("failures", []),
                   "published": v.get("published", []), "inherited": v.get("inherited", [])}
               for k, v in records.items()}
    print(json.dumps(summary, indent=2))
    return 0 if all(v["status"] == "passed" for v in records.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
