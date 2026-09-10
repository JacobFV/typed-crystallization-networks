"""The record's evidence gate runs with the suite.

`research/record-audit/verify.py --gate` reads committed artifacts only — it never
trains, enumerates, runs a demo or writes an artifact — and takes a few seconds. It
fails when a document claim disagrees with the artifact it cites, when a checked claim
is reworded away, when a section is cited that does not exist, or when a FINDINGS
section has neither a verified claim nor a declared reason it cannot have one. For the
full table, run it directly:

    .venv/bin/python research/record-audit/verify.py --gate --coverage
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_record_gate():
    proc = subprocess.run(
        [sys.executable, str(ROOT / "research" / "record-audit" / "verify.py"), "--gate", "--quiet"],
        cwd=ROOT, capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, "record gate failed:\n" + proc.stdout[-6000:] + proc.stderr[-2000:]
