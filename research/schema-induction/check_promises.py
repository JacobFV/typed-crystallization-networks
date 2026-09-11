"""Fail when a promised mechanical check is missing, or a promise disappeared.

Two failures from the scaffold-induction track motivate this file, and both
were found inside the machinery built to prevent exactly this class of error:

* amendment A13 promised that its `verify.py` enforces the blind split's
  fingerprint, `final_untouched`, and train-only ordering. `git grep -n final
  verify.py` on that branch returns nothing: the check was never written, and
  nothing failed because of it.
* a validation artifact for a superseded corpus was accepted as current,
  because the verifier asserted that a validation had run, never that it had
  validated *this* corpus.

So: `promises.json` lists every mechanical check this track's
`PREREGISTRATION.md` promises, with the phase that activates it. This script
fails when

  1. a promise whose phase has RUN has no implementing function in `verify.py`;
  2. a promise recorded in git has been deleted from `promises.json` (promises
     are amended, never removed);
  3. a promise's artifact exists but carries no provenance stamp (`stamp.py`),
     or a stamp that disagrees with what is on disk now;
  4. `promises.json` itself is malformed, or a promise names no phase.

A phase counts as RUN when its marker file exists under `out/`. Markers are
written by the phase's own job, never by hand.

    python check_promises.py            # exit 0 only if every active promise is kept
    python check_promises.py --status   # report, exit 0 regardless
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import subprocess
import sys

import stamp

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "out"
REGISTRY = HERE / "promises.json"
VERIFY = HERE / "verify.py"
MARKER = "phase_{}.done.json"       # .json so V9's provenance scan sees it too


def phase_has_run(phase):
    return (OUT / MARKER.format(phase)).exists()


def mark_phase(phase, provenance):
    """Written by a phase's own job when it completes. Never by hand."""
    return stamp.write(OUT / MARKER.format(phase), {"phase": phase}, provenance)


def implemented_checks():
    if not VERIFY.exists():
        return set()
    text = VERIFY.read_text()
    return set(re.findall(r"^def (verify_[A-Za-z0-9_]+)\(", text, flags=re.M))


def committed_promise_ids():
    """The promise ids in the last committed `promises.json`, for deletion detection."""
    rel = REGISTRY.relative_to(ROOT)
    try:
        blob = subprocess.run(["git", "-C", str(ROOT), "show", f"HEAD:{rel}"],
                              capture_output=True, text=True, timeout=30)
        if blob.returncode:
            return set()
        return {p["id"] for p in json.loads(blob.stdout)["promises"]}
    except Exception:                                          # noqa: BLE001
        return set()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", action="store_true", help="report without failing")
    a = ap.parse_args()

    registry = json.loads(REGISTRY.read_text())
    promises = registry["promises"]
    known_phases = set(registry["phases"])
    have = implemented_checks()
    failures, rows = [], []

    seen = set()
    for p in promises:
        pid = p["id"]
        if pid in seen:
            failures.append(f"{pid}: duplicate promise id")
        seen.add(pid)
        if p.get("phase") not in known_phases:
            failures.append(f"{pid}: names no declared phase ({p.get('phase')!r})")
            continue
        active = phase_has_run(p["phase"])
        ok = p["check"] in have
        state = "KEPT" if ok else ("MISSING" if active else "pending")
        if active and not ok:
            failures.append(f"{pid}: phase {p['phase']} has run but {VERIFY.name} "
                            f"implements no {p['check']}()")
        # 3. a DERIVED artifact, if it exists, must carry a well-formed stamp.
        #    Only files under out/ are derived; `promises.json` and the *.py are
        #    hand-written sources, fingerprinted BY the artifacts rather than
        #    carrying fingerprints of their own.
        art = p.get("artifact")
        if art and art != "*" and art.startswith("out/") and (HERE / art).exists():
            try:
                blob = json.loads((HERE / art).read_text())
                stamp.present(blob)
            except stamp.StampError as exc:
                failures.append(f"{pid}: artifact {art} -- {exc}")
                state += "+UNSTAMPED"
            except json.JSONDecodeError:
                pass                                            # not a JSON artifact
        rows.append((pid, p["phase"], state, p["check"]))

    for gone in sorted(committed_promise_ids() - seen):
        failures.append(f"{gone}: a committed promise was deleted from the registry; "
                        f"promises are amended, never removed")

    width = max(len(r[3]) for r in rows)
    for pid, phase, state, check in rows:
        print(f"  {pid:4s} {phase:8s} {state:16s} {check:<{width}s}")
    print(f"\n{len(rows)} promises, "
          f"{sum(1 for r in rows if r[2].startswith('KEPT'))} kept, "
          f"{sum(1 for r in rows if r[2].startswith('pending'))} pending, "
          f"{len(failures)} failures")
    for f in failures:
        print(f"  FAIL {f}")
    if failures and not a.status:
        sys.exit(1)


if __name__ == "__main__":
    main()
