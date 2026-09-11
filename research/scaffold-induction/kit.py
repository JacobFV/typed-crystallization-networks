"""Shared paths, the capped-resource helper, and JSON dump/load.

Nothing under `tcn/` or `generators/` is imported for modification; this track
reads the core and the shipped generators only.
"""
from __future__ import annotations

import json
import os
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent
OUT = HERE / "out"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def dump(name, obj):
    OUT.mkdir(exist_ok=True)
    path = OUT / f"{name}.json"
    path.write_text(json.dumps(obj, indent=1, sort_keys=True, default=str))
    return str(path)


def load(name):
    return json.loads((OUT / f"{name}.json").read_text())


def mem_available_gb():
    for line in pathlib.Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / (1024 * 1024)
    raise RuntimeError("MemAvailable not found")


# --- the memory floor (amended; see RESULTS section 10 -------------------------
#
# The original rule was a flat 25 GB of MemAvailable before any phase.  It was
# written for heavy phases and it cannot tell a 50 MB job from a 50 GB one, so
# on a host loaded by *other* projects it blocks work that could not possibly
# tip the host: this track's heaviest phase measures 0.025 GB.  The rule now
# scales to the phase's own measured peak:
#
#     a phase may start when MemAvailable >= max(2 GB, 20 x measured peak RSS)
#     and at least 1 GB of headroom remains once the phase is resident.
#     A phase with no measured peak keeps the original 25 GB floor.
#
# The 20x multiple leaves two orders of magnitude of slack over the measurement;
# the 2 GB minimum keeps a floor even for a trivial job; the 1 GB headroom check
# is what actually protects a host that is already nearly full.
FLOOR_UNMEASURED_GB = 25.0
FLOOR_MIN_GB = 2.0
FLOOR_MULTIPLE = 20
FLOOR_HEADROOM_GB = 1.0


def required_floor(peak_gb=None):
    """(required MemAvailable, why) for a phase with this measured peak."""
    if peak_gb is None:
        return FLOOR_UNMEASURED_GB, "no measured peak; unmeasured-phase floor"
    need = max(FLOOR_MIN_GB, FLOOR_MULTIPLE * peak_gb)
    return need, (f"max({FLOOR_MIN_GB}GB, {FLOOR_MULTIPLE} x measured peak "
                  f"{peak_gb:.3f}GB)")


def measured_peak(domain):
    """The committed peak-RSS measurement for a domain, or None if unmeasured."""
    path = OUT / f"mem_{domain}.json"
    if not path.exists():
        return None
    marks = json.loads(path.read_text()).get("peak_rss_gb", {})
    return max(marks.values()) if marks else None


def check_floor(phase, peak_gb=None, log="memory_floor.log"):
    """Refuse to start a phase below the available-memory floor (HANDOFF trap)."""
    avail = mem_available_gb()
    need, why = required_floor(peak_gb)
    headroom = avail - (peak_gb or 0.0)
    ok = avail >= need and headroom >= FLOOR_HEADROOM_GB
    OUT.mkdir(exist_ok=True)
    with open(OUT / log, "a") as fh:
        fh.write(f"{phase}\tMemAvailable={avail:.2f}GB\trequired={need:.2f}GB "
                 f"({why})\theadroom_after={headroom:.2f}GB\t"
                 f"{'START' if ok else 'REFUSED'}\n")
    if not ok:
        raise SystemExit(
            f"refusing {phase}: MemAvailable {avail:.2f} GB, required "
            f"{need:.2f} GB ({why}), headroom after {headroom:.2f} GB "
            f"(need {FLOOR_HEADROOM_GB} GB)")
    return avail


def peak_rss_gb():
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)
