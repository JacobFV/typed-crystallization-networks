"""Provenance stamps: every derived artifact records the identity of its inputs.

The incident this exists to prevent (scaffold-induction, 2026-09-10): a
validation JSON describing a superseded corpus reappeared in `out/` *after* the
directory was cleaned, because the superseded job was still running and wrote
afterwards. Cleaning a directory cannot remove artifacts that have not been
produced yet. The verifier accepted it, because its checks asserted "a
validation ran and reported zero mismatches" and never "it validated *this*
corpus".

The rule, and both of its corollaries, are enforced here mechanically:

* every artifact this track writes carries a `provenance` block naming the
  files, pools, splits and configuration it was derived from, each by digest;
* `require(artifact, expected)` fails when the block **disagrees** and equally
  when it is **absent** -- so an artifact predating this rule cannot silently
  count as current;
* there is no function that adds a stamp to an existing artifact. A stamp is
  obtained by re-running the job that produces it. A hand-added stamp asserts
  exactly what it cannot check.

Digests are SHA-256, compared by token equality on the full hex string. No
substring matching: that has produced three separate false-pass bugs in this
project.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import subprocess
import time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FORMAT = "schema-induction/provenance-2"

#: The owner's permanent provenance standard (2026-09-11): every evidence
#: artifact carries the base scaffold/program digest, the split/corpus digest,
#: the mutation-grammar version digest, the producing commit, and the
#: pre-registration/amendment revision. The first two are per-artifact and live
#: in `inputs`; the last three are recorded on every artifact by `make`.
STANDARD = ("base_digest", "splits_digest", "grammar_digest", "commit", "revision")

#: The pre-registration revision this code implements. Bumped by hand when
#: PREREGISTRATION.md changes in a way that changes what an artifact means; the
#: digest beside it is what actually detects drift.
REVISION = "r3-2026-09-11-clean-producer-and-structural-c3"


class StampError(RuntimeError):
    """An artifact's provenance is absent, malformed, or disagrees with disk."""


def digest_bytes(data):
    return hashlib.sha256(data).hexdigest()


def digest_file(path):
    path = pathlib.Path(path)
    if not path.exists():
        raise StampError(f"cannot fingerprint a missing input: {path}")
    return digest_bytes(path.read_bytes())


def digest_json(obj):
    """A digest of a Python object, stable under dict ordering."""
    return digest_bytes(json.dumps(obj, sort_keys=True, default=str).encode())


def digest_episodes(episodes):
    """A split's fingerprint, computed without using it for any decision.

    The fields are the ones a decision could depend on: the actor's
    observations and the probes that score them. Deliberately not the whole
    record, so a change to an unused bookkeeping field does not invalidate a
    corpus -- and deliberately including `owner` and `target`, so a change to
    what counts as a hit does.
    """
    h = hashlib.sha256()
    for ep in episodes:
        for key in ("gap", "seed", "index", "split", "relation", "colour",
                    "text_length", "text_bytes", "pixels", "owner", "target",
                    "anchor", "width", "height"):
            h.update(repr(ep.get(key)).encode())
    return h.hexdigest()


def _git(*args):
    return subprocess.run(["git", "-C", str(ROOT), *args],
                          capture_output=True, text=True, timeout=60)


#: The run's own output surface. Regenerating a committed artifact necessarily
#: modifies or deletes it, so counting `out/` would make a clean re-run
#: impossible -- the first attempt at the fail-closed gate failed for exactly
#: that reason, on deletions this track's own re-run had just made.
OUTPUT_SURFACE = "research/schema-induction/out/"


def git_head():
    """The producing commit, and whether it is an exact snapshot of the SOURCES.

    What the gate guarantees, stated precisely: **every file that produced this
    artifact is committed and unmodified.** Three parts:

    * `dirty` counts tracked modifications **outside the run's own output
      surface**. Outputs are what a run produces; they cannot make their own
      producer dirty. Modifications inside `out/` are still recorded, in
      `dirty_outputs`, so nothing is hidden -- they are disclosed, not counted.
    * untracked files do not count either, for the same reason.
    * the hole both of those open -- an untracked or modified *source*, which is
      in no commit -- is closed by `source_digests` recording `tracked` per file
      and `present` refusing any artifact with an untracked source, plus
      `require`'s per-source digest comparison.

    So the producing commit pins a real snapshot of the code and the
    pre-registration, which is what "not an exact snapshot" was about.
    """
    try:
        head = _git("rev-parse", "HEAD").stdout.strip() or None
        rows = _git("status", "--porcelain", "--untracked-files=no").stdout.splitlines()
        outputs = [r for r in rows if r[3:].startswith(OUTPUT_SURFACE)]
        sources = [r for r in rows if not r[3:].startswith(OUTPUT_SURFACE)]
        untracked = _git("ls-files", "--others", "--exclude-standard").stdout.split()
        return {"head": head, "dirty": bool(sources),
                "dirty_tracked_paths": sources,
                "dirty_outputs": outputs,
                "output_surface": OUTPUT_SURFACE,
                "untracked_count": len(untracked),
                "dirty_means": "tracked modifications to SOURCES (everything outside "
                               "the run's own out/ directory). Output churn is "
                               "disclosed in `dirty_outputs`, not counted; untracked "
                               "SOURCES are refused outright by `present`"}
    except Exception as exc:                                   # noqa: BLE001
        return {"head": None, "dirty": None, "error": repr(exc)}


def source_digests(paths):
    """{relative path: {digest, tracked}} for every source that produced an artifact."""
    out = {}
    for p in paths:
        rel = str(pathlib.Path(p).relative_to(ROOT))
        tracked = _git("ls-files", "--error-unmatch", rel).returncode == 0
        out[rel] = {"digest": digest_file(p), "tracked": tracked}
    return out


def prereg_digest():
    return digest_file(HERE / "PREREGISTRATION.md")


def grammar_digest():
    """The mutation grammar's version digest, or an explicit absence.

    A phase that runs before any grammar exists records `None` with a reason,
    rather than omitting the field — an omitted field reads as "not recorded"
    and would let a later artifact inherit the silence.
    """
    path = HERE / "grammar.py"
    if not path.exists():
        return {"digest": None, "version": None,
                "absent_because": "no mutation grammar existed when this artifact "
                                  "was produced (phase 0 precedes the grammar)"}
    import importlib
    module = importlib.import_module("grammar")
    return {"digest": digest_file(path),
            "version": getattr(module, "GRAMMAR_VERSION", None),
            "families": list(getattr(module, "FAMILIES", ())),
            "absent_because": None}


def make(kind, inputs, parameters, sources, base=None, splits=None):
    """The provenance block to embed in an artifact.

    `inputs`     -- {name: digest} of every file or split the job read.
    `parameters` -- the declared constants a reader must be able to check the
                    artifact against (pool contents, pool sizes, gaps, seeds).
    `sources`    -- the code files whose behaviour produced the artifact.
    `base`       -- the base scaffold/program this artifact is about, as a
                    digestible object (its pools). Required by the owner's
                    standard; pass the schema the job started from.
    `splits`     -- {split name: digest} of every corpus split involved,
                    including ones deliberately NOT read (the blind final
                    split's digest belongs here, recorded but unopened).
    """
    return {"format": FORMAT, "kind": kind,
            "inputs": dict(inputs),
            "parameters": parameters,
            "parameters_digest": digest_json(parameters),
            "sources": source_digests(sources),
            "base_digest": digest_json(base) if base is not None else None,
            "base": base,
            "splits_digest": digest_json(splits) if splits is not None else None,
            "splits": splits,
            "grammar_digest": grammar_digest(),
            "revision": REVISION,
            "prereg_digest": prereg_digest(),
            "commit": git_head(),
            "git": git_head(),
            "written_unix": time.time(),
            "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def write(path, payload, provenance):
    """Write an artifact with its stamp. The only supported way to produce one."""
    if "provenance" in payload:
        raise StampError("refusing to overwrite an existing provenance block")
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload, provenance=provenance), indent=1))
    return path


REQUIRED_FIELDS = ("kind", "inputs", "parameters", "parameters_digest", "sources",
                   "git", "written_unix") + STANDARD


def present(artifact):
    """The block exists and is well formed — a real check, not a no-op.

    This is the weakest admissible assertion, used where the caller only knows
    that an artifact *should* be stamped (a registry sweep) and not what it
    should be stamped with. It still fails on an absent block, an unknown
    format, a missing field, and an empty `inputs` or `sources` — an artifact
    derived from nothing, or by nothing, cannot prove it describes the object
    on disk. Where the caller knows the expected inputs, use `require`.
    """
    if not isinstance(artifact, dict):
        raise StampError("artifact is not a JSON object")
    p = artifact.get("provenance")
    if p is None:
        raise StampError("artifact carries no provenance block: it cannot prove "
                         "it describes the object currently on disk, so it is "
                         "refused rather than assumed current")
    if p.get("format") != FORMAT:
        raise StampError(f"unknown provenance format {p.get('format')!r}")
    missing = [f for f in REQUIRED_FIELDS if f not in p]
    if missing:
        raise StampError(f"provenance block is missing {missing}")
    if not p["inputs"]:
        raise StampError("provenance names no inputs: an artifact derived from "
                         "nothing cannot be checked against anything")
    if not p["sources"]:
        raise StampError("provenance names no sources")
    if p["parameters_digest"] != digest_json(p["parameters"]):
        raise StampError("parameters_digest does not match the recorded parameters; "
                         "the block was edited after it was written")
    # The owner's permanent standard: each of these is recorded, and an
    # explicit absence (with a reason) is acceptable where an omission is not.
    if not isinstance(p["grammar_digest"], dict) or \
            set(p["grammar_digest"]) < {"digest", "absent_because"}:
        raise StampError("grammar_digest must record the mutation grammar's version "
                         "digest, or an explicit absence with a reason")
    if p["grammar_digest"]["digest"] is None and not p["grammar_digest"]["absent_because"]:
        raise StampError("grammar_digest is absent with no reason given")
    if not isinstance(p["revision"], str) or not p["revision"]:
        raise StampError("no pre-registration revision recorded")
    if not (p["commit"] or {}).get("head"):
        raise StampError("no producing commit recorded")
    # FAIL CLOSED on a dirty producer (owner, 2026-09-11). A dirty tree is not
    # an exact snapshot, and source hashes alone only mean the evidence is not
    # lost -- they do not make the commit field mean what it says. The required
    # path is: commit the code, re-run from the clean commit, commit outputs.
    if (p["commit"] or {}).get("dirty") is not False:
        raise StampError(
            "produced from a dirty tree: an evidence artifact must come from a "
            "clean source state. Commit the code and pre-registration, re-run, "
            "then commit the outputs")
    untracked = sorted(k for k, v in p["sources"].items()
                       if isinstance(v, dict) and not v.get("tracked"))
    if untracked:
        raise StampError(f"produced by untracked source(s) {untracked}: they are in "
                         f"no commit, so the producing commit pins nothing")
    for field in ("base_digest", "splits_digest"):
        if field not in p:
            raise StampError(f"provenance omits {field}")
        if p[field] is not None and p[field] != digest_json(p[field.split("_")[0]]):
            raise StampError(f"{field} does not match the recorded {field.split('_')[0]}; "
                             f"the block was edited after it was written")
    return sorted(p["inputs"]) + sorted(p["sources"])


def require(artifact, kind=None, inputs=None, parameters=None, sources=None,
            allow_dirty=False):
    """Fail when the stamp is absent, malformed, or disagrees with disk.

    Returns the list of agreed field names, so a verifier can assert that the
    comparison it believes it made actually happened.
    """
    if not isinstance(artifact, dict):
        raise StampError("artifact is not a JSON object")
    p = artifact.get("provenance")
    if p is None:
        raise StampError("artifact carries no provenance block: it cannot prove "
                         "it describes the object currently on disk, so it is "
                         "refused rather than assumed current")
    if p.get("format") != FORMAT:
        raise StampError(f"unknown provenance format {p.get('format')!r}")
    checked = []
    if kind is not None:
        if p.get("kind") != kind:
            raise StampError(f"provenance kind {p.get('kind')!r} != {kind!r}")
        checked.append("kind")
    for name, want in (inputs or {}).items():
        got = (p.get("inputs") or {}).get(name)
        if got is None:
            raise StampError(f"provenance names no input {name!r}")
        if got != want:
            raise StampError(f"input {name!r} digest {got} != {want} on disk now")
        checked.append(f"inputs.{name}")
    if parameters is not None:
        want = digest_json(parameters)
        if p.get("parameters_digest") != want:
            raise StampError(f"parameters digest {p.get('parameters_digest')} != {want}; "
                             f"the artifact describes a different declared object")
        checked.append("parameters_digest")
    for path in (sources or []):
        rel = str(pathlib.Path(path).relative_to(ROOT))
        got = (p.get("sources") or {}).get(rel)
        if got is None:
            raise StampError(f"provenance names no source {rel!r}")
        if got.get("digest") != digest_file(path):
            raise StampError(f"source {rel!r} changed since the artifact was written; "
                             f"re-run the job rather than trusting the stamp")
        if not got.get("tracked"):
            raise StampError(f"source {rel!r} was untracked when the artifact was written")
        checked.append(f"sources.{rel}")
    if not allow_dirty and (p.get("commit") or {}).get("dirty") is not False:
        raise StampError("artifact was produced from a dirty tree")
    if not checked:
        raise StampError("require() was called with nothing to compare: an "
                         "assertion that checks nothing must fail, not pass")
    return checked
