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
FORMAT = "schema-induction/provenance-1"


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


def git_head():
    try:
        out = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=30)
        head = out.stdout.strip() or None
        dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain"],
                               capture_output=True, text=True, timeout=30).stdout.strip()
        return {"head": head, "dirty": bool(dirty)}
    except Exception as exc:                                   # noqa: BLE001
        return {"head": None, "dirty": None, "error": repr(exc)}


def source_digests(paths):
    return {str(pathlib.Path(p).relative_to(ROOT)): digest_file(p) for p in paths}


def make(kind, inputs, parameters, sources):
    """The provenance block to embed in an artifact.

    `inputs`     -- {name: digest} of every file or split the job read.
    `parameters` -- the declared constants a reader must be able to check the
                    artifact against (pool contents, pool sizes, gaps, seeds).
    `sources`    -- the code files whose behaviour produced the artifact.
    """
    return {"format": FORMAT, "kind": kind,
            "inputs": dict(inputs),
            "parameters": parameters,
            "parameters_digest": digest_json(parameters),
            "sources": source_digests(sources),
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
                   "git", "written_unix")


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
    return sorted(p["inputs"]) + sorted(p["sources"])


def require(artifact, kind=None, inputs=None, parameters=None, sources=None,
            allow_dirty=True):
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
        if got != digest_file(path):
            raise StampError(f"source {rel!r} changed since the artifact was written; "
                             f"re-run the job rather than trusting the stamp")
        checked.append(f"sources.{rel}")
    if not allow_dirty and (p.get("git") or {}).get("dirty"):
        raise StampError("artifact was produced from a dirty tree")
    if not checked:
        raise StampError("require() was called with nothing to compare: an "
                         "assertion that checks nothing must fail, not pass")
    return checked
