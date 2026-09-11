"""Reconstruct provenance for corpora built before A21, with git as the evidence.

A20's rule is that back-filling a provenance stamp makes the stamp worthless: a
stamp asserts "this is what produced me", and writing it afterwards asserts only
"this is what I now believe produced me".

So this does **not** write a `provenance` block. It writes
`provenance_reconstructed`, which is a different claim and is labelled as one:
*the producing files were unchanged between the commit that recorded the corpus
and now, therefore the current digests also describe the corpus.* That is
checkable — `git log` is the evidence — and the verifier treats a reconstructed
block as weaker than a stamped one rather than equivalent.

    python reconstruct_provenance.py bool rel
"""
from __future__ import annotations

import subprocess
import sys

import kit

PRODUCERS = ("edits.py", "domains.py", "run_domain.py")


def last_commit(path):
    r = subprocess.run(["git", "log", "-1", "--format=%H %ci", "--", path],
                       cwd=kit.HERE, capture_output=True, text=True, timeout=30)
    return r.stdout.strip() or None


def corpus_commit(name):
    r = subprocess.run(["git", "log", "-1", "--format=%H %ci", "--",
                        f"out/{name}.json"], cwd=kit.HERE,
                       capture_output=True, text=True, timeout=30)
    return r.stdout.strip() or None


def diff_summary(name):
    """What changed in the producing files between the corpus commit and HEAD.

    A reconstruction is only as good as this: if a producing file changed in a
    way that touches a decision, the current digests do not describe the corpus
    and the reconstruction must say so rather than paper over it.
    """
    at = (corpus_commit(name) or "").split(" ")[0]
    if not at:
        return None
    out = {}
    for f in PRODUCERS:
        r = subprocess.run(["git", "diff", "--numstat", at, "HEAD", "--", f],
                           cwd=kit.HERE, capture_output=True, text=True, timeout=30)
        line = r.stdout.strip()
        if not line:
            out[f] = "unchanged since the corpus was recorded"
            continue
        added, removed, _ = line.split(None, 2)
        body = subprocess.run(["git", "diff", at, "HEAD", "--", f],
                              cwd=kit.HERE, capture_output=True, text=True, timeout=30)
        out[f] = {"lines_added": int(added), "lines_removed": int(removed),
                  "REVIEW_REQUIRED": "a changed producer means the current "
                                     "digests may not describe this corpus",
                  "diff": body.stdout[:4000]}
    return out


def main():
    for dom in sys.argv[1:] or ["bool", "rel"]:
        name = f"cases_{dom}"
        c = kit.load(name)
        if c.get("provenance"):
            print(f"{dom}: already stamped at build time; nothing to reconstruct")
            continue
        rec = {
            "kind": "reconstructed, not stamped at build time",
            "why": "the corpus predates A21; a stamp written now would assert "
                   "only what is currently believed, so the producing files' "
                   "git history is given as the evidence instead",
            "producing_files": {f: last_commit(f) for f in PRODUCERS},
            "corpus_recorded_at": corpus_commit(name),
            "diff_since_corpus": diff_summary(name),
            "digests_now": kit.provenance(
                base_digest=(c.get("base") or {}).get("digest"),
                split_digest=c.get("final_digest")),
        }
        c["provenance_reconstructed"] = rec
        kit.dump(name, c)
        print(f"{dom}: reconstructed")
        for f, cm in rec["producing_files"].items():
            print(f"    {f:16s} last changed {cm}")
        print(f"    corpus recorded  {rec['corpus_recorded_at']}")


if __name__ == "__main__":
    main()
