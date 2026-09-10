#!/usr/bin/env python3
"""The record's evidence gate: every claim it can check, checked against committed artifacts.

This began as the one-off record audit (`research/record-audit/RESULTS.md`, FINDINGS
§62) and is now a routine gate. It reads the *documents* — `research/FINDINGS.md`,
`STATUS.md`, `README.md`, `HANDOFF.md`, `docs/CORRECTIONS.md`, `docs/VALIDATION.md` —
extracts the figure each claim states, and compares it with the value in the committed
raw artifact under `research/<track>/`. Nothing is compared against a summary.

Rules the checks follow
-----------------------
* **The claim is read from the document, not hard-coded.** A check names a regex that
  captures the figure as the document states it and a function that reads the artifact.
  If the document no longer contains the claim, the check FAILS ("claim text not
  found") rather than silently passing: an edit that removes or rewords a checked claim
  must update its check in the same commit.
* **Struck-through text is the retained original** (the house rule keeps what was
  believed and when). It is removed before matching, so `~~old~~ new` checks `new`.
* **A figure matches at the precision the document quotes** (0.9986 matches
  0.99861878…; 3.9 does not match 3.833), unless a check states a tolerance.
* Statuses: **PASS**; **FAIL** (the document and the artifact disagree, or a claim
  vanished); **UNCHECKABLE** (no committed artifact can decide it — the reason is
  printed, and it is never counted as verified); **OPEN** (a known discrepancy that the
  committed artifacts cannot resolve, recorded in RESULTS.md with its audit id; it does
  not fail the gate, and if it ever starts to pass the gate FAILS so the marker is
  retired); **WARN** (informational).
* **Every FINDINGS section must be covered**: at least one PASSing artifact check, or an
  explicit entry in `SECTION_UNCHECKABLE` with the reason. A new section with neither
  FAILS the gate.

Usage
-----
    .venv/bin/python research/record-audit/verify.py --gate        # fast gate: run by tests/test_record_gate.py
    .venv/bin/python research/record-audit/verify.py               # gate + test-count + demo summary if present
    .venv/bin/python research/record-audit/verify.py --demo        # also run the ten demonstrations first (slow)
    .venv/bin/python research/record-audit/verify.py --tests       # also run pytest (slow)
    .venv/bin/python research/record-audit/verify.py --coverage    # per-section coverage table
    .venv/bin/python research/record-audit/verify.py --emit-headlines   # rewrite HEADLINES.md from artifacts

`--gate` never runs pytest, never runs a demo, never trains or enumerates, and never
writes an artifact. Exit status is non-zero iff any check FAILs.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FINDINGS = ROOT / "research" / "FINDINGS.md"
STATUS = ROOT / "STATUS.md"
README = ROOT / "README.md"
CORRECTIONS = ROOT / "docs" / "CORRECTIONS.md"
VALIDATION = ROOT / "docs" / "VALIDATION.md"
HANDOFF = ROOT / "HANDOFF.md"
HEADLINES = ROOT / "research" / "record-audit" / "HEADLINES.md"

PROSE_FILES = [FINDINGS, STATUS, README, CORRECTIONS, VALIDATION, HANDOFF]

PASS, FAIL, WARN, UNCHECKABLE, OPEN = "PASS", "FAIL", "WARN", "UNCHECKABLE", "OPEN"


@dataclass
class Result:
    status: str
    ident: str
    detail: str
    sections: tuple
    artifact: bool


results: list[Result] = []
headline_rows: list[tuple] = []


def record(status: str, ident: str, detail: str, sections=(), artifact: bool = False) -> None:
    results.append(Result(status, ident, detail, tuple(sections), artifact))


class Unavailable(Exception):
    """The artifact exists only somewhere this checkout cannot read (another branch)."""


# --------------------------------------------------------------------------
# documents
# --------------------------------------------------------------------------

@lru_cache(maxsize=None)
def raw(path: Path) -> str:
    return path.read_text()


def live(text: str) -> str:
    """Remove struck-through spans (the retained originals)."""
    return re.sub(r"~~.*?~~", " ", text, flags=re.S)


def norm(text: str) -> str:
    text = text.replace("**", "")
    return re.sub(r"\s+", " ", text)


@lru_cache(maxsize=None)
def T(path: Path) -> str:
    """A document's live text, normalised for matching."""
    return norm(live(raw(path)))


HEAD = re.compile(r"^## (\d+)([a-z]?)\. (.*)$", re.M)


@lru_cache(maxsize=None)
def findings_sections() -> tuple:
    """(label, title, body) for every FINDINGS heading, in file order."""
    text = raw(FINDINGS)
    heads = list(HEAD.finditer(text))
    out = []
    for i, m in enumerate(heads):
        end = heads[i + 1].start() if i + 1 < len(heads) else len(text)
        out.append((m.group(1) + m.group(2), m.group(3), text[m.end():end]))
    return tuple(out)


def is_tombstone(title: str) -> bool:
    return "never assigned" in title.lower()


@lru_cache(maxsize=None)
def S(label: str) -> str:
    """Live, normalised text of one FINDINGS section ('19', '14a', ...)."""
    for lab, title, body in findings_sections():
        if lab == label:
            return norm(live(title + " " + body))
    raise KeyError(f"no FINDINGS section {label}")


def sec_ids(*labels) -> tuple:
    return tuple("§" + str(x) for x in labels)


# --------------------------------------------------------------------------
# artifacts
# --------------------------------------------------------------------------

def _parse(text: str):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    out, dec, idx = [], json.JSONDecoder(), 0
    while idx < len(text):
        while idx < len(text) and text[idx] in " \t\r\n":
            idx += 1
        if idx >= len(text):
            break
        obj, idx = dec.raw_decode(text, idx)
        out.append(obj)
    return out


@lru_cache(maxsize=None)
def J(rel: str):
    """Load a committed JSON/JSONL artifact (concatenated objects come back as a list)."""
    return _parse((ROOT / rel).read_text())


@lru_cache(maxsize=None)
def git_refs() -> tuple:
    try:
        out = subprocess.run(["git", "for-each-ref", "--format=%(refname:short)", "refs/heads", "refs/remotes"],
                             cwd=ROOT, capture_output=True, text=True, timeout=30).stdout
    except Exception:
        return ()
    return tuple(out.split())


def on_ref(refs, rel: str):
    """Return the first ref among `refs` that has `rel` committed, or None."""
    have = set(git_refs())
    for ref in refs:
        if ref not in have:
            continue
        ok = subprocess.run(["git", "cat-file", "-e", f"{ref}:{rel}"], cwd=ROOT,
                            capture_output=True, timeout=30).returncode == 0
        if ok:
            return ref
    return None


@lru_cache(maxsize=None)
def JB(refs: tuple, rel: str):
    """Load an artifact committed on another branch. Raises Unavailable when no listed
    ref is present in this clone (an external clone of main alone, for instance)."""
    ref = on_ref(refs, rel)
    if ref is None:
        raise Unavailable(f"{rel} is committed only on {' / '.join(refs)}, which this clone does not have")
    out = subprocess.run(["git", "show", f"{ref}:{rel}"], cwd=ROOT, capture_output=True, text=True, timeout=60)
    return _parse(out.stdout)


def dig(obj, dotted: str):
    cur = obj
    for part in dotted.split("."):
        if part == "":
            continue
        cur = cur[int(part)] if isinstance(cur, list) else cur[part]
    return cur


# --------------------------------------------------------------------------
# claim primitives
# --------------------------------------------------------------------------

def num(s: str) -> float:
    s = s.replace(",", "").replace("−", "-").strip()
    s = re.sub(r"[x×%]+$", "", s)
    return float(s)


def at_precision(claimed: str, actual: float) -> bool:
    """True if `actual`, rounded half-up to the decimals `claimed` shows, equals it."""
    c = claimed.replace(",", "").replace("−", "-").strip()
    c = re.sub(r"[x×%]+$", "", c)
    if "e" in c.lower():
        mant, exp = c.lower().split("e")
        dec = len(mant.split(".")[1]) if "." in mant else 0
        a = float(actual)
        if a == 0:
            return float(c) == 0
        e = int(exp)
        return at_precision(mant, a / (10 ** e))
    dec = len(c.split(".")[1]) if "." in c else 0
    q = Decimal(1).scaleb(-dec)
    return Decimal(repr(float(actual))).quantize(q, rounding=ROUND_HALF_UP) == Decimal(c).quantize(q)


def fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)


def _run(fn):
    try:
        return True, fn()
    except Unavailable as exc:
        return None, str(exc)
    except Exception as exc:  # noqa: BLE001 — every failure is reported, none swallowed
        return False, f"{type(exc).__name__}: {exc}"


def claim(ident, secs, text, pattern, actual, src, tol=None, scale=1.0, headline=True):
    """A numeric claim: `pattern`'s first group, read from `text`, must equal `actual()`."""
    m = re.search(pattern, text)
    if not m:
        record(FAIL, ident, f"claim text not found: /{pattern}/ — the document changed; update this check "
                            "or restore the claim", secs, True)
        return
    ok, val = _run(actual if callable(actual) else (lambda: actual))
    if ok is None:
        record(UNCHECKABLE, ident, val, secs, True)
        return
    if not ok:
        record(FAIL, ident, f"artifact read failed ({src}): {val}", secs, True)
        return
    got = val * scale
    good = (abs(got - num(m.group(1))) <= tol) if tol is not None else at_precision(m.group(1), got)
    record(PASS if good else FAIL, ident,
           f"document: '{m.group(0)[:90]}' | {src} = {fmt(got)}", secs, True)
    if headline:
        headline_rows.append((secs, ident, fmt(got), src))


def fact(ident, secs, text, pattern, test, src):
    """A non-numeric claim. `pattern` must be present in `text` (it anchors the check to
    the sentence making the claim); `test()` returns (ok, detail) from the artifact."""
    if pattern is not None and not re.search(pattern, text):
        record(FAIL, ident, f"claim text not found: /{pattern}/ — update this check or restore the claim", secs, True)
        return
    ok, val = _run(test)
    if ok is None:
        record(UNCHECKABLE, ident, val, secs, True)
        return
    if not ok:
        record(FAIL, ident, f"artifact read failed ({src}): {val}", secs, True)
        return
    good, detail = val
    record(PASS if good else FAIL, ident, f"{detail} ({src})", secs, True)


def absent(ident, secs, text, pattern, why, history=None):
    """Wording the record has retracted must not reappear un-struck. A match whose
    preceding 90 characters match `history` is a sentence *about* the retracted
    figure ("this row first said 9/12"), not an assertion of it, and is skipped."""
    m = None
    for cand in re.finditer(pattern, text):
        if history and re.search(history, text[max(0, cand.start() - 90):cand.start()]):
            continue
        m = cand
        break
    record(FAIL if m else PASS, ident, (f"found '{m.group(0)[:90]}' — {why}" if m else "absent (retracted wording is struck or gone)"),
           secs, False)


def present(ident, secs, text, pattern, why):
    m = re.search(pattern, text)
    record(PASS if m else FAIL, ident, ("present" if m else f"missing: /{pattern}/ — {why}"), secs, False)


def open_item(ident, secs, audit_id, detail, still_open: bool = True):
    """A known discrepancy the committed artifacts cannot resolve. If `still_open` turns
    False the discrepancy has been resolved and the OPEN marker must be retired."""
    if still_open:
        record(OPEN, ident, f"[{audit_id}] {detail}", secs, False)
    else:
        record(FAIL, ident, f"[{audit_id}] no longer open — resolve it in RESULTS.md and retire this marker", secs, False)


def uncheckable(ident, secs, reason):
    record(UNCHECKABLE, ident, reason, secs, False)


# --------------------------------------------------------------------------
# 1. structural integrity of the record
# --------------------------------------------------------------------------

# A number used twice is only tolerated when the two headings carry distinct letter
# suffixes and the ambiguity is documented; a gap only when a tombstone heading marks it.
ALLOWED_SPLITS = {"14": ("14a", "14b")}


def check_section_numbering() -> None:
    secs = findings_sections()
    labels = [lab for lab, _, _ in secs]
    dupes = sorted({lab for lab in labels if labels.count(lab) > 1})
    record(FAIL if dupes else PASS, "record/no-duplicate-sections",
           f"duplicate FINDINGS.md section labels: {dupes}" if dupes else f"{len(labels)} headings, labels unique")

    by_num: dict[str, list[str]] = {}
    for lab in labels:
        by_num.setdefault(re.match(r"\d+", lab).group(0), []).append(lab)
    bad_splits = [f"{n}: {v}" for n, v in by_num.items()
                  if len(v) > 1 and tuple(sorted(v)) != ALLOWED_SPLITS.get(n)]
    record(FAIL if bad_splits else PASS, "record/split-sections-documented",
           f"a number is shared by undocumented headings: {bad_splits}" if bad_splits
           else "the only shared number is §14, headed §14a and §14b")

    nums = sorted(int(n) for n in by_num)
    gaps = [n for n in range(1, nums[-1] + 1) if n not in nums]
    record(FAIL if gaps else PASS, "record/no-missing-sections",
           f"section numbers with no heading (not even a tombstone): {gaps}" if gaps
           else f"sections 1..{nums[-1]} all have a heading (tombstones: "
                f"{[lab for lab, t, _ in secs if is_tombstone(t)]})")


def check_cross_references() -> None:
    """Every §N / 'section N' reference must name a section that exists, and a bare §14
    is ambiguous now that §14a and §14b exist."""
    labels = {lab for lab, _, _ in findings_sections()}
    numbers = {re.match(r"\d+", lab).group(0) for lab in labels}
    top = max(int(n) for n in numbers)
    dangling, ambiguous = [], []
    for path in PROSE_FILES:
        if not path.exists():
            continue
        text = live(raw(path))
        for m in re.finditer(r"(?:§|[Ss]ection )(\d{1,2})([a-z]?)\b", text):
            n, suf = m.group(1), m.group(2)
            before = text[max(0, m.start() - 90):m.start()]
            if re.search(r"(RESULTS\.md|DESIGN|VALIDATION\.md|ARCHITECTURE|BLUEPRINT|IMPLEMENTATION|PREREGISTRATION)[^.]{0,20}$",
                         before):
                continue
            where = f"{path.relative_to(ROOT)}:{text[:m.start()].count(chr(10)) + 1}"
            if int(n) <= top and n not in numbers:
                dangling.append(f"{where} -> §{n}")
            elif n in ALLOWED_SPLITS and not suf:
                after = text[m.end():m.end() + 20]
                if not re.match(r"\s*(appears twice|is used twice|was used twice|\(now)|\"|;", after):
                    ambiguous.append(where)
    record(FAIL if dangling else PASS, "record/no-dangling-section-refs",
           "; ".join(sorted(set(dangling))) if dangling else "no dangling section references")
    record(FAIL if ambiguous else PASS, "record/no-ambiguous-14-refs",
           "bare §14 / section 14 (say §14a or §14b): " + "; ".join(sorted(set(ambiguous))) if ambiguous
           else "every reference to §14 names §14a or §14b")


# Paths the record cites that live only on an unmerged branch. They are not failures,
# but nothing on this branch can check them.
OFF_BRANCH = {
    "research/seasons": ("seasons", "origin/seasons"),
    "research/emitter-guards": ("research/emitter-guards", "origin/emitter-guards"),
    "research/refinement-bounds": ("worktree-agent-a0350f1fdeac805b2", "origin/refinement-bounds"),
}


def check_cited_paths() -> None:
    missing: dict[str, set] = {}
    pat = re.compile(r"`(research/[A-Za-z0-9_\-./]+)`")
    for path in PROSE_FILES:
        if not path.exists():
            continue
        for m in pat.finditer(live(raw(path))):
            target = m.group(1).rstrip("/.,;:")
            if not (ROOT / target).exists():
                missing.setdefault(target, set()).add(path.name)
    hard, offb = [], []
    for target, where in sorted(missing.items()):
        track = "/".join(target.split("/")[:2])
        refs = OFF_BRANCH.get(track)
        if refs is None:
            hard.append(f"{target} (cited in {','.join(sorted(where))})")
            continue
        sub = target[len(track) + 1:] if target != track else ""
        ref = on_ref(refs, target if sub else target + "/RESULTS.md")
        offb.append(f"{target} -> {ref or 'no listed branch in this clone: ' + '/'.join(refs)}")
    record(FAIL if hard else PASS, "record/cited-paths-exist",
           ("paths cited but committed nowhere known: " + "; ".join(hard)) if hard
           else "every cited research/ path resolves here or on its recorded branch")
    if offb:
        uncheckable("record/cited-paths-off-branch", sec_ids(37, 59, 61),
                    "cited paths committed only on unmerged branches, so a clone of main cannot "
                    "check them: " + "; ".join(offb))


# --------------------------------------------------------------------------
# 2. retracted wording must not reappear un-struck
# --------------------------------------------------------------------------

def check_retracted_claims() -> None:
    F, ST, RM, CO, VA = T(FINDINGS), T(STATUS), T(README), T(CORRECTIONS), T(VALIDATION)
    absent("retracted/language-1.000-in-VALIDATION", sec_ids(43), VA,
           r"grammaticality rule 1\.000|reaches 1\.000 on 724",
           "CORRECTIONS row 9: the language capability is 0.9986 on 724")
    absent("retracted/language-stream-unqualified-in-README", sec_ids(45), RM,
           r"language 0\.9986(?! on the pre-audit stream)",
           "§45: every quotation of §19's number must name the stream")
    absent("retracted/3069-as-a-type-bound", sec_ids(59, 61), F,
           r"missing refinement bound[^.]*\(0, 3069\)(?![^\]]*CORRECTED)",
           "CORRECTIONS row 20: (0, 3069) is unsound as a type bound")
    absent("retracted/§19-exact-8-through-16", sec_ids(19), S("19"), r"exact at lengths 8 through 16",
           "final_eval.json per_length[16] = 0.5")
    absent("retracted/§14a-validation-filter-fixed-it", sec_ids("14a"), F + ST,
           r"Requiring exactness at every position of a validation split is what fixed it|"
           r"requiring exactness on a validation split fixed both",
           "tiebreak.json: validation_filtered returns the lexicographic program")
    absent("retracted/§21-every-context", sec_ids(21), F,
           r"advantage exactly 0\.0000 at every context for",
           "bounds.json: zero only at raw-byte contexts")
    absent("retracted/§43-20-of-20-every-screen", sec_ids(43), F + RM,
           r"20/20 rectangles, 20/20 parent links|against 20/20 rectangles",
           "tcn_visual.json: screens carry 15-20 widgets; totals 227/227")
    absent("retracted/footprint-across-the-board", sec_ids(36, 43), F + RM,
           r"wins deployment footprint across the board|typed side wins across the board is deployment",
           "pyz_visual.json: 377.5 MB RSS, above the 270.8 MB torch baseline")
    absent("retracted/9-of-12", sec_ids(39), S("39") + CO + norm(live(raw(ROOT / "research/inference-cost/RESULTS.md"))),
           r"\b9/12\b", "inproc.json: 7/12",
           history=r"(first said|originally recorded|not the|said the|Row 10 said|was wrong)")
    absent("retracted/2.00-best-constant", sec_ids(20), S("20") + CO, r"best constant is 2\.00/4",
           "the recorded-protocol best constant is 3.00/4")
    absent("retracted/§41-spans-4-through-29", sec_ids(41), S("41"), r"none exists` at spans 4 through 29|none exists at spans 4 through 29",
           "only 8 spans in that range were enumerated")


def check_status_freshness(slow: bool) -> None:
    text = T(STATUS)
    checks = [
        ("status/policy-learning-has-results",
         "no `RESULTS.md`" in text and (ROOT / "research/policy-learning/RESULTS.md").exists(),
         "STATUS.md says research/policy-learning has no RESULTS.md; it does (FINDINGS §22)"),
        ("status/object-identity-placeholders",
         "unrendered" in text and "{{" not in raw(ROOT / "research/object-identity/RESULTS.md"),
         "STATUS.md reports unrendered {{ }} placeholders in object-identity/RESULTS.md; there are none"),
        ("status/B7-entry-point", "The documented entry point does not run" in text,
         "STATUS.md B7 says .venv/bin/tcn does not run; its own header says B7 is closed"),
    ]
    for ident, stale, why in checks:
        record(FAIL if stale else PASS, ident, why if stale else "fresh")
    if not slow:
        return
    m = re.search(r"(\d+) Python tests collected", text)
    if not m:
        record(FAIL, "status/test-count", "STATUS.md no longer states a Python test count in the checked form")
        return
    claimed, collected = int(m.group(1)), count_tests()
    if collected is None:
        uncheckable("status/test-count", (), "pytest --collect-only did not report a count")
    else:
        record(PASS if claimed == collected else FAIL, "status/test-count",
               f"STATUS.md claims {claimed} Python tests collected; pytest collects {collected}")


def count_tests():
    try:
        out = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q"],
                             cwd=ROOT, capture_output=True, text=True, timeout=600).stdout
    except Exception:
        return None
    m = re.search(r"(\d+) tests? collected", out)
    return int(m.group(1)) if m else None


# --------------------------------------------------------------------------
# 3. per-section claims against artifacts
# --------------------------------------------------------------------------

def checks_language() -> None:
    """§19, §39, §43, §45, §63 and the documents that quote the language figure."""
    fe = "research/language-capability/final_eval.json"
    acc = lambda: dig(J(fe), "heldout_unseen_lengths.accuracy")  # noqa: E731
    n = lambda: dig(J(fe), "heldout_unseen_lengths.n")  # noqa: E731
    claim("§19/language-accuracy", sec_ids(19), S("19"), r"reaches (0\.9986) on 724 held-out", acc, fe)
    claim("§19/language-n", sec_ids(19), S("19"), r"reaches 0\.9986 on (724) held-out", n, fe)
    claim("§19/majority", sec_ids(19), S("19"), r"Against a (0\.548) majority constant",
          lambda: dig(J(fe), "heldout_unseen_lengths.majority_constant"), fe)
    fact("§19/per-length", sec_ids(19), S("19"),
         r"exact at lengths 8, 10, 12 and 14 and scores 0\.5 at length 16",
         lambda: (lambda p: ([k for k, v in p.items() if v == 1.0] == ["8", "10", "12", "14"] and p["16"] == 0.5,
                             f"per_length {p}"))(dig(J(fe), "heldout_unseen_lengths.per_length")), fe)
    sb = "research/language-capability/stage_b.json"
    fact("§19/stage-B-not-unique", sec_ids(19), S("19"), r"stage-B search was `unique: false` with 10 conforming",
         lambda: (J(sb)["enumeration"]["unique"] is False and J(sb)["enumeration"]["conforming"] == 10,
                  f"stage_b.json unique={J(sb)['enumeration']['unique']} conforming={J(sb)['enumeration']['conforming']}"),
         sb)
    sa = "research/language-capability/stage_a.json"
    claim("§19/stage-A-space", sec_ids(19), S("19"), r"Enumeration exhausts ([\d,]+) programs in 3\.5 s",
          lambda: _first_key(J(sa), ("space_size",)), sa)

    sc = "research/language-capability/reproduce/stream_check.json"
    rows = lambda: J(sc) if isinstance(J(sc), list) else [J(sc)]  # noqa: E731
    pre = lambda: next(r for r in rows() if "none" in str(r.get("stream", "")))  # noqa: E731
    post = lambda: next(r for r in rows() if "post-audit" in str(r.get("stream", "")))  # noqa: E731
    s39 = S("39")
    claim("§39/preaudit-n", sec_ids(39), s39, r"pre-audit stream \(`hardening='none'`\) \| (\d+) \|", lambda: pre()["n_unseen"], sc)
    claim("§39/preaudit-acc", sec_ids(39), s39, r"pre-audit stream \(`hardening='none'`\) \| 724 \| ([\d.]+) \|",
          lambda: pre()["unseen_accuracy"], sc)
    claim("§39/preaudit-majority", sec_ids(39), s39, r"\| 724 \| 0\.99862 \| ([\d.]+) \|", lambda: pre()["majority_baseline"], sc)
    claim("§39/postaudit-acc", sec_ids(39), s39, r"\| 1500 \| ([\d.]+) \|", lambda: post()["unseen_accuracy"], sc)
    claim("§39/postaudit-majority", sec_ids(39), s39, r"\| 1500 \| 0\.4733 \| ([\d.]+) \|", lambda: post()["majority_baseline"], sc)
    fact("§39/postaudit-empty-train-split", sec_ids(39), s39, r"its training split is empty",
         lambda: (post()["n_train_lengths_246"] == 0, f"train episodes at lengths 2/4/6 = {post()['n_train_lengths_246']}"), sc)

    ip = "research/inference-cost/out/inproc.json"
    agree = lambda: sum(1 for e in dig(J(ip), "language.episodes") if e.get("agree"))  # noqa: E731
    disagree = lambda: [i for i, e in enumerate(dig(J(ip), "language.episodes")) if not e.get("agree")]  # noqa: E731
    claim("§39/inproc-agreement", sec_ids(39), s39, r"so it is (\d+)/12", agree, ip)
    claim("§62/inproc-agreement", sec_ids(62), S("62"), r"It is (\d+)/12", agree, ip)
    claim("CORRECTIONS/row10-agreement", sec_ids(39), T(CORRECTIONS), r"\| (\d+)/12 — five episodes disagree", agree, ip)
    icr = norm(live(raw(ROOT / "research/inference-cost/RESULTS.md")))
    claim("inference-cost/RESULTS-agreement", sec_ids(39), icr, r"CORRECTED: (\d+)/12, not 12/12", agree, ip)
    fact("inference-cost/RESULTS-disagreeing-episodes", sec_ids(39), icr, r"episodes 2, 3, 6, 8 and 9 disagree",
         lambda: (disagree() == [2, 3, 6, 8, 9], f"disagreeing episodes {disagree()}"), ip)

    s45 = S("45")
    for arm in ("16", "22"):
        p = f"research/language-post-audit/stage_b_p{arm}.json"
        fact(f"§45/positions-{arm}-complete", sec_ids(45), s45, rf"`positions={arm}`",
             lambda p=p: (lambda e: (e.get("exhausted") is True and e.get("conforming") == 0
                                     and e.get("certificate") == "complete" and e.get("space_size") == 45375,
                                     f"exhausted={e.get('exhausted')} conforming={e.get('conforming')} "
                                     f"certificate={e.get('certificate')} space={e.get('space_size')}"))(_enum(J(p))), p)
    dw = "research/language-post-audit/dyck_witness.json"
    claim("§45/witness-n", sec_ids(45), s45, r"1\.000 on all (\d+) held-out episodes", lambda: dig(J(dw), "heldout_unseen_lengths.n"), dw)


def _first_key(obj, names):
    for sub in _walk(obj):
        if isinstance(sub, dict):
            for n in names:
                if n in sub and isinstance(sub[n], (int, float)) and not isinstance(sub[n], bool):
                    return sub[n]
    raise KeyError(f"none of {names}")


def _walk(obj):
    yield obj
    if isinstance(obj, dict):
        for v in obj.values():
            yield from _walk(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk(v)


def _enum(obj):
    """The enumeration record inside a track file: the first dict with a certificate."""
    for sub in _walk(obj):
        if isinstance(sub, dict) and "certificate" in sub and "exhausted" in sub:
            return sub
    raise KeyError("no enumeration record (certificate + exhausted) in file")


def check_joint_constant_baseline() -> None:
    """§20 / CORRECTIONS row 8: the joint result's constant baseline, recomputed on the
    recorded protocol (split='test', indices 30000-30015, objectives cycled as
    tcn/cli.py:_demo_joint does). Cheap: 32 short episodes, no training."""
    def compute():
        sys.path.insert(0, str(ROOT))
        from examples.joint import trainer            # type: ignore
        from tcn.generation import Host, Action       # type: ignore
        from tcn.types import BOOL, Value             # type: ignore
        cfg = trainer(1).config

        def constant(answer: bool) -> float:
            totals = []
            for i, index in enumerate(range(30000, 30016)):
                host = Host.create(cfg.generator, seed=cfg.seed, index=index, split="test",
                                   configuration=cfg.generator_config | {"horizon": cfg.horizon},
                                   objective=cfg.objectives[i % len(cfg.objectives)])
                for _ in range(cfg.horizon):
                    if host.records[-1].done:
                        break
                    host.step((Action("answer", arguments=(("value", Value.of(BOOL, answer)),)),), cfg.dt)
                totals.append(sum(sum(v.decoded for v in r.reward_components.values()) for r in host.records))
            return sum(totals) / len(totals)
        return constant(True), constant(False)

    ok, val = _run(compute)
    src = "recomputed via tcn.generation.Host on the recorded protocol"
    if not ok:
        record(FAIL, "§20/joint-constant-baseline", f"could not recompute: {val}", sec_ids(20), True)
        return
    t, f = val
    s20 = S("20")
    claim("§20/always-true", sec_ids(20), s20, r"always-True ([\d.]+), always-False", t, src)
    claim("§20/always-false", sec_ids(20), s20, r"always-True 3\.00, always-False ([\d.]+)", f, src)
    claim("CORRECTIONS/row8-always-true", sec_ids(20), T(CORRECTIONS), r"always-True ([\d.]+), always-False", t, src, headline=False)
    claim("§62/always-true", sec_ids(62), S("62"), r"always-True ([\d.]+), always-False", t, src, headline=False)


def checks_perception() -> None:
    """§11, §14a, §16, §21."""
    rc = "research/perception-ladder/out/rung3_colour.json"
    s11 = S("11")
    claim("§11/segmentation-seeds", sec_ids(11), s11, r"(\d+)/12 seeds, exact on training",
          lambda: dig(J(rc), "R2.summary.generalises"), rc)
    fact("§11/resolutions", sec_ids(11), s11, r"at 2x2 and 4x4",
         lambda: (sorted(k for k in J(rc) if k.startswith("R")) == ["R2", "R4"], f"resolution keys {sorted(J(rc))}"), rc)
    claim("§11/segmentation-space", sec_ids(11), s11, r"certifying the solution unique among ([\d,]+) programs",
          lambda: dig(J(rc), "R2.enumeration.space_size") if dig(J(rc), "R2.enumeration.unique") else -1, rc)

    tb = "research/discrete-perception/out/tiebreak.json"
    s14 = S("14a")
    rules = lambda: J(tb)["rows"][0]["rules"]  # noqa: E731
    fact("§14a/validation-filter-identical", sec_ids("14a"), s14,
         r"`validation_filtered` rule returns the identical selection vector as `lexicographic`, with the identical 3 wrong slots of 384",
         lambda: (rules()["validation_filtered"]["selections"] == rules()["lexicographic"]["selections"]
                  and rules()["validation_filtered"]["wrong_slots"] == 3 == rules()["lexicographic"]["wrong_slots"]
                  and rules()["lexicographic"]["slots"] == 384,
                  f"lexicographic {rules()['lexicographic']['wrong_slots']}/{rules()['lexicographic']['slots']} wrong, "
                  f"validation_filtered {rules()['validation_filtered']['wrong_slots']}, same vector: "
                  f"{rules()['validation_filtered']['selections'] == rules()['lexicographic']['selections']}"), tb)
    fact("§14a/only-gradient-exact", sec_ids("14a"), s14, r"only the `gradient` rule reaches `max_error` 0\.0",
         lambda: ([k for k, v in rules().items() if v["max_error"] == 0.0] == ["gradient"],
                  f"rules at max_error 0.0: {[k for k, v in rules().items() if v['max_error'] == 0.0]}"), tb)
    fact("§14a/R8-row-missing", sec_ids("14a"), s14, r"the R=8 row was never written",
         lambda: ([r["resolution"] for r in J(tb)["rows"]] == [4] and J(tb)["arguments"]["resolutions"] == [4, 8],
                  f"rows at resolutions {[r['resolution'] for r in J(tb)['rows']]}, requested {J(tb)['arguments']['resolutions']}"), tb)
    claim("§14a/disagree-R4", sec_ids("14a"), s14, r"([\d.]+)% at R=4, 10\.4% at R=8",
          lambda: J(tb)["rows"][0]["fraction_conforming_that_fail_validation"], tb, scale=100)
    m8 = "research/discrete-perception/out/rung3_mask_r8.json"
    claim("§14a/disagree-R8", sec_ids("14a"), s14, r"5\.19% at R=4, ([\d.]+)% at R=8",
          lambda: (lambda c: (c["conforming"] - c["also_exact_on_held_out"]) / c["conforming"])(
              J(m8)["rows"][0]["identification"]["curve"][-1]), m8, scale=100)
    m4 = "research/discrete-perception/out/rung3_mask.json"
    claim("§14a/mask-conforming-R4", sec_ids("14a"), s14, r"The mask arms have ([\d,]+)-2,608 of 32,000",
          lambda: J(m4)["rows"][0]["identification"]["conforming_train"], m4)
    claim("§14a/mask-conforming-R8", sec_ids("14a"), s14, r"The mask arms have 2,464-([\d,]+) of 32,000",
          lambda: J(m8)["rows"][0]["identification"]["conforming_train"], m8)
    inc = "research/discrete-perception/out/incremental.json"
    counts = lambda: [r["incremental"]["count"] for r in J(inc)["rows"]]  # noqa: E731
    fact("§14a/prefix-walk-counts", sec_ids("14a"), s14, r"finds 2,608, 1,888, 1,664 and 2,608 at R=8, 16, 32 and 64",
         lambda: (counts() == [2608, 1888, 1664, 2608] and [r["resolution"] for r in J(inc)["rows"]] == [8, 16, 32, 64],
                  f"incremental counts {counts()} at R={[r['resolution'] for r in J(inc)['rows']]}"), inc)
    claim("§14a/range-low", sec_ids("14a"), s14, r"the range is ([\d,]+)–2,752", lambda: min(counts()), inc)
    claim("§14a/range-high", sec_ids("14a"), s14, r"the range is 1,664–([\d,]+)",
          lambda: max(c["conforming"] for f in (m4, m8) for c in J(f)["rows"][0]["identification"]["curve"]), f"{m4}, {m8}")
    fact("§14a/gradient-4-of-4-each", sec_ids("14a"), s14, r"4/4 seeds exact at each resolution",
         lambda: (lambda a, b: (a == (4, 4) and b == (4, 4), f"R=4 {a}, R=8 {b} (ok, runs)"))(
             _grad(J(m4)), _grad(J(m8))), f"{m4}, {m8}")
    r4o = "research/discrete-perception/out/rung4_objects.json"
    claim("§14a/colours-per-object", sec_ids("14a"), s14, r"mean ([\d.]+) distinct RGBs",
          lambda: J(r4o)["colour_multiplicity"]["mean_distinct_colours_per_object"], r4o)
    claim("§14a/single-coloured", sec_ids("14a"), s14, r"([\d.]+)% single-coloured",
          lambda: J(r4o)["colour_multiplicity"]["fraction_single_coloured"], r4o, scale=100)
    claim("§14a/depth-train-accuracy", sec_ids("14a"), s14, r"`depth` fits training pixels only to ([\d.]+)",
          lambda: next(c["train_accuracy"] for c in J(r4o)["ceilings"] if c["target"] == "depth_bin"), r4o)
    rw = "research/discrete-perception/out/rung35_window.json"
    claim("§14a/edge-space", sec_ids("14a"), s14, r"then exhaust (\d+) programs in 4\.6 s returning exactly one",
          lambda: J(rw)["staged"]["space_size"], rw)
    fact("§14a/edge-unique", sec_ids("14a"), s14, r"returning exactly one",
         lambda: (J(rw)["staged"]["conforming"] == 1 and J(rw)["staged"]["unique"] is True,
                  f"staged conforming {J(rw)['staged']['conforming']} unique {J(rw)['staged']['unique']}"), rw)

    lw = "research/address-wall/out/landscape.json"
    s16 = S("16")
    im = lambda: J(lw)["index_map"]  # noqa: E731
    claim("§16/local-minima-max", sec_ids(16), s16, r"per-run local minima span 1 to (\d+)",
          lambda: max(r["local_minima"] for r in im()), lw)
    claim("§16/basin-max", sec_ids(16), s16, r"basin widths 0\.09 to ([\d.]+)", lambda: max(r["basin_width"] for r in im()), lw)
    claim("§16/index-runs", sec_ids(16), s16, r"over the (\d+) recorded runs", lambda: len(im()), lw)

    b = "research/object-identity/out/bounds.json"
    s21 = S("21")
    ceil = lambda: J(b)["ceilings"]  # noqa: E731
    RAW_CONTEXTS = ("pixel", "pixel_pos", "pair_right", "cross4", "win3x3", "pixel_agg")
    T3 = ("object_ids", "raster_rank", "is_object_0")
    claim("§21/rows", sec_ids(21), s21, r"holds (\d+) rows, 15 contexts × 14 targets", lambda: len(ceil()), b)
    fact("§21/shape", sec_ids(21), s21, r"15 contexts × 14 targets",
         lambda: ((len({r['context'] for r in ceil()}), len({r['target'] for r in ceil()})) == (15, 14),
                  f"{len({r['context'] for r in ceil()})} contexts × {len({r['target'] for r in ceil()})} targets"), b)
    fact("§21/raw-byte-zero", sec_ids(21), s21, r"exactly 0\.0000 for those three targets at every \*?raw-byte\*? context",
         lambda: (lambda bad: (not bad, f"non-zero raw-byte cells: {bad}" if bad else
                               f"all {len(RAW_CONTEXTS) * 3} raw-byte cells at advantage 0.0"))(
             [(r["context"], r["target"]) for r in ceil() if r["context"] in RAW_CONTEXTS and r["target"] in T3
              and abs(r["advantage"]) > 1e-12]), b)
    fact("§21/nonzero-counts", sec_ids(21), s21, r"non-zero in 8, 8 and 2 of the 15 contexts",
         lambda: (lambda c: (c == [8, 8, 2], f"non-zero contexts per target {dict(zip(T3, c))}"))(
             [sum(1 for r in ceil() if r["target"] == t and abs(r["advantage"]) > 1e-12) for t in T3]), b)
    claim("§21/max-advantage", sec_ids(21), s21, r"up to \+([\d.]+) \(`eqbg_cross4`, `raster_rank`\)",
          lambda: max(r["advantage"] for r in ceil() if r["target"] in T3), b)
    claim("§21/recurrence-pixel", sec_ids(21), s21, r"falls from ([\d.]+) at one pixel",
          lambda: next(r["held_key_recurrence"] for r in ceil() if r["context"] == "pixel" and r["target"] == "object_ids"), b)
    claim("§21/recurrence-3x3", sec_ids(21), s21, r"at one pixel to ([\d.]+) at 3x3",
          lambda: next(r["held_key_recurrence"] for r in ceil() if r["context"] == "win3x3" and r["target"] == "object_ids"), b)
    w3 = "research/object-identity/out/world3d.json"
    claim("§21/world3d-majority", sec_ids(21), s21, r"majority baselines ([\d.]+) \(`world_3d`\)",
          lambda: J(w3)["world_3d"]["collinearity_transfer"]["majority_baseline"], w3)
    claim("§21/world2d-majority", sec_ids(21), s21, r"and ([\d.]+) \(`world_2d`\)",
          lambda: J(w3)["world_2d"]["collinearity_transfer"]["majority_baseline"], w3)
    ap = "research/object-identity/out/apply.json"
    slots = lambda: sum(r["slots"] for r in J(ap)["rows"])  # noqa: E731
    wrong = lambda: sum(r["wrong_slots"] for r in J(ap)["rows"])  # noqa: E731
    claim("§21/apply-slots", sec_ids(21), s21, r"one wrong slot in ([\d,]+)", slots, ap)
    claim("§30/apply-slots", sec_ids(30), S("30"), r"one wrong slot in ([\d,]+) across R=8", slots, ap)
    fact("§30/apply-one-wrong", sec_ids(30), S("30"), r"one wrong slot in 5,520 across R=8, 16, 24 and 32",
         lambda: (wrong() == 1 and sorted(r["resolution"] for r in J(ap)["rows"]) == [8, 16, 24, 32],
                  f"wrong slots {wrong()} of {slots()} at resolutions {[r['resolution'] for r in J(ap)['rows']]}"), ap)
    rs = "research/discrete-perception/out/rung35_window_subsampled.json"
    fact("§14a/sibling-run-zero", sec_ids("14a"), s14, r"returns 0 conforming",
         lambda: (min(_ints_named(J(rs), "conforming")) == 0, f"conforming values {_ints_named(J(rs), 'conforming')}"), rs)
    dc = "research/object-identity/out/discoverable.json"
    fact("§21/callers-discovered", sec_ids(21), s21, r"8 programs, exhausted, unique",
         lambda: (lambda e: (e["space_size"] == 8 and e["exhausted"] and e["unique"],
                             f"space {e['space_size']} exhausted={e['exhausted']} unique={e['unique']}"))(J(dc)["search"]), dc)


def _ints_named(obj, name):
    out = []
    stack = [obj]
    while stack:
        cur = stack.pop()
        if isinstance(cur, dict):
            for k, v in cur.items():
                if k == name and isinstance(v, int) and not isinstance(v, bool):
                    out.append(v)
                stack.append(v)
        elif isinstance(cur, list):
            stack.extend(cur)
    return out


def _grad(d):
    rows = d["rows"][0]["gradient"]["rows"]
    return (sum(1 for r in rows if r["ok"]), len(rows))


def _enum_or(obj, key):
    for sub in _walk(obj):
        if isinstance(sub, dict) and key in sub and "exhausted" in sub:
            return sub[key]
    raise KeyError(key)


def checks_environment_and_language_audit() -> None:
    """§12, §13, §17, §24, §25, §26."""
    s12 = S("12")
    bl = "research/recursive-abstraction-retest/baseline.json"
    claim("§12/solutions", sec_ids(12), s12, r"arm B's contains (\d+) solutions",
          lambda: J(bl)["tight_B_solution_density"]["solutions"], bl)
    claim("§12/chance-tight", sec_ids(12), s12, r"chance rate for that metric is ([\d.]+) on the tight scaffold",
          lambda: J(bl)["tight_B"]["module_on_output_path_rate"], bl)
    claim("§12/chance-wide", sec_ids(12), s12, r"tight scaffold and ([\d.]+) on the wide",
          lambda: J(bl)["wide_B"]["module_on_output_path_rate"], bl)
    e1, e2 = "research/recursive-abstraction/e1_results.json", "research/recursive-abstraction/e2_results.json"
    succ = lambda: sum(1 for f in (e1, e2) for r in J(f)["runs"] if r["arm"] == "B" and r.get("final_conformant"))  # noqa: E731
    claim("§12/track5-successes", sec_ids(12), s12, r"0 of 20 arm-B runs, 0 of (\d+) successes", succ, f"{e1} + {e2}")

    nd = "research/nondegenerate-generalization/out/results.json"
    s13 = S("13")
    arms = ("lookup_ab", "lookup_ba", "lookup_wired_ab", "lookup_wired_ba")
    means = lambda: [sum(r["unseen"] for r in J(nd)[a]["rows"]) / len(J(nd)[a]["rows"]) for a in arms]  # noqa: E731
    claim("§13/unseen-low", sec_ids(13), s13, r"reaches ([\d.]+)[-–]4\.00 on gate families", lambda: min(means()), nd)
    claim("§13/unseen-high", sec_ids(13), s13, r"reaches 3\.38[-–]([\d.]+) on gate families", lambda: max(means()), nd)
    rec = lambda: [sum(r[k] for r in J(nd)[a]["rows"]) / 8 for a in ("record_ab", "record_ba") for k in ("seen", "unseen")]  # noqa: E731
    claim("§13/record-low", sec_ids(13), s13, r"at chance \(([\d.]+)[-–]2\.03 against", lambda: min(rec()), nd)
    claim("§13/record-high", sec_ids(13), s13, r"at chance \(1\.95[-–]([\d.]+) against", lambda: max(rec()), nd)
    claim("§13/best-constant-low", sec_ids(13), s13, r"best constant of ([\d.]+)[-–]2\.56\)",
          lambda: J(nd)["baselines"]["pool_a"]["majority_constant"], nd)
    claim("§13/best-constant-high", sec_ids(13), s13, r"best constant of 2\.13[-–]([\d.]+)\)",
          lambda: J(nd)["baselines"]["pool_b"]["majority_constant"], nd)
    claim("§13/enum-seconds", sec_ids(13), s13, r"exhausts in (\d+) s and reaches 4\.00 held-out",
          lambda: J(nd)["enumeration_lookup_ab"]["seconds"], nd, tol=0.5)
    fact("§13/exhaustion-inferred", sec_ids(13), s13, r"records no `exhausted` or `unique` key",
         lambda: (lambda e: ("exhausted" not in e and "unique" not in e and e["evaluated"] == e["space_size"] == 272,
                             f"keys {sorted(e)}; evaluated {e['evaluated']} of {e['space_size']}"))(J(nd)["enumeration_lookup_ab"]), nd)
    fact("§13/interpreter-8-of-8", sec_ids(13), s13, r"selected in 8/8 seeds in every condition",
         lambda: (all(sum(bool(r["interpreter_selected"]) for r in J(nd)[a]["rows"]) == 8 for a in arms),
                  f"interpreter selected per arm {[sum(bool(r['interpreter_selected']) for r in J(nd)[a]['rows']) for a in arms]}"), nd)

    rp = "research/external-environments/out/replay_check.json"
    s17 = S("17")
    claim("§17/contract-share", sec_ids(17), s17, r"([\d.]+)% of a step is contract", lambda: J(rp)["contract_overhead_fraction"], rp, scale=100)
    claim("§17/mujoco-ms", sec_ids(17), s17, r"— ([\d.]+) ms of MuJoCo inside", lambda: J(rp)["seconds_per_raw_integration"], rp, scale=1000)
    claim("§17/step-ms", sec_ids(17), s17, r"inside a ([\d.]+) ms typed step", lambda: J(rp)["seconds_per_host_step"], rp, scale=1000)
    claim("§17/upright", sec_ids(17), s17, r"reaches upright ([\d.]+) where", lambda: J(rp)["pendulum_upright_max"], rp)
    claim("§17/qvel", sec_ids(17), s17, r"\|qvel\| = ([\d.]+) rad/s", lambda: J(rp)["peak_abs_qvel_during_swing_up"], rp)
    ol = "research/external-environments/out/operator_legality.json"
    fact("§17/byte-operators", sec_ids(17), s17, r"admits `count`, `eq` and `pack`",
         lambda: (lambda legal: (legal == ["count", "eq", "pack"], f"legal on role=byte: {legal}"))(
             sorted(o for o, r in J(ol)["table"]["int[8] role=byte (pixels)"]["operators"].items()
                    if isinstance(r, dict) and r.get("legal"))), ol)
    uncheckable("§17/system-episodes", sec_ids(17),
                "'all three artifacts/system/*/episode.json.gz' (STATUS.md B5 says eleven): artifacts/ is gitignored, "
                "so no committed file records either count")

    la, lp = "research/lesson-audit/stream_legacy.json", "research/lesson-audit/stream_preaudit.json"
    s24 = S("24")
    claim("§24/episodes", sec_ids(24), s24, r"— ([\d,]+) episodes — against a digest", lambda: J(la)["episodes"], la)
    claim("§24/content-identical", sec_ids(24), s24, r"content-identical on (\d+)/179 lessons",
          lambda: sum(J(la)["digests"][k]["content"] == J(lp)["digests"][k]["content"] for k in J(la)["digests"]), f"{la} vs {lp}")
    claim("§24/ids-differ", sec_ids(24), s24, r"instance-id digests differ on (\d+)/179",
          lambda: sum(J(la)["digests"][k]["ids"] != J(lp)["digests"][k]["ids"] for k in J(la)["digests"]), f"{la} vs {lp}")

    ms = "research/core-gradient-fixes/measure.json"
    s25 = S("25")
    spread = lambda k: J(ms)["compare_minimum"]["arms"][k]["loss_spread"]  # noqa: E731
    claim("§25/spread-collapse", sec_ids(25), s25, r"collapsing the loss spread about (\d+)×",
          lambda: spread("tau=1 (shipped, kept)") / spread("tau=256 (carrier, rejected)"), ms)
    lg = "research/core-gradient-fixes/language.json"
    runs = lambda: [(st, arm, len(v["runs"])) for st in ("stage_a", "stage_b") for arm, v in J(lg)[st]["arms"].items()]  # noqa: E731
    claim("§25/language-runs", sec_ids(25), s25, r"records (\d+) gradient runs", lambda: sum(n for _, _, n in runs()), lg)
    claim("§25/language-postfix-runs", sec_ids(25), s25, r"(\d+) of them after the fix",
          lambda: sum(n for _, arm, n in runs() if arm != "before"), lg)
    fact("§25/language-none-conform", sec_ids(25), s25, r"none conforms",
         lambda: (all(v.get("conformant", v.get("exact_train_conformant")) == 0 for st in ("stage_a", "stage_b")
                      for v in J(lg)[st]["arms"].values()), "conformant per arm: "
                  + str({f"{st}/{a}": v.get('conformant', v.get('exact_train_conformant')) for st in ('stage_a', 'stage_b')
                         for a, v in J(lg)[st]['arms'].items()})), lg)

    cv = "research/byte-numeric/out/convolution.json"
    s26 = S("26")
    claim("§26/seeds-exact", sec_ids(26), s26, r"(\d)/4 seeds recover the exact reference kernel",
          lambda: J(cv)["arm_a"]["exact_after_rounding"], cv)
    claim("§26/best-constant", sec_ids(26), s26, r"against ([\d.]+e4) for the best constant",
          lambda: min(J(cv)["arm_a"]["baselines"].values()), cv)
    claim("§26/interpret-space", sec_ids(26), s26, r"The identical ([\d,]+)-program space",
          lambda: J(cv)["arm_b_interpret"]["all_conforming"]["space_size"], cv)
    lgl = "research/byte-numeric/out/legality.json"
    fact("§26/symbol-row-unmeasured", sec_ids(26), s26, r"never measured",
         lambda: ("symbol" not in json.dumps(J(lgl)), "legality.json contains no 'symbol' entry"), lgl)


def checks_cost_line() -> None:
    """§36, §41, §43, §47, §48 and README/STATUS where they repeat them."""
    pyz = lambda n: J(f"research/inference-cost/out/pyz_{n}.json")[0]  # noqa: E731
    s36, s43 = S("36"), S("43")
    band = lambda key, names: [pyz(n)[key] for n in names]  # noqa: E731
    three = ("mixed", "language", "computer")
    claim("§36/rss-low", sec_ids(36), s36, r"(\d+)[-–]60 MB RSS, 51[-–]410 ms cold start for mixed, language and computer",
          lambda: min(band("cold_start_peak_rss_mb", three)), "inference-cost/out/pyz_*.json")
    claim("§36/rss-high", sec_ids(36), s36, r"19[-–](\d+) MB RSS, 51[-–]410 ms", lambda: max(band("cold_start_peak_rss_mb", three)),
          "inference-cost/out/pyz_*.json")
    claim("§36/cold-low", sec_ids(36), s36, r"MB RSS, (\d+)[-–]410 ms cold start", lambda: min(band("cold_start_min_ms", three)),
          "inference-cost/out/pyz_*.json")
    claim("§36/cold-high", sec_ids(36), s36, r"MB RSS, 51[-–](\d+) ms cold start", lambda: max(band("cold_start_min_ms", three)),
          "inference-cost/out/pyz_*.json")
    for label, text, ident in ((sec_ids(36), s36, "§36"), (sec_ids(43), s43, "§43"), (sec_ids(43), T(README), "README")):
        claim(f"{ident}/visual-pyz-rss", label, text, r"`visual\.pyz` is ([\d.]+) MB RSS",
              lambda: pyz("visual")["cold_start_peak_rss_mb"], "inference-cost/out/pyz_visual.json", headline=ident == "§36")
        claim(f"{ident}/visual-pyz-cold", label, text, r"`visual\.pyz` is 377\.5 MB RSS and ([\d,]+) ms cold start",
              lambda: pyz("visual")["cold_start_min_ms"], "inference-cost/out/pyz_visual.json", headline=ident == "§36")
    lat = "research/neural-baselines/out/latency.json"
    imp = lambda: [r["import_torch_ms"] for r in J(lat)["cold_process"]]  # noqa: E731
    claim("§43/torch-import-low", sec_ids(43), s43, r"~270 MB and ([\d.]+)[-–]9\.59 s just to import torch", lambda: min(imp()), lat, scale=1e-3)
    claim("§43/torch-import-high", sec_ids(43), s43, r"~270 MB and 0\.83[-–]([\d.]+) s just to import torch", lambda: max(imp()), lat, scale=1e-3)
    claim("§43/torch-rss", sec_ids(43), s43, r"larger than the ([\d.]+) MB torch baseline",
          lambda: max(r["peak_rss_mb"] for r in J(lat)["cold_process"]), lat)
    sz = lambda: next(e for e in J("research/inference-cost/out/sizes.json") if e["name"] == "visual")  # noqa: E731
    claim("§36/gzip-kb", sec_ids(36), s36, r"It gzips to (\d+) KB", lambda: sz()["json_gzip_bytes"] / 1000, "inference-cost/out/sizes.json")
    claim("§36/gzip-ratio", sec_ids(36), s36, r"minified-then-gzipped JSON, (\d+)× the `.pyz`",
          lambda: sz()["gzip_ratio_vs_pyz"], "inference-cost/out/sizes.json")
    claim("§36/type-share", sec_ids(36), s36, r"of which ([\d.]+)% is repeated type declarations",
          lambda: sz()["type_declaration_share_of_minified"], "inference-cost/out/sizes.json", scale=100)
    claim("§36/minified-mb", sec_ids(36), s36, r"share of the ([\d.]+) MB minified JSON", lambda: sz()["json_minified_bytes"] / 1e6,
          "inference-cost/out/sizes.json")
    pf = "research/inference-cost/out/profile.json"
    share = lambda: J(pf)["by_role_share"]  # noqa: E731
    claim("§36/marshalling-share", sec_ids(36), s36, r"round-trip account for ([\d.]+)% of execution",
          lambda: sum(v for k, v in share().items() if k not in ("operator semantics + graph walk", "other")), pf, scale=100)
    claim("§36/four-roles-share", sec_ids(36), s36, r"the four roles first named alone are ([\d.]+)%",
          lambda: sum(v for k, v in share().items() if k.startswith(("Type.", "validate_raw", "numeric/type"))), pf, scale=100)
    claim("§36/operator-seconds", sec_ids(36), s36, r"graph walk are ([\d.]+) s of 56\.87 s",
          lambda: J(pf)["by_role_seconds"]["operator semantics + graph walk"], pf)
    claim("§36/operator-share", sec_ids(36), s36, r"of 56\.87 s — ([\d.]+)%",
          lambda: J(pf)["by_role_share"]["operator semantics + graph walk"], pf, scale=100)

    tv = "research/neural-baselines/out/tcn_visual.json"
    eps = lambda: J(tv)["quality"]["episodes"]  # noqa: E731
    for text, ident, secs in ((s43, "§43", sec_ids(43)), (T(README), "README", sec_ids(43))):
        claim(f"{ident}/rectangles", secs, text, r"(\d+)/227 rectangles", lambda: sum(e["widgets_in_probe"] for e in eps()), tv,
              headline=ident == "§43")
        claim(f"{ident}/links", secs, text, r"(\d+)/227 parent links", lambda: sum(e["parent_links_correct"] for e in eps()), tv,
              headline=ident == "§43")
        claim(f"{ident}/trees", secs, text, r"(\d+)/12 exact trees", lambda: sum(1 for e in eps() if e["tree_exact"]), tv,
              headline=ident == "§43")
        claim(f"{ident}/widgets-min", secs, text, r"screens of (\d+)[-–]20 widgets", lambda: min(e["widgets_in_probe"] for e in eps()), tv,
              headline=ident == "§43")

    att, bench = "research/compiled-runtime/out/attribute_visual.json", "research/compiled-runtime/out/bench_visual.json"
    ops = lambda: J(att)["ratios"]["C_over_D_opcodes"]  # noqa: E731
    per = lambda: J(att)["ns_per_opcode"]["C"] / J(att)["ns_per_opcode"]["D"]  # noqa: E731
    s48 = S("48")
    claim("§48/gap", sec_ids(48), s48, r"\| 0\.1861 ms \| 3,359x \| ([\d.]+)x \|", lambda: J(bench)["ratios"]["C_over_D_program"], bench)
    claim("§48/opcodes", sec_ids(48), s48, r"decomposes as ([\d.]+)x more elementary operations", ops, att)
    claim("§48/per-bytecode", sec_ids(48), s48, r"times ([\d.]+)x per-bytecode cost", per, att)
    for text, ident, secs in ((s48, "§48", sec_ids(48)), (S("41"), "§41", sec_ids(41)), (T(README), "README", sec_ids(48)),
                              (T(STATUS), "STATUS", sec_ids(48))):
        claim(f"{ident}/decomposition-product", secs, text, r"multiply to ([\d.]+)[x×]?,? `attribute_visual\.json`'s own",
              lambda: ops() * per(), att, headline=ident == "§48")

    s41 = S("41")
    sw = lambda: J("research/program-length/out/span_sweep.json")["rows"] + J("research/program-length/out/span_sweep_fine.json")["rows"]  # noqa: E731
    fact("§41/spans-enumerated", sec_ids(41), s41, r"at spans 4, 8, 12, 16, 20, 24, 28 and 29",
         lambda: (lambda none: (none == [4, 8, 12, 16, 20, 24, 28, 29], f"spans certified 'none exists': {none}"))(
             sorted(r["span"] for r in sw() if r["certificate"] == "none exists" and r["exhausted"])), "program-length/out/span_sweep*.json")
    fact("§41/unique-at-30", sec_ids(41), s41, r"`unique` at 30",
         lambda: (any(r["span"] == 30 and r["certificate"] == "unique" for r in sw()), "span 30 certificate unique"),
         "program-length/out/span_sweep_fine.json")

    q2 = "research/dyck-learnability/out/q2_summary.json"
    arms = lambda: J(q2)["arms"] if isinstance(J(q2), dict) else J(q2)  # noqa: E731
    s47 = S("47")

    def pairs():
        by = {(a["tau_lt"], a["steps"], a["scaffold"]): a for a in arms()}
        keys = sorted({(k[0], k[1]) for k in by})
        same = [k for k in keys if [r["c"] for r in by[(*k, "dyck")]["chosen_per_seed"]] ==
                [r["c"] for r in by[(*k, "counting")]["chosen_per_seed"]]]
        return same, keys
    claim("§47/identical-pairs", sec_ids(47), s47, r"identical `c` at every seed in (\d) of 4 arm pairs", lambda: len(pairs()[0]), q2)
    fact("§47/identical-medians", sec_ids(47), s47, r"identical median 0\.4738",
         lambda: (len({a["median_heldout_unseen"] for a in arms()}) == 1, f"medians {sorted({a['median_heldout_unseen'] for a in arms()})}"), q2)
    fact("§47/symbols-gradient", sec_ids(47), s47, r"exactly 0\.0 at `tau_lt=0`",
         lambda: (all((a["first_step_choice_gradients_seed0"]["symbols"] == 0.0) == (a["tau_lt"] == 0) for a in arms()),
                  "symbols first-step gradient: " + str(sorted({(a['tau_lt'], a['scaffold'], a['first_step_choice_gradients_seed0']['symbols'])
                                                                for a in arms()}))), q2)


def checks_library_line() -> None:
    """§44, §46, §52, §53, §54, §55, §56, §57, §58."""
    s44 = S("44")
    orr = "research/earned-abstraction/out/order_robustness.json"
    claim("§44/maj3-declared-order", sec_ids(44), s44, r"a `MAJ3` body survives in (\d) of 6 solved programs",
          lambda: next(o["tasks_containing_a_maj3_body"] for o in J(orr)["orders"] if o["order"] == ["and", "or", "xor"]), orr)
    claim("§44/maj3-other-order", sec_ids(44), s44, r"(\d) of 6 when `or` is declared first",
          lambda: max(o["tasks_containing_a_maj3_body"] for o in J(orr)["orders"]), orr)
    for arm, want in (("arm1_none", 230400), ("arm3_authored", 2709504)):
        p = f"research/earned-abstraction/out/enum_tight_{arm}.json"
        fact(f"§44/{arm}-exhausted", sec_ids(44), s44, rf"{want:,}",
             lambda p=p, want=want: (lambda e: (e["exhausted"] and e["certificate"] == "complete" and e["space_size"] == want,
                                                f"space {e['space_size']} exhausted={e['exhausted']} certificate={e['certificate']} "
                                                f"conforming={e.get('conforming')}"))(_enum(J(p))), p)

    s46 = S("46")
    pc = "research/premin-abstraction/out/proposal_C-plus1.json"
    claim("§46/plus1-rank", sec_ids(46), s46, r"`C-plus1` \| 166 \| (\d+) \|", lambda: J(pc)["best_maj3_rank"], pc)
    claim("§46/plus1-rank-prose", sec_ids(46), s46, r"ranks it 11th to (\d+)th", lambda: J(pc)["best_maj3_rank"], pc, headline=False)

    s52 = S("52")
    qp = "research/class-identity/out/q2_partition.json"
    fact("§52/one-class-two-names", sec_ids(52), s52, r"one class \(`tt/3/57`",
         lambda: (sorted(J(qp)["classes_with_multiple_members"].get("tt/3/57", [])) ==
                  ["ca785ec948250acbf39b8f66", "ec516b3808ddef7e7d0e7d22"],
                  f"tt/3/57 members {J(qp)['classes_with_multiple_members'].get('tt/3/57')}"), qp)

    s53 = S("53")
    ex = "research/depth-encoding/out/exhaust.json"
    fact("§53/unique-at-six-depths", sec_ids(53), s53, r"\| 8 \| 24 \| 272 \| 272 \| true \| 1 \| `unique` \|",
         lambda: (lambda rows: (all(r["space_size"] == 272 and r["exhausted"] and r["conforming"] == 1 and r["certificate"] == "unique"
                                    for r in rows.values()) and sorted(int(k) for k in rows) == [1, 2, 3, 4, 6, 8],
                                f"depths {sorted(int(k) for k in rows)} all unique over 272"))(J(ex)), ex)
    ws = "research/depth-encoding/out/wrong_schema.json"
    fact("§53/armF", sec_ids(53), s53, r"depths 2, 3, 4, 6, 8 \| 272 \| true \| 0 \| `complete`",
         lambda: (lambda e: (e["1"]["certificate"] == "unique" and all(e[k]["conforming"] == 0 for k in e if k != "1"),
                             "wrong schema: " + str({k: (e[k]['conforming'], e[k]['certificate']) for k in e})))(J(ws)["exhaust"]), ws)
    rec, base = "research/depth-encoding/out/record.json", "research/depth-encoding/out/baselines.json"
    above = lambda: [(d, v["mean"]) for t in J(rec)["transfer"] for d, v in t["per_depth"].items()  # noqa: E731
                     if v["mean"] > J(base)["held_out"][d]["best_constant"]]
    claim("§53/criterion-4-cells", sec_ids(53), s53, r"(\d+) of 24 cells sit above", lambda: len(above()), f"{rec} vs {base}")

    s54 = S("54")
    rk = "research/reuse-ranking/out/ranked.json"
    digests = lambda obj: {v["tables"][obj]["rank1"]["digest"] for v in J(rk)["corpora"].values()}  # noqa: E731
    fact("§54/o1-digests", sec_ids(54), s54, r"flips — three digests",
         lambda: (len(digests("O1")) == 3, f"O1 rank-1 digests across six corpora: {sorted(d[:12] for d in digests('O1'))}"), rk)
    fact("§54/o2-stable", sec_ids(54), s54, r"one, stable `165bc290d9c8`",
         lambda: (digests("O2") == {"165bc290d9c82b70a8ea3cc2"}, f"O2 rank-1 digests {sorted(digests('O2'))}"), rk)
    fact("§54/b2-flips", sec_ids(54), s54, r"B2 flips",
         lambda: (len(digests("B2")) > 1 and len(digests("B1")) == 1,
                  f"B1 digests {len(digests('B1'))}, B2 digests {len(digests('B2'))}"), rk)

    s55 = S("55")
    wi, wo = "research/class-identity/out/with.json", "research/class-identity/out/without.json"
    fact("§55/instantiation-depths", sec_ids(55), s55, r"at depths 5, 7 and 12",
         lambda: (sorted(int(k) for k in J(wi)["per_width"]) == [5, 7, 12]
                  and all(v["programs_evaluated"] == 1 and v["episodes"] == 64 and v["certificate"] == "none"
                          for v in J(wi)["per_width"].values()),
                  "with.json per_width keys (depths) " + str(sorted(J(wi)['per_width']))), wi)

    s56 = S("56")
    lr = "research/residual-gap/out/latency_run1.json"
    rr = lambda k: J(lr)["ratios"]["deploy"][k]["ratio"]  # noqa: E731
    total = lambda: math.prod(rr(k) for k in ("R0_over_R1", "R1_over_R2", "R2_over_R3", "R3_over_R4", "R4_over_R5",  # noqa: E731
                                              "R5_over_R6", "R6_over_R7"))
    claim("§56/guard-rung", sec_ids(56), s56, r"typed-guard elimination \(range \+ index checks\) \| ([\d.]+)×",
          lambda: rr("R4_over_R5"), lr)
    claim("§56/r5-r6-factor", sec_ids(56), s56, r"loop-bound strength reduction \| ([\d.]+)× \*?\(a regression\)",
          lambda: rr("R5_over_R6"), lr)
    claim("§56/r5-r6-share", sec_ids(56), s56, r"\(a regression\)\*? \| ([−\-\d.]+)%",
          lambda: 100 * math.log(rr("R5_over_R6")) / math.log(total()), lr)

    s57 = S("57")
    ae = "research/second-family/out/arms_enum.json"
    dig57 = lambda arm: {r["module"] for r in J(ae) if r.get("band") == "C-minall" and r.get("arm") == arm}  # noqa: E731
    fact("§57/shared-digest", sec_ids(57), s57, r"`arm4s_runnerup` and `arm4s_matched` share digest `1e8b0e63e5f636a7e47b08cd`",
         lambda: (dig57("arm4s_runnerup") == dig57("arm4s_matched") == {"module:1e8b0e63e5f636a7e47b08cd"},
                  f"runnerup {dig57('arm4s_runnerup')}, matched {dig57('arm4s_matched')}"), ae)
    rkm = "research/second-family/out/ranked_C-minall.json"
    open_item("§57/C-minall-11-of-15", sec_ids(57), "M11",
              "§57's 'C-minall: O1 11 of 15, O2 11 of 15' does not reconstruct: ranked_C-minall.json puts the window "
              "at rank 1 in " + str(sum(1 for v in J(rkm)["corpora"].values() if v["tables"]["O1"].get("best_window_rank") == 1))
              + " of 6 corpora under O1, and no committed file records a 15-task denominator. Left open, marked in §57.")

    rpt = "research/baselines/out/tcn_mixed/report.json"
    s10 = S("10")
    claim("§10/relaxed-loss", sec_ids(10), s10, r"`relaxed_loss` ([\d.]+e-\d+) and `exact_max_error` 0\.0", lambda: J(rpt)["loss"], rpt)
    fact("§10/exact", sec_ids(10), s10, r"`exact_max_error` 0\.0", lambda: (J(rpt)["exact_conformance"] is True,
                                                                          f"exact_conformance={J(rpt)['exact_conformance']}"), rpt)
    run = "research/code-generation/out/run.json"
    s34 = S("34")
    claim("§34/projected-seconds", sec_ids(34), s34, r"`projected_seconds` is ([\d,.]+) s", lambda: _first_key(J(run), ("projected_seconds",)), run)
    claim("§34/projected-hours", sec_ids(34), s34, r"80,530\.6 s = ([\d.]+) h", lambda: _first_key(J(run), ("projected_seconds",)) / 3600, run)
    claim("§34/stage-a-space", sec_ids(34), s34, r"space (\d+), exhausted, 2 conforming", lambda: J(run)["stage_a"]["space_size"], run)
    fact("§34/stage-a-two", sec_ids(34), s34, r"space 432, exhausted, 2 conforming",
         lambda: (J(run)["stage_a"]["conforming"] == 2 and J(run)["stage_a"]["exhausted"] is True,
                  f"stage_a conforming {J(run)['stage_a']['conforming']} exhausted {J(run)['stage_a']['exhausted']}"), run)
    dep = "research/lazy-latency/out/deployment.json"
    s51 = S("51")
    st3 = lambda: J(dep)["stage3"]  # noqa: E731
    claim("§51/cold-start", sec_ids(51), s51, r"cold-starts ([\d.]+)× faster",
          lambda: st3()["A"]["cold_start_ms_median"] / st3()["B"]["cold_start_ms_median"], dep)
    claim("§51/cold-start-min", sec_ids(51), s51, r"([\d.]+)× by minimum",
          lambda: st3()["A"]["cold_start_ms_min"] / st3()["B"]["cold_start_ms_min"], dep)
    claim("§51/disk", sec_ids(51), s51, r"B is ([\d.]+)× smaller on disk",
          lambda: st3()["A"]["bytes_on_disk"] / st3()["B"]["bytes_on_disk"], dep)
    pf = "research/abstraction-preference/prune_fixtures.json"
    claim("§14b/prune-rows", sec_ids("14b"), S("14b"), r"semantics over ([\d,]+) row comparisons", lambda: sum(r["rows_compared"] for r in J(pf)), pf)
    fact("§14b/prune-ok", sec_ids("14b"), S("14b"), r"on every fixture",
         lambda: (all(r["ok"] for r in J(pf)), f"{len(J(pf))} fixtures, all ok={all(r['ok'] for r in J(pf))}"), pf)
    ex = "research/depth-generalization/out/exactness.json"
    s15 = S("15")
    claim("§15/episodes-per-depth", sec_ids(15), s15, r"on (\d+) episodes at each of depths",
          lambda: (lambda d: d["1"]["episodes"] if all(v["episodes"] == d["1"]["episodes"] for v in d.values()) else -1)(J(ex)["depths"]), ex)
    fact("§15/depths-exact", sec_ids(15), s15, r"at each of depths 1, 2, 3, 4, 6 and 8",
         lambda: (sorted(int(k) for k in J(ex)["depths"]) == [1, 2, 3, 4, 6, 8]
                  and all(v["settled_exact"] == v["episodes"] for v in J(ex)["depths"].values()),
                  "depths " + str(sorted(int(k) for k in J(ex)['depths'])) + ", all settled exact"), ex)
    fact("§15/settles-at-d-minus-1", sec_ids(15), s15, r"settling at exactly tick d-1",
         lambda: (all(v["first_settled_tick_max"] == int(k) - 1 for k, v in J(ex)["depths"].items()),
                  "first_settled_tick_max per depth " + str({k: v['first_settled_tick_max'] for k, v in J(ex)['depths'].items()})), ex)
    cs, cl = "research/computer-capability/out/search.json", "research/computer-capability/out/closed_loop.json"
    s23 = S("23")
    claim("§23/held-out-solved", sec_ids(23), s23, r"held-out documents \| 10 \| [\d.]+ \((\d+)/10 solved\)", lambda: J(cl)["A_unseen_documents"]["solved"], cl)
    claim("§23/always-5", sec_ids(23), s23, r"always write \"5\" \| 10 \| ([\d.]+) \|", lambda: J(cl)["B_baselines"][0]["mean_return"], cl)
    claim("§23/transform-space", sec_ids(23), s23, r"Enumeration exhausted ([\d,]+) programs in 168 s", lambda: J(cs)["transform"]["space_size"], cs)
    claim("§23/transform-conforming", sec_ids(23), s23, r"programs in 168 s with (\d+) conforming", lambda: J(cs)["transform"]["enumeration"]["conforming"], cs)
    claim("§23/address-candidates", sec_ids(23), s23, r"in a (\d+)-candidate space", lambda: J(cs)["transform"]["candidate_counts"]["pos"], cs)
    r2 = "research/visual-ladder/out/rung2.json"
    s33 = S("33")
    claim("§33/pairs", sec_ids(33), s33, r"over all ([\d,]+) held-out pairs", lambda: J(r2)["relation_bound"]["held_pairs"], r2)
    claim("§33/accuracy", sec_ids(33), s33, r"held-out pairs it is ([\d.]+)", lambda: J(r2)["relation_bound"]["key_equality_accuracy"], r2)
    claim("§33/always-different", sec_ids(33), s33, r"always-different baseline of ([\d.]+)", lambda: J(r2)["relation_bound"]["always_different_baseline"], r2)
    lg1 = "research/lazy-guard/out/stage1.json"
    s49 = S("49")
    claim("§49/stage1-candidates", sec_ids(49), s49, r"candidates enumerated \| ([\d,]+) \|", lambda: J(lg1)["search"]["candidates_considered"], lg1)
    claim("§49/stage1-domain", sec_ids(49), s49, r"\| exactness \| ([\d,]+) / 65,536", lambda: J(lg1)["domain_size"], lg1)

    SEASONS = ("seasons", "origin/seasons")
    sj = "research/seasons/results/joint-40.jsonl"
    s37 = S("37")

    def arm_mean(name):
        rows = [r for r in JB(SEASONS, sj) if r.get("arm") == name and r.get("post_crystallization")]
        if not rows:
            raise KeyError(f"no '{name}' rows; arms present: {sorted({r.get('arm') for r in JB(SEASONS, sj)})}")
        return sum(r["post_crystallization"]["mean_return"] for r in rows) / len(rows)
    claim("§37/shipped", sec_ids(37), s37, r"\| shipped crystallizer \| ([\d.]+) \|", lambda: arm_mean("shipped"), f"{sj} (branch seasons)")
    claim("§37/seasons", sec_ids(37), s37, r"\| seasons \| ([\d.]+) \|", lambda: arm_mean("seasons"), f"{sj} (branch seasons)")

    EG = ("research/emitter-guards", "origin/emitter-guards")
    rz = "research/emitter-guards/out/residue.json"
    s59 = S("59")
    elim = lambda: sum(v["stats"]["guard_range_eliminated"] for v in JB(EG, rz).values())  # noqa: E731
    total_g = lambda: sum(v["stats"]["guard_range_eliminated"] + v["stats"]["guard_range_emitted"] for v in JB(EG, rz).values())  # noqa: E731
    claim("§59/eliminated", sec_ids(59), s59, r"declared types alone: (\d+) of 282", elim, f"{rz} (branch research/emitter-guards)")
    claim("§59/total", sec_ids(59), s59, r"declared types alone: 206 of (\d+)", total_g, f"{rz} (branch research/emitter-guards)")

    RB = ("worktree-agent-a0350f1fdeac805b2", "origin/refinement-bounds")
    sd = "research/refinement-bounds/out/soundness.json"
    s61 = S("61")
    claim("§61/pairs", sec_ids(61), s61, r"verified over ([\d,]+) pairs with 0 mismatches",
          lambda: JB(RB, sd)["clamp_identity"]["full_rectangle_0_to_L"]["pairs"], f"{sd} (branch refinement-bounds)")
    claim("§61/a-plus-2", sec_ids(61), s61, r"carries `a`'s type and reaches (\d+)", lambda: JB(RB, sd)["root"]["derived_a_plus_2_max"],
          f"{sd} (branch refinement-bounds)")
    claim("§61/preclamp", sec_ids(61), s61, r"pre-clamp value from ([\d,]+) to 3,069",
          lambda: JB(RB, sd)["clamp_identity"]["full_rectangle_0_to_L"]["max_preclamp_value"], f"{sd} (branch refinement-bounds)")
    uncheckable("§23/configuration-count", sec_ids(23),
                "'across 27 configurations' — no committed file records a configuration count (audit L24)")
    open_item("§46/79x", sec_ids(46), "L16",
              "§46's 'k+1 half cost 79× more': out/corpora.json's DFS counts give 72× (166,506,943 / 2,307,181) and no "
              "committed file records the CPU ratio, so 79× can be neither confirmed nor corrected")
    open_item("§15/soft-floor", sec_ids(15), "L19",
              "§15's soft-model '3.20-3.31': the audit reports a floor of 3.19, but this gate found no soft-score field in "
              "depth-generalization/out/ to confirm or correct either figure")
    open_item("§19/overlap-subset", sec_ids(19), "L25",
              "§19's '0% string overlap' — the audit reports it is certified only for an n=242 subset at lengths "
              "8-14; this gate found no committed file under research/language-capability/ that records overlap "
              "for either set, so the claim can be neither confirmed nor corrected here")
    s58 = S("58")
    for band, want in (("C-trace", 10), ("C-minall", 7)):
        p = f"research/cross-domain/out/control_3_{band}.json"
        claim(f"§58/control-{band}", sec_ids(58), s58, rf"(\d+) non-trivial shared classes on the `{band}` band" if band == "C-trace"
              else r"and (\d+) on `C-minall`", lambda p=p: J(p)["P1_boolean_control"]["cross_family_D"]["n_nontrivial"], p)
    claim("§58/control-S-rule", sec_ids(58), s58, r"the `S` identity rule gives (\d+) on both bands",
          lambda: J("research/cross-domain/out/control_3_C-trace.json")["P1_boolean_control"]["cross_family_S"]["n_nontrivial"],
          "research/cross-domain/out/control_3_C-trace.json")


def _rows(d):
    return d if isinstance(d, list) else d.get("rows", d)


def checks_early_sections() -> None:
    """§1-§9 (the original eight-track summary), and sections whose figures sit in a
    single track file: §18, §22, §27, §29, §31, §32, §35, §38, §40, §50, §60, §63."""
    fh, fd = "research/search-scaling/FH_hard.json", "research/search-scaling/FD_supervision.json"
    depths = (3, 4, 6, 8, 12, 16)
    rate = lambda p: [sum(r["success"] for r in _rows(J(p)) if r["config"]["wiring"] == "free"  # noqa: E731
                          and r["config"]["depth"] == k) / 16 for k in depths]
    s1, s2, s3 = S("1"), S("2"), S("3")
    claim("§1/free-low", sec_ids(1), s1, r"from (\d+)-38% to 88-94%", lambda: min(rate(fh)), fh, scale=100)
    claim("§1/free-high", sec_ids(1), s1, r"from 19-(\d+)% to 88-94%", lambda: max(rate(fh)), fh, scale=100)
    claim("§1/dense-low", sec_ids(1), s1, r"19-38% to (\d+)-94%", lambda: min(rate(fd)), fd, scale=100)
    claim("§1/dense-high", sec_ids(1), s1, r"19-38% to 88-(\d+)%", lambda: max(rate(fd)), fd, scale=100)
    tm, mb = "research/baselines/out/tcn_mixed_eval.json", "research/baselines/out/mixed_baseline.json"
    claim("§2/tcn-interp", sec_ids(2), s2, r"Track 6: ([\d.]+e-\d+) interpolating", lambda: J(tm)["interp"]["rmse"], tm)
    claim("§2/tcn-extrap", sec_ids(2), s2, r"interpolating and ([\d.]+e-\d+) extrapolating", lambda: J(tm)["extrap"]["rmse"], tm)
    claim("§2/mlp-interp", sec_ids(2), s2, r"tuned MLP's ([\d.]+e-\d+) and", lambda: dig(J(mb), "selected.answer.interp_rmse"), mb)
    claim("§2/mlp-extrap", sec_ids(2), s2, r"tuned MLP's [\d.]+e-\d+ and ([\d.]+)\.", lambda: dig(J(mb), "selected.answer.extrap_rmse"), mb)
    claim("§2/widest-node", sec_ids(2), s2, r"at ([\d,]+) candidates on the widest node",
          lambda: max(max(r["n_candidates"]) for r in _rows(J(fd))), fd)
    ma = "research/crystallization-ablation/results/mixed-30.jsonl"
    arm = lambda a: [r for r in J(ma) if r["arm"] == a]  # noqa: E731
    claim("§3/scheduler-3-of-16", sec_ids(3), s3, r"the scheduler conforms (\d+)/16", lambda: sum(r["exact_conformance"] for r in arm("A")), ma)
    claim("§3/argmax-16-of-16", sec_ids(3), s3, r"argmax conforms (\d+)/16", lambda: sum(r["exact_conformance"] for r in arm("B")), ma)
    claim("§3/discarded-steps", sec_ids(3), s3, r"discarding ([\d.]+)% of its optimizer steps",
          lambda: 1 - sum(r["base_steps"] + r.get("retained_trial_steps", 0) for r in arm("A")) / sum(r["total_steps"] for r in arm("A")),
          ma, scale=100)
    tj = "research/baselines/out/tcn_joint_eval.json"
    claim("§3/joint-bits", sec_ids(3), s3, r"joint ships ([\d,]+) bits to learn", lambda: J(tj)["description_bits"], tj)
    sm = "research/recursive-abstraction/summary.json"
    claim("§3/track5-successes", sec_ids(3), s3, r"including 0 of (\d+) successes",
          lambda: sum(int(a["success"].split("/")[0]) for e in ("e1", "e2") for a in J(sm)[e] if a["arm"] == "B module"), sm)
    cd = "research/crystallization-ablation/results/conformance-diagnosis-{}.json"
    share = lambda n: (lambda h: h["mismatch ONLY on untouched still-soft nodes (checker artifact)"] / sum(h.values()))(  # noqa: E731
        J(cd.format(n))["mismatch_role_histogram"])
    s4 = S("4")
    claim("§4/f-conf-low", sec_ids(4), s4, r"correct\. (\d+)-74% of rejections", lambda: share(30), cd.format(30), scale=100)
    claim("§4/f-conf-high", sec_ids(4), s4, r"correct\. 57-(\d+)% of rejections", lambda: share(50), cd.format(50), scale=100)
    cc = "research/recursive-abstraction/candidate_cost.json"
    s5 = S("5")
    # "16-98x": the stored minimum is 16.525 — the text truncates rather than rounds; a
    # rounding-style difference, not an error (the audit's manufactured-discrepancy rule).
    claim("§5/module-cost-low", sec_ids(5), s5, r"module candidates (\d+)-98x costlier", lambda: min(v["ratio"] for v in J(cc).values()),
          cc, tol=0.6)
    claim("§5/module-cost-high", sec_ids(5), s5, r"module candidates 16-(\d+)x costlier", lambda: max(v["ratio"] for v in J(cc).values()), cc)
    mc = "research/enumerative-baseline/out/mixed_constants.json"
    claim("§5/constant-miss", sec_ids(5), s5, r"tolerance by ([\d.]+e-\d+) purely", lambda: dig(J(mc), "gradient_fairness_sweep.0.k_error"), mc)
    eb = "research/enumerative-baseline/out/"
    meth = lambda f, m: next(r for r in J(eb + f)["results"] if r["method"] == m)  # noqa: E731
    s7 = S("7")
    claim("§7/joint-enumeration-ms", sec_ids(7), s7, r"settles in ([\d.]+) ms \(143 ms",
          lambda: meth("joint.json", "exhaustive_enumeration_probe_full")["seconds"], eb + "joint.json", scale=1000)
    claim("§7/joint-runtime-ms", sec_ids(7), s7, r"\((\d+) ms if every candidate",
          lambda: meth("headline.json", "exhaustive_enumeration_probe_tcn_runtime_full")["seconds"], eb + "headline.json", scale=1000)
    sh = eb + "scaling_hard_d3_4_5_6.json"
    gsr = lambda k: [r["gradient"]["success_rate"] for r in J(sh)["rows"] if r["depth"] == k]  # noqa: E731
    claim("§7/depth3", sec_ids(7), s7, r"at depth 3 it succeeds ([\d.]+) of the time", lambda: sum(gsr(3)) / len(gsr(3)), sh)
    claim("§7/depth4", sec_ids(7), s7, r"at depth 4 it succeeds ([\d.]+)\.", lambda: sum(gsr(4)) / len(gsr(4)), sh)
    sg = eb + "scaling_generator_d1_2_3_4_5_6.json"
    dens = lambda k: __import__("statistics").median(r["solution_density"] for r in J(sg)["rows"] if r["depth"] == k)  # noqa: E731
    claim("§7/density-d1", sec_ids(7), s7, r"\(([\d.]+e-\d+) at depth 1 to", lambda: dens(1), sg)
    claim("§7/density-d6", sec_ids(7), s7, r"at depth 1 to ([\d.]+e-\d+) at depth 6", lambda: dens(6), sg)
    s8 = S("8")
    claim("§8/gradient-5-of-5", sec_ids(8), s8, r"the gradient path succeeds (\d+)/5 at a budget",
          lambda: sum(r["solved"] for r in J(eb + "joint.json")["results"] if r["method"] == "gradient_tcn"), eb + "joint.json")
    claim("§8/random-0-of-20", sec_ids(8), s8, r"the same candidates succeeds (\d+)/20",
          lambda: sum(r["solved"] for r in J(eb + "joint.json")["results"] if r["method"] == "random_search_env_matched"), eb + "joint.json")

    gb = "research/gui-hierarchy/out/bounds.json"
    s18 = S("18")
    claim("§18/edge-majority", sec_ids(18), s18, r"oracle 1\.0000 against a ([\d.]+) majority",
          lambda: J(gb)["rung1_edge"]["flat (default) | x"]["majority_baseline"], gb)
    claim("§18/kind-oracle", sec_ids(18), s18, r"\(([\d.]+) against 0\.2982\)",
          lambda: J(gb)["rung3_kind"]["colour_mode=random | fill colour"]["oracle_accuracy"], gb)
    for name, key, pat in (("narrow", "narrow", r"\(([\d,]+) / 13,056 / 81,920 programs\)"),
                           ("wide", "wide_offsets_ablation", r"\(1,280 / ([\d,]+) / 81,920 programs\)"),
                           ("free", "free_operand_ablation", r"\(1,280 / 13,056 / ([\d,]+) programs\)")):
        p = f"research/gui-hierarchy/out/rung1_{name}.json"
        claim(f"§18/space-{name}", sec_ids(18), s18, pat, lambda p=p, key=key: J(p)[key]["space_size"], p)

    rf = "research/policy-learning/out/refs.json"
    s22 = S("22")
    claim("§22/always-false", sec_ids(22), s22, r"against always-false ([\d.]+), always-true", lambda: J(rf)["4"]["mean_return"]["always_false"], rf)
    claim("§22/always-true", sec_ids(22), s22, r"always-true ([\d.]+), uniform", lambda: J(rf)["4"]["mean_return"]["always_true"], rf)
    claim("§22/uniform", sec_ids(22), s22, r"uniform ([\d.]+) and an oracle", lambda: J(rf)["4"]["mean_return"]["uniform"], rf)
    cn = "research/policy-learning/out/cancellation.json"
    claim("§22/zero-jacobian", sec_ids(22), s22, r"gradient (-[\d.]+e-\d+)\)", lambda: J(cn)["0.3,0.7"]["d_da"], cn)

    bs = "research/discrete-backend/out/beam_search.json"
    s27 = S("27")
    claim("§27/prefix-seconds", sec_ids(27), s27, r"([\d.]+) s against the feed-forward", lambda: dig(J(bs), "perception.enumerate_prefix.seconds"), bs)
    claim("§27/fit-seconds", sec_ids(27), s27, r"baseline's ([\d.]+) s on the same", lambda: dig(J(bs), "perception.enumerate_fit.seconds"), bs)
    claim("§27/prefix-conforming", sec_ids(27), s27, r"identical conforming set of ([\d,]+)\.", lambda: dig(J(bs), "perception.enumerate_prefix.conforming"), bs)
    fact("§27/sets-identical", sec_ids(27), s27, r"identical conforming set",
         lambda: (dig(J(bs), "perception.enumerate_prefix.conforming") == dig(J(bs), "perception.enumerate_fit.conforming"),
                  f"prefix {dig(J(bs), 'perception.enumerate_prefix.conforming')} vs fit {dig(J(bs), 'perception.enumerate_fit.conforming')}"), bs)

    an, mpc = "research/credit-assignment/out/analysis.json", "research/credit-assignment/out/mpc.json"
    s29 = S("29")
    claim("§29/myopic", sec_ids(29), s29, r"myopic optimum of ([\d.]+)", lambda: dig(J(an), "values.5.myopic"), an)
    claim("§29/exploration", sec_ids(29), s29, r"an ([\d,]+)x improvement", lambda: J(an)["exploration"]["6"]["improvement_over_write_text"], an)
    claim("§29/mpc-episodes", sec_ids(29), s29, r"held-out in (\d+) environment episodes", lambda: J(mpc)["environment_episodes"], mpc)
    ag = "research/credit-assignment/out/aggregate.json"
    s31 = S("31")
    claim("§31/composite", sec_ids(31), s31, r"offered \| ([\d.]+) \(sd 0\.061", lambda: J(ag)["macros"][1]["eval"], ag)
    claim("§31/stare-control", sec_ids(31), s31, r"`stare`-only control \| ([\d.]+) \|", lambda: J(ag)["macros"][0]["eval"], ag)
    claim("§31/env-episodes", sec_ids(31), s31, r"and (\d+) environment episodes each", lambda: J(ag)["macros"][1]["env_episodes"], ag)
    rp = "research/visual-ladder/out/rung3_parse.json"
    s32 = S("32")
    claim("§32/rectangles", sec_ids(32), s32, r"\): (\d+) of 215 rectangles", lambda: sum(e["rects_true"] for e in J(rp)["episodes"]), rp)
    claim("§32/links", sec_ids(32), s32, r"and (\d+) of 215 parent links correct", lambda: sum(e["parent_links_correct"] for e in J(rp)["episodes"]), rp)
    claim("§32/root-baseline", sec_ids(32), s32, r"`parent = root` baseline of (0\.\d{3})", lambda: J(rp)["baselines"]["parent_is_root_accuracy"], rp)
    s35 = S("35")
    claim("§35/visual-pyz", sec_ids(35), s35, r"`visual\.pyz` \| ([\d.]+) MB", lambda: J("research/inference-cost/out/pyz_visual.json")[0]["pyz_bytes"] / 2 ** 20,
          "research/inference-cost/out/pyz_visual.json")
    claim("§35/mixed-pyz", sec_ids(35), s35, r"`mixed\.pyz` \| ([\d.]+) MB",
          lambda: next(r for r in J("research/inference-cost/out/sizes.json") if r["name"] == "mixed")["pyz_bytes"] / 2 ** 20,
          "research/inference-cost/out/sizes.json")
    uncheckable("§35/gzip-column", sec_ids(35), "the gzip -9 column (0.573 MB, 206x, …) has no raw file: sizes.json stores only the "
                                                "gzip of the minified JSON, and the four .pyz files are not committed")
    rm8, ab = "research/stranded-block/results/reproduce-main-8seeds.jsonl", "research/stranded-block/results/ab.jsonl"
    s38 = S("38")
    claim("§38/refusals-24", sec_ids(38), s38, r"behind (\d+) \"disconnected", lambda: next(r for r in J(rm8) if r["seed"] == 7)["disconnected_refusals"], rm8)
    claim("§38/refusals-48", sec_ids(38), s38, r"with (\d+) refusals over 220",
          lambda: next(r for r in J(ab) if r["seed"] == 7 and r["rounds"] == 48)["disconnected_refusals"], ab)
    claim("§38/block-before", sec_ids(38), s38, r"objective ([\d.]+) → 2\.4682",
          lambda: next(r for r in J(ab) if r["seed"] == 7 and r["close_block"])["block_events"][0]["before"], ab)
    rl = ROOT / "research/nondegenerate-generalization/run.log"
    lines = lambda: rl.read_text().splitlines()  # noqa: E731
    s40 = S("40")
    fact("§40/lookup-ba-line", sec_ids(40), s40, r"lookup_ba seen 4\.00 unseen 4\.00 interpreter chosen 8/8",
         lambda: (any(re.match(r"lookup_ba\s+seen 4\.00\s+unseen 4\.00\s+interpreter chosen 8/8", ln) for ln in lines()),
                  "run.log carries the lookup_ba summary row"), "research/nondegenerate-generalization/run.log")
    claim("§40/log-rows", sec_ids(40), s40, r"in the (\d+)-row log", lambda: sum("seen" in ln for ln in lines()), "research/nondegenerate-generalization/run.log")
    sc = "research/scaffold-diagnosis/out/scored.json"
    s50 = S("50")
    for key, pat in (("probe", r"\(information \+ sweep\) \| (\d+) \|"), ("sweeponly", r"diagnosis deleted \| (\d+) \|"),
                     ("allholes_exploratory", r"no diagnosis at all \| (\d+) \|")):
        claim(f"§50/{key}", sec_ids(50), s50, pat, lambda key=key: J(sc)["summary"][key]["correct"], sc)
    t2 = "research/motif-unification/out/transfer2.json"
    arm60 = lambda name: next(a for a in J(t2)["arms"] if a["arm"] == name)  # noqa: E731
    s60 = S("60")
    claim("§60/no-library", sec_ids(60), s60, r"no library, matched node count \| (\d+) \|", lambda: arm60("A1_budget3")["conforming"], t2)
    claim("§60/certified-class", sec_ids(60), s60, r"the certified class \| (\d+) \|", lambda: arm60("A2_class")["conforming"], t2)
    claim("§60/reselect-space", sec_ids(60), s60, r"schema's vector \| 1 \| true \| `unique` \| (\d+) \|", lambda: arm60("A2b_schema")["space_size"], t2)
    da, dq = "research/demo-language-fix/out/demo-after-suite/summary.json", "research/demo-language-fix/out/demo-after-quick/summary.json"
    s63 = S("63")
    claim("§63/ten-of-ten", sec_ids(63), s63, r"now reaches (\d+) of 10, exit 0", lambda: J(da)["passed"], da)
    claim("§63/counting-on-shipped-stream", sec_ids(63), s63, r"scores ([\d.]+), exactly the majority",
          lambda: dig(J(dq), "demos.0.detail.stage_b_post_audit.section_19_counting_program_here"), dq)
    cp, dw = "research/language-post-audit/control_preaudit.json", "research/language-post-audit/dyck_p22_c95-110.json"
    s45 = S("45")
    claim("§45/window-conforming", sec_ids(45), s45, r"exhausted, (\d+) conforming, certificate `complete`", lambda: _enum(J(dw))["conforming"], dw)
    claim("§45/control-accuracy", sec_ids(45), s45, r"accuracy (0\.9986187845303868) on n=724",
          lambda: dig(J(cp), "heldout_unseen_lengths.accuracy"), cp)


# --------------------------------------------------------------------------
# 4. single-configuration labels (requirement 4)
# --------------------------------------------------------------------------

LABEL = r"\[single-(?:configuration|family|width|seed-set|task|scaffold|renderer|lesson|subroutine|artifact) evidence:"
CERT_CAUTION = r"not evidence of a correct schema — §53 arm F, §60"

# (audit id, FINDINGS section, anchor regex near the claim, is it one of the nine
#  single-configuration `unique` certificates presented as evidence?)
LABELS = [
    ("A1", "13", r"reaches 3\.38-4\.00 on gate families", False),
    ("A2", "34", r"certificate `unique` on both heads", True),
    ("A3", "32", r"S2: 25/25, 1 conforming, certificate `unique`", True),
    ("A4", "19", r"stage-B search was `unique: false`", False),
    ("A5", "23", r"certifying the conforming address unique", False),
    ("A6", "29", r"`probe_only` scores 0\.000", False),
    ("A9", "52", r"exactly one yields any conforming program", False),
    ("A12", "56", r"Structural 3\.5%, representational 93\.7%", False),
    ("A13", "51", r"2\.079× CI", False),
    ("A17", "21", r"transfers to both unchanged at held-out 1\.000000", False),
    ("A18", "26", r"4/4 seeds recover the exact reference kernel", False),
    ("A19", "14a", r"exhaust 48 programs in 4\.6 s returning exactly one", True),
    ("A20", "18", r"then chosen by a second program over a same-shaped distractor", True),
    ("A21", "21", r"8 programs, exhausted, unique", True),
    ("A22", "16", r"unique global optimum", False),
    ("A23", "27", r"reward proves 1 of those 2", True),
    ("A24", "12", r"all of which use the module", False),
    ("A25", "60", r"The certified class scores zero where no-library succeeds", False),
    ("cert:§10", "10", r"certifies the solution unique, and selects the identical program", True),
    ("cert:§11", "11", r"certifying the solution unique among 65,536 programs", True),
    ("cert:§41", "41", r"visual S2 1/25 `unique`", True),
]


def check_labels() -> None:
    for aid, label, anchor, is_cert in LABELS:
        text = S(label)
        m = re.search(anchor, text)
        if not m:
            record(FAIL, f"labels/{aid}", f"anchor not found in §{label}: /{anchor}/", sec_ids(label))
            continue
        window = text[max(0, m.start() - 200):m.end() + 600]
        lab = re.search(LABEL + r"[^\]]*\]", window)
        if not lab:
            record(FAIL, f"labels/{aid}", f"§{label}: the claim '{m.group(0)[:60]}' carries no single-configuration label "
                                         "in its sentence", sec_ids(label))
            continue
        if is_cert and not re.search(CERT_CAUTION, lab.group(0)):
            record(FAIL, f"labels/{aid}", f"§{label}: a single-configuration `unique` certificate must say it is "
                                         f"'{CERT_CAUTION}'", sec_ids(label))
            continue
        record(PASS, f"labels/{aid}", f"§{label}: labelled", sec_ids(label))


# --------------------------------------------------------------------------
# 5. sections no committed artifact can check
# --------------------------------------------------------------------------

SECTION_UNCHECKABLE = {
    "42": "tombstone — the number was never assigned (see the heading)",
    "6": "open questions; the live-node fraction 1.00 -> 0.38 is stored in no search-scaling file, track 2's SNR "
         "0.221 has no track directory, and 88.4% is a literature figure",
    "9": "an ordered to-do list; its only numbers are recommendation and section indices",
    "28": "its only figure is a test count ('264 tests pass'), and no artifact records test counts",
}


def check_coverage(verbose: bool) -> tuple:
    labels = [lab for lab, t, _ in findings_sections() if not is_tombstone(t)]
    verified = {s.lstrip("§") for r in results if r.status == PASS and r.artifact for s in r.sections}
    declared = {lab: why for lab, why in SECTION_UNCHECKABLE.items()}
    for lab in labels:
        if lab in verified:
            continue
        if lab in declared:
            uncheckable(f"coverage/§{lab}", sec_ids(lab), declared[lab])
        else:
            record(FAIL, f"coverage/§{lab}", "no PASSing artifact check and no SECTION_UNCHECKABLE entry — add a check "
                                             "against a committed artifact, or declare why none exists", sec_ids(lab))
    n_ver = sum(1 for lab in labels if lab in verified)
    if verbose:
        print("\nsection coverage")
        for lab in labels:
            state = "VERIFIED" if lab in verified else ("UNCHECKABLE: " + declared[lab] if lab in declared else "NONE")
            print(f"  §{lab:<4} {state}")
    return n_ver, len(labels)


# --------------------------------------------------------------------------
# 6. the shipped demonstrations (full audit only: needs artifacts/demo/summary.json)
# --------------------------------------------------------------------------

STATUS_CLAIMS = {
    "synthesis":    {"held_out_max_error": ("<=", 1e-7), "held_out_points": ("==", 584)},
    "joint":        {"frozen_program_return": ("==", 4.0)},
    "structure":    {"unseen_return.mean": (">=", 2.75)},   # STATUS row 3: arm means 3.38-4.00, per-seed floor 2.75
    "depth":        {"exact_frozen_return.8": ("==", 4.0)},
    "abstraction":  {"arms.B.module_on_output_path": ("==", True)},
    "positional":   {"wide.positions": ("==", 1024), "wide.observation_values": ("==", 3072),
                     "wide.caller_nodes": ("==", 3)},
    "segmentation": {"held_out_max_error": ("==", 0.0), "space_size": ("==", 65536),
                     "majority_class_accuracy": ("~", 0.854)},
    "edge":         {"held_out_accuracy": ("==", 1.0), "staged_space": ("==", 48),
                     "majority_class_accuracy": ("~", 0.844)},
    "control":      {"max_state_difference_after_8_steps": ("==", 0.0)},
    "language":     {},
}


def check_demo(summary_path: Path | None, run: bool) -> None:
    if run:
        out = ROOT / "artifacts" / "demo"
        subprocess.run([sys.executable, "-m", "tcn", "demo", "--out", str(out)], cwd=ROOT, check=False)
        summary_path = out / "summary.json"
    if summary_path is None:
        summary_path = ROOT / "artifacts" / "demo" / "summary.json"
    if not summary_path.exists():
        uncheckable("demo/summary", (), f"no demo summary at {summary_path} (artifacts/ is gitignored); run --demo")
        return
    d = J(str(summary_path.relative_to(ROOT))) if summary_path.is_relative_to(ROOT) else _parse(summary_path.read_text())
    names = [row["demo"] for row in d["demos"]]
    if sorted(names) != sorted(STATUS_CLAIMS):
        # The audit's baseline found this check passing on a one-demo summary (1/1).
        uncheckable("demo/summary", (), f"{summary_path} covers {len(names)} of {len(STATUS_CLAIMS)} demos ({names}); "
                                        "STATUS.md's 'Ten capabilities' cannot be checked from it")
        return
    by_name = {row["demo"]: row for row in d["demos"]}
    record(PASS if d["passed"] == d["total"] == 10 else FAIL, "demo/all-pass",
           f"{d['passed']}/{d['total']} demonstrations reproduced in {d['seconds']} s")
    for name, claims in STATUS_CLAIMS.items():
        row = by_name[name]
        if not row.get("ok"):
            record(FAIL, f"demo/{name}", f"FAIL — {row['measured']}")
            continue
        for key, (op, want) in claims.items():
            try:
                got = dig(row["detail"], key)
            except (KeyError, IndexError, TypeError):
                record(WARN, f"demo/{name}/{key}", "key absent from demo detail")
                continue
            ok = {"==": lambda: got == want, "<=": lambda: got <= want, ">=": lambda: got >= want,
                  "~": lambda: close(got, want, 1e-2)}[op]()
            record(PASS if ok else FAIL, f"demo/{name}/{key}", f"measured {got}, STATUS claims {op} {want}")


def close(a, b, rel=1e-3) -> bool:
    a, b = float(a), float(b)
    return abs(a) <= rel if b == 0 else abs(a - b) <= rel * abs(b)


def check_tests() -> None:
    proc = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=ROOT, capture_output=True, text=True)
    tail = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
    m = re.search(r"(?:(\d+) failed, )?(\d+) passed", tail)
    failed = int(m.group(1)) if m and m.group(1) else 0
    passed = int(m.group(2)) if m else 0
    known = "test_panel_episode_replays_and_restores" in proc.stdout
    record(PASS if failed == 0 or (failed == 1 and known) else FAIL, "tests/suite",
           f"{passed} passed, {failed} failed" + (" (the documented worktree-only panel replay failure)" if failed == 1 and known else ""))


# --------------------------------------------------------------------------
# 7. HEADLINES.md — the machine-generated headline table
# --------------------------------------------------------------------------

def headlines_text() -> str:
    lines = ["# Headline figures, generated from committed artifacts",
             "",
             "**Generated by `research/record-audit/verify.py --emit-headlines`. Do not edit by hand.**",
             "Every value below is read from the named artifact, not transcribed. A document quoting one of",
             "these figures is checked against this value by the gate (`verify.py --gate`). Quote from here.",
             "",
             "| section | check | value from artifact | artifact |",
             "|---|---|---|---|"]
    for secs, ident, val, src in sorted(headline_rows, key=lambda r: (_sortkey(r[0]), r[1])):
        lines.append(f"| {', '.join(secs)} | `{ident}` | {val} | `{src}` |")
    return "\n".join(lines) + "\n"


def _sortkey(secs):
    if not secs:
        return (999, "")
    m = re.match(r"§(\d+)([a-z]?)", secs[0])
    return (int(m.group(1)), m.group(2)) if m else (999, secs[0])


def check_headlines_current() -> None:
    want = headlines_text()
    have = HEADLINES.read_text() if HEADLINES.exists() else ""
    record(PASS if have == want else FAIL, "record/headlines-current",
           "HEADLINES.md matches the artifacts" if have == want
           else "HEADLINES.md is stale — run verify.py --emit-headlines and commit it")


# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gate", action="store_true", help="fast gate only: no pytest collection, no demo")
    ap.add_argument("--demo", action="store_true", help="run the ten demonstrations first (slow)")
    ap.add_argument("--demo-from", type=Path, default=None, help="read an existing demo summary.json")
    ap.add_argument("--tests", action="store_true", help="also run the pytest suite (slow)")
    ap.add_argument("--coverage", action="store_true", help="print the per-section coverage table")
    ap.add_argument("--emit-headlines", action="store_true", help="rewrite HEADLINES.md from the artifacts")
    ap.add_argument("--quiet", action="store_true", help="print only non-PASS rows and the summary")
    args = ap.parse_args()

    def guarded(fn, *a):
        """A check group that raises records a FAIL rather than hiding the groups after it."""
        try:
            fn(*a)
        except Exception as exc:  # noqa: BLE001
            record(FAIL, f"crash/{fn.__name__}", f"{type(exc).__name__}: {exc} — the checks after this point in "
                                                 "the group did not run")

    guarded(check_section_numbering)
    guarded(check_cross_references)
    guarded(check_cited_paths)
    guarded(check_retracted_claims)
    guarded(check_status_freshness, not args.gate)
    guarded(checks_language)
    guarded(check_joint_constant_baseline)
    guarded(checks_perception)
    guarded(checks_environment_and_language_audit)
    guarded(checks_cost_line)
    guarded(checks_library_line)
    guarded(checks_early_sections)
    guarded(check_labels)
    if args.emit_headlines:
        HEADLINES.write_text(headlines_text())
    check_headlines_current()
    if not args.gate:
        check_demo(args.demo_from, args.demo)
    if args.tests:
        check_tests()
    n_ver, n_sec = check_coverage(args.coverage)

    width = max(len(r.ident) for r in results)
    for r in results:
        if args.quiet and r.status == PASS:
            continue
        print(f"{r.status:11}  {r.ident:<{width}}  {r.detail}")
    count = {s: sum(1 for r in results if r.status == s) for s in (PASS, FAIL, WARN, UNCHECKABLE, OPEN)}
    print(f"\n{len(results)} checks: {count[PASS]} pass, {count[FAIL]} fail, {count[WARN]} warn, "
          f"{count[UNCHECKABLE]} uncheckable, {count[OPEN]} open")
    print(f"section coverage: {n_ver} of {n_sec} FINDINGS sections have at least one verified claim")
    return 1 if count[FAIL] else 0


if __name__ == "__main__":
    raise SystemExit(main())
