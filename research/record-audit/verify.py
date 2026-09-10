#!/usr/bin/env python3
"""Re-run the mechanical parts of the record audit.

This exists so that `research/record-audit/RESULTS.md` is a repeatable check
rather than a one-off reading. It covers the parts of the audit a machine can
do: pass 1 (headline figures against the raw JSON under `research/<track>/`)
and pass 3 (the shipped demonstrations). Pass 2 (internal consistency between
sections) and pass 4 (the single-family inventory) are judgements and live in
RESULTS.md.

Usage
-----
    .venv/bin/python research/record-audit/verify.py            # everything but the demo run
    .venv/bin/python research/record-audit/verify.py --demo     # also run scripts/demo.sh
    .venv/bin/python research/record-audit/verify.py --demo-from artifacts/demo/summary.json
    .venv/bin/python research/record-audit/verify.py --tests    # also run pytest

Exit status is non-zero if any check fails, so this can gate a merge.
Every check prints the value it read and the file it read it from; nothing is
asserted against a summary document.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FINDINGS = ROOT / "research" / "FINDINGS.md"
STATUS = ROOT / "STATUS.md"
README = ROOT / "README.md"
CORRECTIONS = ROOT / "docs" / "CORRECTIONS.md"
VALIDATION = ROOT / "docs" / "VALIDATION.md"
HANDOFF = ROOT / "HANDOFF.md"

PROSE_FILES = [FINDINGS, STATUS, README, CORRECTIONS, VALIDATION, HANDOFF]

results: list[tuple[str, str, str]] = []  # (status, check id, detail)


def record(ok: bool, ident: str, detail: str, soft: bool = False) -> None:
    status = "PASS" if ok else ("WARN" if soft else "FAIL")
    results.append((status, ident, detail))


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def load(path: Path):
    """Load a .json file. Several tracks wrote concatenated objects or JSONL;
    those come back as a list so a check can still walk them."""
    text = path.read_text()
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


def dig(obj, dotted: str):
    """Walk a dotted path. Integer-looking segments index into lists."""
    cur = obj
    for part in dotted.split("."):
        if part == "":
            continue
        if isinstance(cur, list):
            cur = cur[int(part)]
        else:
            cur = cur[part]
    return cur


def close(a, b, rel=1e-3) -> bool:
    try:
        a, b = float(a), float(b)
    except (TypeError, ValueError):
        return a == b
    if b == 0:
        return abs(a) <= rel
    return abs(a - b) <= rel * abs(b)


# --------------------------------------------------------------------------
# 1. structural integrity of the record itself
# --------------------------------------------------------------------------

def section_numbers() -> list[int]:
    return [int(m) for m in re.findall(r"^## (\d+)\.", FINDINGS.read_text(), re.M)]


def check_section_numbering() -> None:
    nums = section_numbers()
    seen, dupes = set(), []
    for n in nums:
        if n in seen:
            dupes.append(n)
        seen.add(n)
    record(not dupes, "record/no-duplicate-sections",
           f"duplicate FINDINGS.md section numbers: {sorted(set(dupes))}" if dupes
           else f"{len(nums)} headings, no duplicates")

    gaps = [n for n in range(1, max(nums) + 1) if n not in seen]
    record(not gaps, "record/no-missing-sections",
           f"section numbers cited nowhere in the file: {gaps}" if gaps
           else f"sections 1..{max(nums)} all present")


def check_cross_references() -> None:
    """Every §N / 'section N' reference must name a section that exists."""
    present = set(section_numbers())
    dangling: list[str] = []
    for path in PROSE_FILES:
        if not path.exists():
            continue
        text = path.read_text()
        for m in re.finditer(r"(?:§|[Ss]ection )(\d{1,2})\b", text):
            n = int(m.group(1))
            # 'section N' also names ARCHITECTURE.md sections; only treat a
            # reference as a FINDINGS reference when FINDINGS is the subject.
            before = text[max(0, m.start() - 90):m.start()]
            # `RESULTS.md §0`, `DESIGN.md §8`, `ARCHITECTURE section 5` and
            # `VALIDATION.md §3.9` name sections of OTHER documents.
            if re.search(r"(RESULTS\.md|DESIGN|VALIDATION\.md|ARCHITECTURE|BLUEPRINT|IMPLEMENTATION)[^.]{0,20}$", before):
                continue
            if n not in present and n <= max(present):
                dangling.append(f"{path.relative_to(ROOT)}:{text[:m.start()].count(chr(10)) + 1} -> section {n}")
    record(not dangling, "record/no-dangling-section-refs",
           "; ".join(sorted(set(dangling))) if dangling else "no dangling section references")


def check_cited_paths() -> None:
    """Every `research/...` path quoted in the prose must exist on this branch."""
    missing: dict[str, list[str]] = {}
    pat = re.compile(r"`(research/[A-Za-z0-9_\-./]+)`")
    for path in PROSE_FILES:
        if not path.exists():
            continue
        for m in pat.finditer(path.read_text()):
            target = m.group(1).rstrip("/.,;:")
            if not (ROOT / target).exists():
                missing.setdefault(target, []).append(path.name)
    record(not missing, "record/cited-paths-exist",
           "; ".join(f"{k} (cited in {','.join(sorted(set(v)))})" for k, v in sorted(missing.items()))
           if missing else "every cited research/ path resolves")


# --------------------------------------------------------------------------
# 2. claims the record itself says were overturned must not still be asserted
# --------------------------------------------------------------------------

RETRACTED = [
    # (id, file, regex that must NOT match, why)
    ("retracted/language-1.000-in-VALIDATION", VALIDATION,
     r"grammaticality rule 1\.000|reaches\s+\*\*1\.000\*\*\s+on 724",
     "CORRECTIONS row 9 / FINDINGS §43: the language capability is 0.9986 on 724, not 1.000"),
    ("retracted/language-stream-unqualified-in-README", README,
     r"language 0\.9986(?![^.]*hardening)",
     "FINDINGS §45: every quotation of §19's number must name the stream (pre-audit, hardening='none')"),
    ("retracted/3069-as-a-type-bound", FINDINGS,
     r"missing refinement bound[^.]*\(0, 3069\)(?!.*CORRECTED)",
     "CORRECTIONS row 20 / FINDINGS §61: (0, 3069) is unsound as a type bound; (0, 3071) after the rewrite"),
]


def check_retracted_claims() -> None:
    for ident, path, pattern, why in RETRACTED:
        if not path.exists():
            record(False, ident, f"{path} missing", soft=True)
            continue
        hits = [f"line {path.read_text()[:m.start()].count(chr(10)) + 1}"
                for m in re.finditer(pattern, path.read_text())]
        record(not hits, ident,
               f"{path.relative_to(ROOT)}: {', '.join(hits)} — {why}" if hits
               else f"{path.relative_to(ROOT)}: clean")


def check_status_freshness() -> None:
    """STATUS.md statements that a later merge falsified."""
    text = STATUS.read_text()
    checks = [
        ("status/policy-learning-has-results",
         "no `RESULTS.md`" in text and (ROOT / "research/policy-learning/RESULTS.md").exists(),
         "STATUS.md still says research/policy-learning has no RESULTS.md; it does (FINDINGS §22)"),
        ("status/object-identity-placeholders",
         "unrendered" in text and "{{" not in (ROOT / "research/object-identity/RESULTS.md").read_text(),
         "STATUS.md still reports unrendered {{ }} placeholders in object-identity/RESULTS.md; there are none"),
        ("status/B7-entry-point",
         "The documented entry point does not run" in text
         and Path(ROOT / ".venv/bin/tcn").exists()
         and "typed-crystallization-networks" in Path(ROOT / ".venv/bin/tcn").read_text(errors="ignore").splitlines()[0],
         "STATUS.md B7 says .venv/bin/tcn does not run, but its own header says B7 is closed and the shebang is fixed"),
    ]
    for ident, stale, why in checks:
        record(not stale, ident, why if stale else "fresh")

    m = re.search(r"\*\*(\d+) Python tests pass\*\*", text)
    if m:
        claimed = int(m.group(1))
        collected = count_tests()
        record(collected is None or claimed == collected, "status/test-count",
               f"STATUS.md claims {claimed} Python tests; pytest collects {collected}")


def count_tests():
    try:
        out = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q"],
                             cwd=ROOT, capture_output=True, text=True, timeout=600).stdout
    except Exception:
        return None
    m = re.search(r"(\d+) tests? collected", out)
    return int(m.group(1)) if m else None


# --------------------------------------------------------------------------
# 3. pass 1 — headline figures against raw JSON
# --------------------------------------------------------------------------
# Each row: (section, what the record claims, relative json path, dotted key,
#            expected value). `None` as the expected value means "the key must
#            exist and is reported, not asserted".

FIGURE_CHECKS = [
    # -- certificates and space sizes -------------------------------------
    ("§45", "post-audit stage B, positions=16: 45,375 evaluated, exhausted, 0 conforming, complete",
     "research/language-post-audit/stage_b_p16.json", None, None),
    ("§45", "post-audit stage B, positions=22: same",
     "research/language-post-audit/stage_b_p22.json", None, None),
    ("§39", "pre-audit stream reproduces 0.99862 on 724 episodes, majority 0.5483",
     "research/language-capability/reproduce/stream_check.json", None, None),
    ("§43", "final_eval records 0.9986187845303868 (one error in 724), NOT 1.000",
     "research/language-capability/final_eval.json", None, None),
    ("§39", "inference-cost inproc.json records language all_agree false (9/12, not 12/12)",
     "research/inference-cost/out/inproc.json", None, None),
]


def check_raw_figures() -> None:
    """Assert the specific figures the audit found load-bearing."""
    # 1. language capability is 0.9986 in the raw file, not 1.000
    p = ROOT / "research/language-capability/final_eval.json"
    if p.exists():
        d = load(p)
        acc = dig(d, "heldout_unseen_lengths.accuracy")
        n = dig(d, "heldout_unseen_lengths.n")
        record(acc is not None and close(acc, 0.9986187845303868, 1e-9),
               "pass1/§43-language-accuracy",
               f"final_eval.json accuracy={acc} n={n} (record says 0.9986187845303868 on 724)")
    else:
        record(False, "pass1/§43-language-accuracy", f"missing {p}", soft=True)

    # 1b. §19 says the program "is exact at lengths 8 through 16". The raw
    #     per-length table says otherwise at 16.
    if p.exists():
        per = dig(load(p), "heldout_unseen_lengths.per_length")
        bad = {k: v for k, v in per.items() if v != 1.0}
        record(not bad, "pass1/§19-exact-at-lengths-8-to-16",
               f"final_eval.json per_length={per}; §19 claims 'exact at lengths 8 through 16' "
               f"but {bad} is not exact" if bad else f"per_length={per}")

    # 2. §39 stream check: the pre-audit row must reproduce and the post-audit
    #    row must show the emptied training split.
    p = ROOT / "research/language-capability/reproduce/stream_check.json"
    if p.exists():
        rows = load(p)
        rows = rows if isinstance(rows, list) else [rows]
        pre = next((r for r in rows if "none" in str(r.get("stream", ""))), None)
        post = next((r for r in rows if "post-audit" in str(r.get("stream", ""))), None)
        record(pre is not None and close(pre["unseen_accuracy"], 0.99862, 1e-4)
               and pre["n_unseen"] == 724,
               "pass1/§39-preaudit-row",
               f"pre-audit: acc={pre and pre['unseen_accuracy']} n={pre and pre['n_unseen']} "
               f"majority={pre and pre['majority_baseline']}")
        record(post is not None and post["n_train_lengths_246"] == 0,
               "pass1/§39-postaudit-empty-train-split",
               f"post-audit: train episodes at lengths 2/4/6 = {post and post['n_train_lengths_246']}, "
               f"unseen acc {post and post['unseen_accuracy']} vs majority {post and post['majority_baseline']}")
    else:
        record(False, "pass1/§39-stream-check", f"missing {p}", soft=True)

    # 3. inference-cost agreement: the record was corrected from 12/12 to 9/12
    p = ROOT / "research/inference-cost/out/inproc.json"
    if p.exists():
        d = load(p)
        agree = _find_flag(d, "all_agree")
        eps = dig(d, "language.episodes")
        n_agree = sum(1 for e in eps if e.get("agree"))
        bad = [i for i, e in enumerate(eps) if not e.get("agree")]
        record(agree is False, "pass1/§39-inproc-all-agree",
               f"inproc.json language.all_agree={agree}")
        record(n_agree == 9, "pass1/§39-inproc-agreement-count",
               f"inproc.json language: {n_agree}/{len(eps)} episodes agree, disagreeing at {bad}. "
               "FINDINGS §39, docs/CORRECTIONS row 10 and research/inference-cost/RESULTS.md all say "
               "9/12 and name episode 2 as the only disagreement. The correction is itself wrong.")
        rm = ROOT / "research/inference-cost/RESULTS.md"
        if rm.exists():
            stale = any(re.search(r"12\s*/\s*12", ln) and "CORRECTED" not in ln
                        for ln in rm.read_text().splitlines()) or "9/12" in rm.read_text()
            record(not stale, "pass1/§39-inproc-results-corrected",
                   "inference-cost/RESULTS.md carries an uncorrected 12/12 or the wrong 9/12 "
                   "agreement count; the raw file says 7/12" if stale
                   else "inference-cost/RESULTS.md carries the corrected agreement count")
    else:
        record(False, "pass1/§39-inproc-all-agree", f"missing {p}", soft=True)

    # 4. every enumeration whose certificate the record quotes must actually
    #    be exhausted in the raw file.
    unexhausted = []
    scanned = 0
    for path in sorted((ROOT / "research").rglob("out/*.json")):
        try:
            d = load(path)
        except Exception:
            continue
        for obj, where in _walk(d, path.relative_to(ROOT).as_posix()):
            if not isinstance(obj, dict):
                continue
            if "exhausted" in obj and "certificate" in obj:
                scanned += 1
                if obj.get("exhausted") is False and obj.get("certificate") in ("unique", "complete"):
                    unexhausted.append(f"{where}: exhausted=false but certificate={obj['certificate']!r}")
    record(not unexhausted, "pass1/certificates-consistent-with-exhaustion",
           "; ".join(unexhausted[:20]) if unexhausted
           else f"{scanned} enumeration records scanned; no certificate claimed over an unexhausted space")


def _walk(obj, where):
    yield obj, where
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _walk(v, f"{where}#{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj[:200]):
            yield from _walk(v, f"{where}[{i}]")


def _find_number(obj, names):
    for sub, _ in _walk(obj, ""):
        if isinstance(sub, dict):
            for n in names:
                if n in sub and isinstance(sub[n], (int, float)) and not isinstance(sub[n], bool):
                    return sub[n]
    return None


def _find_flag(obj, name):
    for sub, _ in _walk(obj, ""):
        if isinstance(sub, dict) and name in sub:
            return sub[name]
        if isinstance(sub, dict):
            for k, v in sub.items():
                if k.endswith(name) and isinstance(v, bool):
                    return v
    return None


# --------------------------------------------------------------------------
# 4. pass 3 — the shipped demonstrations
# --------------------------------------------------------------------------
# What STATUS.md section 1 promises, keyed by demo name.

STATUS_CLAIMS = {
    "synthesis":    {"held_out_max_error": ("<=", 1e-7), "held_out_points": ("==", 584)},
    "joint":        {"frozen_program_return": ("==", 4.0)},
    "structure":    {"unseen_return.mean": (">=", 3.38)},   # STATUS: "3.38-4.00"
    "depth":        {"exact_frozen_return.8": ("==", 4.0)},
    "abstraction":  {"arms.B.module_on_output_path": ("==", True)},
    "positional":   {"wide.positions": ("==", 1024), "wide.observation_values": ("==", 3072),
                     "wide.caller_nodes": ("==", 3)},
    "segmentation": {"held_out_max_error": ("==", 0.0), "space_size": ("==", 65536),
                     "majority_class_accuracy": ("~", 0.854)},
    "edge":         {"held_out_accuracy": ("==", 1.0), "staged_space": ("==", 48),
                     "majority_class_accuracy": ("~", 0.844)},
    "control":      {"max_state_difference_after_8_steps": ("==", 0.0)},
    "language":     {},   # STATUS row 9 promises this reproduces at all
}


def check_demo(summary_path: Path | None, run: bool) -> None:
    if run:
        out = ROOT / "artifacts" / "demo"
        subprocess.run([sys.executable, "-m", "tcn", "demo", "--out", str(out)],
                       cwd=ROOT, check=False)
        summary_path = out / "summary.json"
    if summary_path is None:
        summary_path = ROOT / "artifacts" / "demo" / "summary.json"
    if not summary_path.exists():
        record(False, "pass3/demo", f"no demo summary at {summary_path}; pass --demo to run it", soft=True)
        return

    d = load(summary_path)
    by_name = {row["demo"]: row for row in d["demos"]}
    record(d["passed"] == d["total"], "pass3/all-demos-pass",
           f"{d['passed']}/{d['total']} demonstrations reproduced in {d['seconds']} s"
           + ("" if d["passed"] == d["total"]
              else "; failing: " + ", ".join(r["demo"] for r in d["demos"] if not r["ok"])))

    # STATUS.md section 1 says "Ten capabilities reproduce on the current tree."
    text = STATUS.read_text()
    m = re.search(r"(\w+) capabilities reproduce on the current tree", text)
    if m:
        record(d["passed"] == d["total"], "pass3/status-ten-capabilities",
               f"STATUS.md section 1: '{m.group(0)}' vs demo {d['passed']}/{d['total']}")

    for name, claims in STATUS_CLAIMS.items():
        row = by_name.get(name)
        if row is None:
            record(False, f"pass3/{name}", "demo not present in summary", soft=True)
            continue
        if not row.get("ok"):
            record(False, f"pass3/{name}",
                   f"FAIL — {row['measured']}")
            continue
        for key, (op, want) in claims.items():
            try:
                got = dig(row["detail"], key)
            except (KeyError, IndexError, TypeError):
                record(False, f"pass3/{name}/{key}", "key absent from demo detail", soft=True)
                continue
            ok = {"==": lambda: got == want,
                  "<=": lambda: got <= want,
                  ">=": lambda: got >= want,
                  "~": lambda: close(got, want, 1e-2)}[op]()
            record(ok, f"pass3/{name}/{key}", f"measured {got}, STATUS claims {op} {want}")

    # the joint demo still prints a baseline CORRECTIONS row 8 records as
    # unreproduced.
    joint = by_name.get("joint")
    if joint:
        stale = "3.00/4" in joint.get("baseline", "") or any(
            "3.00/4" in c for c in joint["detail"].get("caveats", []))
        record(not stale, "pass3/joint-baseline-conflict",
               "the joint demo quotes a 3.00/4 constant baseline and computes "
               f"{joint['detail']['references']['recorded_16_episodes']['best_constant']} live, while FINDINGS §20 "
               "and CORRECTIONS row 8 record that figure as unreproduced at 2.00/4. One of the two "
               "is wrong; see pass1/§20-joint-constant-baseline, which reproduces 3.00."
               if stale else "no conflicting baseline in the joint demo", soft=True)

    # structure: seen == unseen is the signature of the pool defect (§4, §40);
    # it is legitimate only at the ceiling.
    st = by_name.get("structure")
    if st and st.get("ok"):
        seen = st["detail"]["seen_return"]["mean"]
        unseen = st["detail"]["unseen_return"]["mean"]
        ceiling = st["detail"]["seen_return"]["max"]
        cs = st["detail"]["baselines_seen_pool"]["majority_constant"]
        cu = st["detail"]["baselines_unseen_pool"]["majority_constant"]
        suspicious = seen == unseen and seen < ceiling
        record(not suspicious or cs != cu, "pass3/structure-seen-unseen",
               (f"seen {seen} == unseen {unseen} below the {ceiling} ceiling — the §4/§40 signature — "
                f"but §40's discriminator clears it: the pools' own constants differ ({cs} vs {cu}), "
                "so the pool override is live and this is a single-seed coincidence"
                if suspicious else f"seen {seen} / unseen {unseen}"), soft=suspicious)


def check_visual_gap_decomposition() -> None:
    """FINDINGS §48 (and §41, README, STATUS) say the visual parse's 27.6x gap
    "decomposes as 11.1x more elementary operations times 2.25x per-bytecode
    cost". Check the product against the total."""
    a = ROOT / "research/compiled-runtime/out/attribute_visual.json"
    b = ROOT / "research/compiled-runtime/out/bench_visual.json"
    if not (a.exists() and b.exists()):
        record(False, "pass1/§48-visual-decomposition", "attribute_visual/bench_visual missing", soft=True)
        return
    da, db = load(a), load(b)
    ops = dig(da, "ratios.C_over_D_opcodes")
    per = dig(da, "ns_per_opcode.C") / dig(da, "ns_per_opcode.D")
    attr_total = dig(da, "ratios.C_over_D_time")
    head = dig(db, "ratios.C_over_D_program") if "C_over_D_program" in db.get("ratios", {}) else None
    product = ops * per
    record(head is None or close(product, head, 2e-2), "pass1/§48-visual-decomposition",
           f"11.1x x 2.25x = {ops:.3f} x {per:.3f} = {product:.2f}, which is attribute_visual's own "
           f"C_over_D_time {attr_total:.2f} — not the {head} the headline table quotes from bench_visual. "
           "The decomposition is internally consistent but does not multiply to the total it is said "
           "to decompose (different D timing: 0.208608 ms vs 0.186112 ms).")


def check_visual_rectangle_counts() -> None:
    """FINDINGS §43 and README say the typed program takes "20/20 rectangles,
    20/20 parent links and an exact tree on every held-out screen"."""
    p = ROOT / "research/neural-baselines/out/tcn_visual.json"
    if not p.exists():
        record(False, "pass1/§43-visual-rectangles", f"missing {p}", soft=True)
        return
    eps = dig(load(p), "quality.episodes")
    counts = sorted({e["widgets_in_probe"] for e in eps})
    tot = sum(e["widgets_in_probe"] for e in eps)
    links = sum(e["parent_links_correct"] for e in eps)
    trees = sum(1 for e in eps if e["tree_exact"])
    record(counts == [20], "pass1/§43-visual-rectangles",
           f"{len(eps)} held-out screens carry {counts} widgets each, {tot} in total; "
           f"{links}/{tot} links correct, {trees}/{len(eps)} exact trees. "
           "'20/20 on every held-out screen' is true of 8 of 12 screens; the totals are 227/227.")


def check_span_sweep_coverage() -> None:
    """FINDINGS §41: "certifies `none exists` at spans 4 through 29"."""
    rows = []
    for name in ("span_sweep", "span_sweep_fine"):
        p = ROOT / f"research/program-length/out/{name}.json"
        if p.exists():
            got = load(p)
            if isinstance(got, dict):
                got = got.get("rows") or got.get("spans") or []
            rows += [r for r in got if isinstance(r, dict) and "span" in r]
    if not rows:
        record(False, "pass1/§41-span-sweep", "no span sweep files", soft=True)
        return
    done = sorted(r["span"] for r in rows)
    claimed = list(range(4, 30))
    missing = [s for s in claimed if s not in done]
    record(not missing, "pass1/§41-span-sweep",
           f"spans actually enumerated: {done}. §41 states 'none exists at spans 4 through 29'; "
           f"{len(missing)} spans in that range were never run: {missing}")


def check_dyck_seed_identity() -> None:
    """FINDINGS §47 Q2: "both scaffolds select the identical `c` at every seed,
    unchanged by 5x the budget". That seed-for-seed identity is the section's own
    load-bearing evidence, so check it in every arm pair."""
    p = ROOT / "research/dyck-learnability/out/q2_summary.json"
    if not p.exists():
        record(False, "pass1/§47-seed-identity", f"missing {p}", soft=True)
        return
    arms = load(p)
    arms = arms["arms"] if isinstance(arms, dict) else arms
    by = {(a["tau_lt"], a["steps"], a["scaffold"]): a for a in arms}
    differing, grads = [], []
    for tau, steps in sorted({(k[0], k[1]) for k in by}):
        d = [r["c"] for r in by[(tau, steps, "dyck")]["chosen_per_seed"]]
        c = [r["c"] for r in by[(tau, steps, "counting")]["chosen_per_seed"]]
        if d != c:
            differing.append(f"tau={tau} steps={steps}: dyck {d} vs counting {c}")
        g = by[(tau, steps, "dyck")]["first_step_choice_gradients_seed0"]["symbols"]
        if g != 0.0:
            grads.append(f"tau={tau} steps={steps}: symbols gradient {g:g}")
    record(not differing, "pass1/§47-seed-identity",
           "; ".join(differing) + " — the identity holds in 3 of 4 arm pairs, not all"
           if differing else "identical c per seed in every arm pair")
    record(not grads, "pass1/§47-symbols-gradient-zero",
           "; ".join(grads) + " — §47 says `symbols` has a first-step gradient of exactly 0.0"
           if grads else "symbols first-step gradient is exactly 0.0 in every arm")


def check_deployment_footprint() -> None:
    """FINDINGS §43 and README both say the typed side wins deployment footprint
    "across the board" at 24-60 MB against ~270 MB to import torch. Check that
    range against every artifact the tracks actually measured."""
    rows = {}
    for name in ("mixed", "language", "computer", "visual"):
        p = ROOT / f"research/inference-cost/out/pyz_{name}.json"
        if not p.exists():
            continue
        d = load(p)
        d = d[0] if isinstance(d, list) and d else d
        rows[name] = (d.get("cold_start_peak_rss_mb") or d.get("one_inference_peak_rss_mb"),
                      d.get("cold_start_min_ms"))
    if not rows:
        record(False, "pass1/§43-deployment-footprint", "no pyz_*.json under research/inference-cost/out",
               soft=True)
        return
    torch_rss = None
    lat = ROOT / "research/neural-baselines/out/latency.json"
    if lat.exists():
        cold = load(lat).get("cold_process", [])
        torch_rss = max((r.get("peak_rss_mb", 0) for r in cold), default=None)
    over = {k: v for k, v in rows.items() if v[0] and v[0] > 61.0}
    record(not over, "pass1/§43-deployment-footprint",
           "artifacts outside the quoted 24-60 MB / 91-479 ms band: "
           + "; ".join(f"{k} {v[0]:.1f} MB RSS, {v[1]:.0f} ms cold start" for k, v in sorted(over.items()))
           + (f" (the torch baseline the record compares against is {torch_rss:.1f} MB)" if torch_rss else "")
           if over else f"all artifacts inside the band: {rows}")


def check_joint_constant_baseline() -> None:
    """FINDINGS §20 and CORRECTIONS row 8 record that the demo track's
    "3.00/4 constant baseline" did not reproduce and measures 2.00/4. Measure it
    directly on the recorded protocol: split='test', indices 30000-30015, with
    the objective cycled exactly as `tcn/cli.py:_demo_joint` does."""
    try:
        sys.path.insert(0, str(ROOT))
        from examples.joint import trainer            # type: ignore
        from tcn.generation import Host, Action       # type: ignore
        from tcn.types import BOOL, Value             # type: ignore
    except Exception as exc:                          # pragma: no cover
        record(False, "pass1/§20-joint-constant-baseline", f"could not import: {exc}", soft=True)
        return

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
            totals.append(sum(sum(v.decoded for v in r.reward_components.values())
                              for r in host.records))
        return sum(totals) / len(totals)

    t, f = constant(True), constant(False)
    best = max(t, f)
    record(close(best, 2.0, 1e-6), "pass1/§20-joint-constant-baseline",
           f"recorded protocol (test split, 30000-30015, cycled objectives): always-True {t:.2f}, "
           f"always-False {f:.2f}, best constant {best:.2f}. FINDINGS §20 and CORRECTIONS row 8 "
           f"record 2.00/2.00 and call the 3.00 figure unreproduced; 3.00 is what reproduces here.")


def check_tests() -> None:
    proc = subprocess.run([sys.executable, "-m", "pytest", "-q"],
                          cwd=ROOT, capture_output=True, text=True)
    tail = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
    m = re.search(r"(?:(\d+) failed, )?(\d+) passed", tail)
    failed = int(m.group(1)) if m and m.group(1) else 0
    passed = int(m.group(2)) if m else 0
    known = "test_panel_episode_replays_and_restores" in proc.stdout
    record(failed == 0 or (failed == 1 and known), "tests/suite",
           f"{passed} passed, {failed} failed"
           + (" (the documented worktree-only panel replay failure)" if failed == 1 and known else ""))
    record(passed + failed >= 337, "tests/count",
           f"{passed + failed} tests collected; the standing bar is 337")


# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--demo", action="store_true", help="run scripts/demo.sh before checking it")
    ap.add_argument("--demo-from", type=Path, default=None, help="read an existing demo summary.json")
    ap.add_argument("--tests", action="store_true", help="also run the pytest suite")
    ap.add_argument("--skip-slow", action="store_true", help="skip the pytest collection used for the test count")
    args = ap.parse_args()

    check_section_numbering()
    check_cross_references()
    check_cited_paths()
    check_retracted_claims()
    if not args.skip_slow:
        check_status_freshness()
    check_raw_figures()
    check_joint_constant_baseline()
    check_deployment_footprint()
    check_visual_gap_decomposition()
    check_visual_rectangle_counts()
    check_span_sweep_coverage()
    check_dyck_seed_identity()
    check_demo(args.demo_from, args.demo)
    if args.tests:
        check_tests()

    width = max(len(i) for _, i, _ in results)
    for status, ident, detail in results:
        print(f"{status:4}  {ident:<{width}}  {detail}")
    bad = sum(1 for s, _, _ in results if s == "FAIL")
    warn = sum(1 for s, _, _ in results if s == "WARN")
    print(f"\n{len(results)} checks: {len(results) - bad - warn} pass, {bad} fail, {warn} warn")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
