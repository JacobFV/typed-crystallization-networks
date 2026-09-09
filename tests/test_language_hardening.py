"""The language curriculum's anti-exploit draws, and the stream they replaced.

Two things have to hold at once. The lessons named in
``generators.language.engine.context.HARDENING`` must no longer be answerable by
the cheap heuristic each one was found to admit --- see
``research/lesson-audit/RESULTS.md`` --- and the draw those lessons used before
must remain reachable, bit for bit, as ``hardening="none"``.

The second half is checked the way FINDINGS section 10 checked the logic
generator: a digest of every implemented lesson over a grid of languages,
difficulties and seeds, compared against a fixture recorded from the generator
as it stood before the audit.
"""

import hashlib
import json
import os
import random
import re
from collections import Counter

import pytest

from generators.language.engine.context import (DEFAULT_HARDENING, HARDENING,
                                                NO_HARDENING, GenerationContext,
                                                resolve_hardening)
from generators.language.engine.generators.base import _is_balanced
from generators.language.engine.registry import all_lessons, get

FIXTURE = os.path.join(os.path.dirname(__file__), "data",
                       "language_preaudit_stream.json")
LANGS = ("english", "turkish")
DIFFS = (None, 0.0, 1.0)
SEEDS = range(4)
LESSON_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "generators", "language", "engine", "lessons")


def implemented():
    return sorted(k for k, l in all_lessons().items() if l.status == "implemented")


def stream_digest(lesson_id, hardening):
    """One digest of a lesson's episodes over the whole configuration grid."""
    h = hashlib.blake2b(digest_size=16)
    for lang in LANGS:
        for d in DIFFS:
            for s in SEEDS:
                try:
                    rec = get(lesson_id).example(seed=s, language=lang, difficulty=d,
                                                 hardening=hardening).to_dict()
                except Exception as exc:                 # pragma: no cover
                    h.update(f"!{type(exc).__name__}".encode())
                    continue
                rec.pop("instance_id")                   # a function of the regime
                h.update(json.dumps(rec, sort_keys=True,
                                    ensure_ascii=False).encode("utf8"))
    return h.hexdigest()


# ---------------------------------------------------------------- the regime

def test_hardening_names_are_lessons_that_consult_them():
    for name in HARDENING:
        assert name in all_lessons(), name
        src = open(os.path.join(LESSON_DIR, f"{name}.py")).read()
        assert f'ctx.hardens("{name}")' in src, f"{name} declares hardening it never reads"


def test_resolve_hardening_spellings():
    assert resolve_hardening(None) == DEFAULT_HARDENING
    assert resolve_hardening("none") == resolve_hardening(False) == NO_HARDENING
    assert resolve_hardening("all") == resolve_hardening(True) == HARDENING
    assert resolve_hardening("unification,ellipsis") == {"unification", "ellipsis"}
    assert resolve_hardening(["unification"]) == {"unification"}
    with pytest.raises(ValueError):
        resolve_hardening("no_such_lesson")
    with pytest.raises(ValueError):
        GenerationContext().hardens("no_such_lesson")


def test_default_is_hardened_and_selective():
    ex = get("unification").example(seed=0)
    assert ex.metadata is not None
    assert get("unification").example(seed=0, hardening="unification").prompt == ex.prompt
    assert get("unification").example(seed=0, hardening="ellipsis").prompt != ex.prompt


def test_instance_id_separates_the_regimes():
    a = get("ellipsis").example(seed=1)
    b = get("ellipsis").example(seed=1, hardening="none")
    assert a.instance_id != b.instance_id
    assert a.instance_id == get("ellipsis").example(seed=1, hardening="all").instance_id


def test_generator_passes_the_regime_through():
    from generators.language.generator import Implementation
    from tcn.generation import Host
    a = Host.create("language", seed=3, configuration={"lesson": "ellipsis"})
    b = Host.create("language", seed=3, configuration={"lesson": "ellipsis",
                                                       "hardening": "none"})
    assert a.state["example"]["prompt"] != b.state["example"]["prompt"]
    c = Host.create("language", seed=3, configuration={"lesson": "ellipsis",
                                                       "difficulty": 1.0})
    assert c.state["example"]["difficulty"] == 1.0


# ------------------------------------------------------- the legacy stream

def test_legacy_stream_is_bit_identical_to_the_pre_audit_generator():
    """179 lessons x 6 configurations x 4 seeds, against a recorded fixture."""
    want = json.load(open(FIXTURE))
    assert want["grid"] == {"languages": list(LANGS),
                            "difficulties": list(DIFFS), "seeds": len(SEEDS)}
    got = {lid: stream_digest(lid, "none") for lid in implemented()}
    assert set(got) == set(want["digests"])
    differing = sorted(k for k in got if got[k] != want["digests"][k])
    assert differing == [], differing


def test_default_stream_changes_only_where_declared():
    want = json.load(open(FIXTURE))["digests"]
    changed = sorted(lid for lid in implemented()
                     if stream_digest(lid, None) != want[lid])
    # the two composing lessons draw sub-episodes under the same regime, so they
    # move with the lessons they compose
    composers = {"general_language_agent", "symbolic_generalist"}
    assert set(HARDENING) <= set(changed) <= set(HARDENING) | composers


# ------------------------------------------------------- the fixes themselves

def test_context_free_negatives_preserve_the_bracket_counts():
    L = get("context_free_language")
    agree = same_len = 0
    n = 400
    for s in range(n):
        h = L.structured(seed=s)
        st = h["hidden"]["string"]
        counted = st.count("(") == st.count(")")
        agree += counted == (h["answer"] == "yes")
        same_len += len(st) >= 4
    assert same_len == n
    assert 0.40 <= agree / n <= 0.62, agree / n           # counting is now chance
    legacy = sum((lambda st: st.count("(") == st.count(")"))(
        L.structured(seed=s, hardening="none")["hidden"]["string"])
        == (L.structured(seed=s, hardening="none")["answer"] == "yes")
        for s in range(200))
    assert legacy == 200                                  # the defect, still reachable


def test_context_free_answer_is_dyck_membership():
    L = get("context_free_language")
    for s in range(300):
        h = L.structured(seed=s)
        assert (h["answer"] == "yes") == _is_balanced(h["hidden"]["string"])


def test_ellipsis_antecedent_is_never_the_last_clause():
    L = get("ellipsis")
    last = 0
    for s in range(200):
        h = L.structured(seed=s)
        lines = h["observation"]["fields"]["discourse"]["items"]
        kinds = [x["head"] for x in lines]
        assert kinds[-1] == "clause"                      # a distractor follows the gap
        gap = max(i for i, k in enumerate(kinds) if k in ("gap", "vp_gap"))
        last += gap == len(kinds) - 1
    assert last == 0


def test_center_embedding_queries_more_than_the_outermost_subject():
    qs = {get("center_embedding").structured(seed=s)["hidden"].get("queried")
          for s in range(200)}
    assert len(qs) > 1
    assert all(get("center_embedding").structured(seed=s, hardening="none")["hidden"]
               .get("queried") is None for s in range(5))


def test_unification_has_exactly_one_matching_fact():
    L = get("unification")
    for s in range(200):
        h = L.structured(seed=s)
        f = h["observation"]["fields"]
        pattern = [t["v"] for t in f["pattern"]["args"]]
        facts = [[t["v"] for t in x["args"]] for x in f["facts"]["items"]]
        var = f["query"]["args"][0]["v"]
        hits = [g for g in facts
                if all(p == var or p == a for p, a in zip(pattern, g))]
        assert len(hits) == 1 and hits[0][pattern.index(var)] == h["answer"]
        assert len(facts) >= 3


def test_symbol_equivalence_lexicon_names_every_colour():
    from generators.language.engine.generators.base import COLORS
    L = get("symbol_equivalence")
    seen = Counter()
    for s in range(200):
        h = L.structured(seed=s)
        facts = h["observation"]["fields"]["lexicon"]["items"]
        assert {x["args"][1]["v"] for x in facts} == set(COLORS)
        seen[h["answer"]] += 1
    assert len(seen) == len(COLORS)
    assert max(seen.values()) / 200 < 0.30                # no colour is the default


def test_tree_to_sequence_asks_for_more_than_the_first_leaf():
    L = get("tree_to_sequence")
    first = 0
    for s in range(200):
        h = L.structured(seed=s)
        y = h["hidden"]["yield"]
        assert h["answer"] == y[h["hidden"]["index"]]
        first += h["answer"] == y[0]
    assert first < 120


def test_underspecification_line_count_is_not_the_answer():
    L = get("underspecification_reasoning")
    pairs = []
    for s in range(300):
        h = L.structured(seed=s)
        lines = len(h["observation"]["fields"]["compatibility"]["items"])
        pairs.append((lines, int(h["answer"])))
        assert int(h["answer"]) == h["hidden"]["admissible"]
    assert sum(a == b for a, b in pairs) < 60             # was 300 of 300
    best = max(Counter(b for a, b in pairs if a == k).most_common(1)[0][1]
               for k in {a for a, _ in pairs})
    assert best < len(pairs)


def test_parse_depth_string_space_is_wide():
    L = get("parse_depth")
    new = {L.structured(seed=s)["hidden"]["string"] for s in range(600)}
    old = {L.structured(seed=s, hardening="none")["hidden"]["string"] for s in range(600)}
    assert len(new) > 4 * len(old)
    for s in range(200):
        h = L.structured(seed=s)
        assert _is_balanced(h["hidden"]["string"])


def test_nesting_depth_compare_strings_have_equal_length():
    L = get("nesting_depth_compare")
    for s in range(300):
        f = L.structured(seed=s)["observation"]["fields"]
        assert len(f["left"]["items"]) == len(f["right"]["items"])


def test_next_symbol_transition_is_shown_and_the_period_varies():
    L = get("next_symbol")
    periods = Counter()
    for s in range(300):
        h = L.structured(seed=s)
        seq = [t["v"] for t in h["observation"]["fields"]["sequence"]["items"]]
        # the pair (last symbol -> answer) has to appear earlier, or the episode
        # is not answerable from what it shows
        assert any(seq[i] == seq[-1] and seq[i + 1] == h["answer"]
                   for i in range(len(seq) - 1))
        periods[h["hidden"]["cycle"]] += 1
    assert len(periods) >= 3 and max(periods.values()) / 300 < 0.6


def test_symbol_discrimination_scale_is_arbitrary():
    L = get("symbol_discrimination")
    by_value = {}
    for s in range(800):
        h = L.structured(seed=s)
        v = h["observation"]["fields"]["query"]["args"][0]["v"]
        by_value.setdefault(v, Counter())[h["answer"]] += 1
    n = sum(sum(c.values()) for c in by_value.values())
    best = sum(c.most_common(1)[0][1] for c in by_value.values()) / n
    assert best < 0.75                                    # reading the number was 0.80


def test_paradigm_shift_every_assumption_is_violated_somewhere():
    from generators.language.engine.generators.selfmodel import (ASSUMPTIONS,
                                                                 _assumption_status)
    L = get("paradigm_shift")
    for s in range(120):
        h = L.structured(seed=s)
        f = h["observation"]["fields"]
        bound, step = h["hidden"]["bound"], h["hidden"]["step"]
        broken = set()
        series = [[r["args"][1]["v"] for r in f["new_regime"]["items"]]]
        series += [[r["args"][1]["v"] for r in g["items"]]
                   for g in f["control_regimes"]["items"]]
        for xs in series:
            st = _assumption_status(xs, bound, step)
            broken |= {a for a in ASSUMPTIONS if not st[a]}
        assert broken == set(ASSUMPTIONS)


def test_language_culture_answer_is_not_the_commonest_word():
    L = get("language_culture")
    hit = 0
    for s in range(300):
        h = L.structured(seed=s)
        log = str(h["observation"]["fields"]["transmission"])
        counts = {w: log.count("'" + w + "'") for w in h["choices"]}
        top = max(counts.values())
        hit += counts[h["answer"]] == top and list(counts.values()).count(top) == 1
    assert hit / 300 < 0.45                               # was 0.57


def test_presupposition_shows_competing_utterances():
    L = get("presupposition")
    for s in range(120):
        f = L.structured(seed=s)["observation"]["fields"]
        assert len(f["context"]["items"]) >= 2


def test_every_lesson_still_verifies():
    from generators.language.engine.verify import verify_lesson
    for lid in sorted(HARDENING):
        r = verify_lesson(lid, episodes=200)
        assert r["ok"] is True, (lid, r)
