"""``ontology_revision`` — the minimal revision that restores consistency after an anomaly.

Ontology and representation.
"""

from __future__ import annotations

import random

from .._structure import Ident, Lst, Pred, Rec
from ..lesson import Lesson
from ..generators.ontology import ENTITIES, PROPS, _apply_op, _assign_entities, _consistent, _inherited, _label_options, _onto_facts, _op_symbol, _random_hierarchy, _revision_pool, _shuffled


def gen_ontology_revision(rng: random.Random, ctx):
    """An established taxonomy meets one anomalous entity; which revision fixes it?

    The anomaly either *lacks* a property its declared type inherits or *has* one
    the type does not license. Four revisions (exception, extra attribute,
    split, merge, generalization, reassignment) are presented; each is applied by
    :func:`_apply_op` and re-checked with :func:`_consistent`, and exactly one of
    the four presented candidates restores consistency.
    """
    for _ in range(200):
        own, parent, leaves, props, _ = _random_hierarchy(rng)
        inst = _assign_entities(rng, leaves, ctx.at(2, 5, default=2))
        observed = {e: _inherited(own, parent, inst[e]) for e in inst}
        if not _consistent(own, parent, inst, observed):
            continue
        t_new = rng.choice([t for t in leaves if sum(1 for e in inst if inst[e] == t) >= 1])
        e_new = next(e for e in ENTITIES if e not in inst)
        base = _inherited(own, parent, t_new)
        kind = rng.choice(["missing", "extra"])
        spare = [p for p in PROPS if all(p not in own[t] for t in own)]
        if kind == "missing":
            gone = rng.choice(sorted(base))
            obs_props = set(base) - {gone}
            anomaly = {"kind": kind, "prop": gone}
        else:
            got = rng.choice(spare)
            obs_props = set(base) | {got}
            anomaly = {"kind": kind, "prop": got}
        inst2, observed2 = dict(inst), dict(observed)
        inst2[e_new] = t_new
        observed2[e_new] = obs_props
        if _consistent(own, parent, inst2, observed2):
            continue                                 # not actually anomalous

        props_seen = {p for v in observed2.values() for p in v} | {
            p for t in own for p in own[t]}
        good, bad, seen = [], [], set()
        for op in _shuffled(rng, _revision_pool(own, parent, inst2, observed2,
                                                e_new, t_new, props_seen)):
            key = str(_op_symbol(op))
            if key in seen:
                continue
            seen.add(key)
            o2, pa2, in2, ex2, xt2 = _apply_op(own, parent, inst2, (), (), op)
            (good if _consistent(o2, pa2, in2, observed2, ex2, xt2) else bad).append(op)
        if not good or len(bad) < 3:
            continue
        correct = good[0]
        # distractors of the *same operator kind* wherever possible: if the kind
        # of the revision predicted whether it works, "always pick the exception"
        # would score far above chance without any ontology being checked at all
        same = [op for op in bad if op[0] == correct[0]]
        distract = (same + [op for op in bad if op[0] != correct[0]])[:3]
        if len(distract) == 3:
            break
    else:                                            # pragma: no cover - construction
        raise RuntimeError("ontology_revision: no episode")

    pairs, vocab, answer = _label_options(rng, correct, distract)
    obs = Rec(
        taxonomy=_onto_facts(rng, own, parent, inst),
        observations=Lst(_shuffled(rng, [Pred("observed", Ident(e), Ident(p))
                                         for e in sorted(observed2) for p in sorted(observed2[e])])),
        anomaly=Pred("new_entity", Ident(e_new), Ident(t_new)),
        revisions=Lst([Pred("revision", Ident(lab), _op_symbol(op)) for lab, op in pairs]),
        query=Pred("restores_consistency", Ident(e_new)),
    )
    hidden = {"anomaly": anomaly["kind"], "anomalous_prop": anomaly["prop"],
              "fix_kind": correct[0], "n_valid_in_pool": len(good), "answer": answer}
    return obs, vocab, answer, hidden


class OntologyRevision(Lesson):
    """The minimal revision that restores consistency after an anomaly."""

    id = "ontology_revision"
    level = 62
    tags = ("ontology", "representation")
    teaches = "the minimal revision that restores consistency after an anomaly"
    capabilities = ('non_monotonic_revision', 'consistency_checking', 'ontology_induction')
    axes = {'world_complexity': 3, 'reasoning_depth': 4, 'ambiguity': 2, 'compositional_depth': 3}

    generate = staticmethod(gen_ontology_revision)
