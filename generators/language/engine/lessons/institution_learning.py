"""``institution_learning`` — infer a decision rule from roles, votes and outcomes.

Protocols, institutions, and distributed intelligence.
"""

from __future__ import annotations

import random
from typing import Any, Mapping

from .._structure import Ident, Lst, Num, Pred, Rec, Term
from ..lesson import Lesson
from ..generators.base import NAMES
from ..generators.reflective import _labels, _shuffled


def gen_institution_learning(rng: random.Random, ctx):
    """Infer the decision rule of an institution from the proposals it passed.

    Members have roles and voting weights; four candidate constitutions are
    stated parametrically (a plain threshold, a weighted threshold, unanimity, a
    role whose assent is required). Exactly one reproduces the outcome of every
    recorded proposal — the others are refuted by at least one vote record.
    """
    def decide(kind: str, param: Any, votes: Mapping[str, bool],
               weight: Mapping[str, int], role: Mapping[str, str]) -> bool:
        yes = [m for m, v in votes.items() if v]
        if kind == "threshold":
            return len(yes) >= int(param)
        if kind == "weighted":
            return sum(weight[m] for m in yes) >= int(param)
        if kind == "unanimity":
            return len(yes) == len(votes)
        holder = [m for m in votes if role[m] == param]
        return bool(holder) and votes[holder[0]] and len(yes) >= 2

    fallback = None
    for _ in range(400):
        members = rng.sample(NAMES, 4)
        roles = _shuffled(rng, ["chair", "clerk", "member", "auditor"])
        role = dict(zip(members, roles))
        weight = {m: rng.randint(1, 4) for m in members}
        ids = _labels(rng, "r", 4)
        specs = [("threshold", rng.randint(2, 3)),
                 ("weighted", rng.randint(4, 8)),
                 ("unanimity", 0),
                 ("role_required", rng.choice(["chair", "auditor"]))]
        spec = dict(zip(ids, _shuffled(rng, specs)))
        truth = rng.choice(ids)
        traces = []
        for _ in range(ctx.at(5, 12, default=5)):
            votes = {m: rng.random() < 0.55 for m in members}
            traces.append((votes, decide(*spec[truth], votes, weight, role)))
        ok = [i for i in ids
              if all(decide(*spec[i], v, weight, role) == out for v, out in traces)]
        cand = (members, role, weight, ids, spec, truth, traces)
        if fallback is None:
            fallback = cand
        if ok == [truth]:
            fallback = cand
            break
    members, role, weight, ids, spec, truth, traces = fallback
    facts: list[Term] = []
    for pi, (votes, out) in enumerate(traces):
        for m in members:
            facts.append(Pred("vote", Num(pi), Ident(m), Ident("yes" if votes[m] else "no")))
        facts.append(Pred("outcome", Num(pi), Ident("pass" if out else "fail")))
    rules = [Pred("rule", Ident(i), Ident(spec[i][0]),
                  Num(spec[i][1]) if isinstance(spec[i][1], int) else Ident(str(spec[i][1])))
             for i in ids]
    obs = Rec(members=Lst([Pred("member", Ident(m), Ident(role[m]), Num(weight[m])) for m in members]),
              records=Lst(facts), rules=Lst(_shuffled(rng, rules)),
              query=Ident("which_rule"))
    return obs, _shuffled(rng, ids), truth, {"rule": [str(x) for x in spec[truth]],
                                             "n_proposals": len(traces)}


class InstitutionLearning(Lesson):
    """Infer a decision rule from roles, votes and outcomes."""

    id = "institution_learning"
    level = 122
    tags = ("protocols", "institutions", "distributed-intelligence")
    teaches = "infer a decision rule from roles, votes and outcomes"
    capabilities = ('multi_agent_coordination', 'scientific_induction', 'ontology_learning')
    axes = {'world_complexity': 4, 'reasoning_depth': 4, 'discourse_horizon': 3}

    generate = staticmethod(gen_institution_learning)
