"""``paraphrase`` — semantic equivalence under voice and synonymy.

Language as action.
"""

from __future__ import annotations

import random

from .._structure import Ident, Lst, Pred, Rec
from ..lesson import Lesson
from ..generators.base import NAMES
from ..generators.semantics import TRANS_VERBS, _shuffled


def gen_paraphrase(rng: random.Random, ctx):
    """Same logical form, different surface: voice alternation plus a synonym.

    Each candidate is given as ``(form, w1, w2, w3)`` where ``form`` says how to
    read the argument positions, so recovering the logical form is a symbolic
    operation rather than string matching; the distractors keep the words and
    change the roles, the verb, or the patient.
    """
    people = rng.sample(NAMES, ctx.at(3, 6, default=3))
    agent, patient, other = people[0], people[1], people[2]
    verbs = rng.sample(TRANS_VERBS, ctx.at(3, 6, default=3))
    v_a, v_b, v_c = verbs[0], verbs[1], verbs[2]        # v_a ~ v_b are synonyms here
    form = rng.choice(["active", "passive"])
    flip = {"active": "passive", "passive": "active"}[form]

    def sent(f: str, ag: str, v: str, pt: str) -> tuple[str, str, str, str]:
        return (f, ag, v, pt) if f == "active" else (f, pt, v, ag)

    good = sent(flip, agent, v_b, patient)
    bad = [sent(flip, patient, v_b, agent),             # roles swapped
           sent(flip, agent, v_c, patient),             # not a synonym
           sent(form, agent, v_a, other)]               # different patient
    bad += [sent(form, agent, v_a, p) for p in people[3:]]      # more wrong patients
    bad += [sent(flip, agent, v, patient) for v in verbs[3:]]   # more non-synonyms
    cands = _shuffled(rng, [good] + bad)
    cids = _shuffled(rng, [f"c{i + 1}" for i in range(len(cands))])
    answer = cids[cands.index(good)]

    target = sent(form, agent, v_a, patient)
    obs = Rec(sentence=Pred("sentence", Ident(target[0]), Ident(target[1]),
                            Ident(target[2]), Ident(target[3])),
              synonyms=Lst([Pred("synonym", Ident(v_a), Ident(v_b))]),
              candidates=Lst([Pred("candidate", Ident(cid), Ident(c[0]), Ident(c[1]),
                                   Ident(c[2]), Ident(c[3]))
                              for cid, c in zip(cids, cands)]),
              query=Ident("which_is_a_paraphrase"))
    return obs, _shuffled(rng, cids), answer, {"agent": agent, "patient": patient,
                                               "verb": v_a, "synonym": v_b}


class Paraphrase(Lesson):
    """Semantic equivalence under voice and synonymy."""

    id = "paraphrase"
    level = 32
    tags = ("pragmatics", "language-as-action")
    teaches = "semantic equivalence under voice and synonymy"
    capabilities = ('abstraction', 'proof_search')
    axes = {'grammar_complexity': 3, 'compositional_depth': 3, 'ambiguity': 2}

    generate = staticmethod(gen_paraphrase)
