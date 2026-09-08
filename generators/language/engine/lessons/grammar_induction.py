"""``grammar_induction`` — inducing a fresh grammar from labelled strings.

Language as action.
"""

from __future__ import annotations

import itertools
import random
from typing import Mapping, Sequence

from .._structure import Ident, Lst, Pred, Rec, Str
from ..lesson import Lesson
from ..generators.semantics import _nonce_words, _shuffled


def gen_grammar_induction(rng: random.Random, ctx):
    """A fresh regular grammar per episode; pick the grammatical held-out string.

    Six nonce tokens are partitioned into two classes and a three-slot class
    template is drawn; labelled examples are the only evidence for either. The
    candidate strings are all unseen, so the answer requires the *rule*, not
    recall of a shown string.

    The episode is kept only if it is **identifiable**: every (partition,
    template) hypothesis consistent with the labelled examples has to accept the
    same single candidate. Without that check the evidence sometimes leaves a
    token's class open — a token that never appears in a positive example is
    unconstrained — and two different candidates are each "correct" under a
    different consistent grammar, which would make the label arbitrary.
    """
    slots = ctx.at(3, 6, default=3)
    for _ in range(300):
        toks = _nonce_words(rng, 6, 2)
        c1, c2 = toks[:3], toks[3:]
        cls = {t: ("c1" if t in c1 else "c2") for t in toks}
        template = [rng.choice(["c1", "c2"]) for _ in range(slots)]
        if len(set(template)) == 1:                       # a trivial grammar teaches nothing
            continue

        def accepts(cl: Mapping[str, str], tp: Sequence[str], s: Sequence[str]) -> bool:
            return all(cl[t] == tp[i] for i, t in enumerate(s))

        def sample(valid: bool) -> tuple[str, ...] | None:
            for _ in range(400):
                s = tuple(rng.choice(toks) for _ in range(slots))
                if accepts(cls, template, s) == valid:
                    return s
            return None                                   # pragma: no cover

        pos: set[tuple[str, ...]] = set()
        neg: set[tuple[str, ...]] = set()
        drawn = [sample(True) for _ in range(14)] + [sample(False) for _ in range(10)]
        if any(d is None for d in drawn):
            continue                                      # pragma: no cover
        for d in drawn:
            (pos if accepts(cls, template, d) else neg).add(d)
        pos_l, neg_l = sorted(pos), sorted(neg)
        if len(pos_l) < 7 or len(neg_l) < 5:
            continue
        answer_s, pos_l = pos_l[-1], pos_l[:6]            # the answer is a held-out positive
        neg_l = neg_l[:5]
        if answer_s in pos_l:
            continue                                      # pragma: no cover
        cands = {answer_s}
        for _ in range(60):
            if len(cands) == 4:
                break
            c = sample(False)
            if c is not None and c not in neg_l:
                cands.add(c)
        if len(cands) != 4:
            continue                                      # pragma: no cover

        # identifiability: enumerate the whole hypothesis family and require that
        # every hypothesis fitting the labels picks out the same one candidate
        answers = set()
        for mask in range(64):
            cl = {t: ("c1" if (mask >> i) & 1 else "c2") for i, t in enumerate(toks)}
            for tp in itertools.product(("c1", "c2"), repeat=slots):
                if all(accepts(cl, tp, s) for s in pos_l) and \
                        not any(accepts(cl, tp, s) for s in neg_l):
                    answers.add(tuple(sorted(c for c in cands if accepts(cl, tp, c))))
        if answers != {(answer_s,)}:
            continue
        pos, neg = set(pos_l), set(neg_l)
        break
    else:                                                  # pragma: no cover
        raise RuntimeError("no identifiable grammar found")

    def j(s: Sequence[str]) -> str:
        return " ".join(s)

    examples = _shuffled(rng, [Pred("ex", Str(j(s)), Ident("yes")) for s in sorted(pos)]
                         + [Pred("ex", Str(j(s)), Ident("no")) for s in sorted(neg)])
    cand_list = _shuffled(rng, sorted(cands))
    obs = Rec(examples=Lst(examples),
              candidates=Lst([Str(j(c)) for c in cand_list]),
              query=Ident("which_is_grammatical"))
    vocab = [j(c) for c in cand_list]
    return obs, vocab, j(answer_s), {"template": template, "classes": cls}


class GrammarInduction(Lesson):
    """Inducing a fresh grammar from labelled strings."""

    id = "grammar_induction"
    level = 29
    tags = ("pragmatics", "language-as-action")
    teaches = "inducing a fresh grammar from labelled strings"
    capabilities = ('finite_state_induction', 'scientific_induction', 'abstraction')
    axes = {'grammar_complexity': 4, 'lexical_novelty': 4, 'reasoning_depth': 3}

    generate = staticmethod(gen_grammar_induction)
