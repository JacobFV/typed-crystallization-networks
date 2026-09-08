"""``scope_ambiguity`` — one string, two logical forms.

Compositional semantics and logical language.
"""

from __future__ import annotations

import random

from .._structure import Ident, Lst, Pred, Rec, Str
from ..lesson import Lesson
from ..generators.base import NAMES
from ..generators.semantics import OBJECTS, _shuffled


def gen_scope_ambiguity(rng: random.Random, ctx):
    """One sentence, two logical forms; the world decides which hold.

    ``r1 = forall a. exists b. reads(a,b)`` and ``r2 = exists b. forall a.
    reads(a,b)``. ``r2`` entails ``r1``, so the honest label set is
    {both, r1_only, neither} — the label is drawn first and the world built to
    realize it.
    """
    label = rng.choice(["both", "r1_only", "neither"])
    agents = rng.sample(NAMES, ctx.at(3, 6, default=3))
    books = rng.sample(OBJECTS, 3)
    edges: list[tuple[str, str]] = []
    for _ in range(200):
        edges = []
        if label == "both":
            shared = rng.choice(books)
            for a in agents:
                edges.append((a, shared))
                if rng.random() < 0.5:
                    edges.append((a, rng.choice(books)))
        elif label == "r1_only":
            for a in agents:
                edges.append((a, rng.choice(books)))
                if rng.random() < 0.4:
                    edges.append((a, rng.choice(books)))
        else:
            silent = rng.choice(agents)
            for a in agents:
                if a == silent:
                    continue
                edges.append((a, rng.choice(books)))
        edges = sorted(set(edges))
        read = {a: {b for (x, b) in edges if x == a} for a in agents}
        r1 = all(read[a] for a in agents)
        r2 = any(all(b in read[a] for a in agents) for b in books)
        got = "both" if r2 else ("r1_only" if r1 else "neither")
        if got == label:
            break
    else:                                                  # pragma: no cover
        label = got

    rng.shuffle(edges)
    obs = Rec(agents=Lst([Ident(a) for a in agents]),
              books=Lst([Ident(b) for b in books]),
              world=Lst([Pred("reads", Ident(a), Ident(b)) for a, b in edges]),
              # kept as parts, not as a finished English string: the two
              # quantifiers are what the lesson is about
              sentence=Pred("nl_transitive", Ident("all"), Ident("agent"),
                            Ident("read"), Ident("some"), Ident("book")),
              readings=Lst([Pred("reading", Ident("r1"), Str("forall a exists b reads(a,b)")),
                            Pred("reading", Ident("r2"), Str("exists b forall a reads(a,b)"))]),
              query=Ident("which_readings_hold"))
    return obs, _shuffled(rng, ["both", "r1_only", "neither"]), label, {"edges": edges}


class ScopeAmbiguity(Lesson):
    """One string, two logical forms."""

    id = "scope_ambiguity"
    level = 14
    tags = ("compositional-semantics", "logic")
    teaches = "one string, two logical forms"
    capabilities = ('quantification', 'proof_search')
    axes = {'ambiguity': 4, 'reasoning_depth': 3, 'compositional_depth': 3}
    answers = ['both', 'r1_only', 'neither']

    generate = staticmethod(gen_scope_ambiguity)
