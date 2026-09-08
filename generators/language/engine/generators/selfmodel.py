"""Support for the self-modeling, open-ended and value lessons.

Private. Ported alongside the lessons that use it; every function here is
called by at least one generator in :mod:`langcurriculum.lessons`.
"""

from __future__ import annotations

import random
from itertools import permutations
from typing import Any, Mapping, Sequence

from ..binding import SlotVocabulary
from .._structure import Ident, Lst, Pred, Term
from .base import COLORS, SHAPES

SKILLS = ["parsing", "arithmetic", "search", "recall", "planning", "grounding"]


STAGES = ("perception", "memory", "representation", "inference", "planning", "execution")


SLOTS = ("parser", "reasoner", "planner")


MODULE_NAMES = ["lexer", "chart", "earley", "tabling", "astar", "bfs", "sat", "rete",
                "trie", "kv", "beam", "cegis"]


FEATURES = ["recursion", "typing", "search", "memory", "probabilistic", "symbolic_io"]


PROPS = SlotVocabulary("property", ["fragile", "portable", "magnetic", "opaque", "heavy", "hollow"])


KINDS = SlotVocabulary("kind", ["tool", "vessel", "device", "fixture"])


SIZES = SlotVocabulary("size", ["small", "medium", "big"])


ASSUMPTIONS = ("positivity", "monotonicity", "boundedness", "small_steps")


def _shuffled(rng: random.Random, xs: Sequence[Any]) -> list[Any]:
    ys = list(xs)
    rng.shuffle(ys)
    return ys


def _labels(rng: random.Random, prefix: str, n: int) -> list[str]:
    """``n`` distinct labels in a random order.

    Structure is generated over integer indices and only then given names, so a
    label carries no information about the role its entity plays and the answer
    *label* is uniform whatever the structure's own distribution is.
    """
    return _shuffled(rng, [f"{prefix}{i}" for i in range(n)])


def _yesno(rng: random.Random, truth: bool) -> tuple[list[str], str]:
    return _shuffled(rng, ["yes", "no"]), ("yes" if truth else "no")


def _rules(*lines: str) -> Term:
    """The scoring semantics, stated in the observation.

    Without this the learner would have to induce the success/cost rule across
    episodes; with it every episode is self-contained and the answer is a pure
    function of what the agent can see.
    """
    # The rule is a sentence, so it is written as one. As `Ident` it was a
    # name and names pass through untouched, which put
    # "a_theory_is_lossless_iff_it_predicts_every_observation_exactly" on the
    # page -- in English too, where it is not a sentence either. As a headed
    # term with no arguments it goes through the label path, which spells the
    # underscores out and renders it a word at a time.
    return Lst([Pred("rule", Pred(t)) for t in lines])


def _paths(edges: Sequence[tuple[int, int]], src: int, dst: int) -> list[list[int]]:
    out: list[list[int]] = []

    def go(node: int, acc: list[int]) -> None:
        if node == dst:
            out.append(list(acc))
            return
        for ei, (u, v) in enumerate(edges):
            if u == node:
                acc.append(ei)
                go(v, acc)
                acc.pop()

    go(src, [])
    return out


def _closure(facts: Sequence[tuple[str, str]], rules: Sequence[tuple[str, str]],
             kind_of: Mapping[str, str]) -> set[tuple[str, str]]:
    out = set(facts)
    for k, p in rules:
        for e, ke in kind_of.items():
            if ke == k:
                out.add((e, p))
    return out


def _predict(family: str, p1: int, p2: int, x: int) -> int:
    if family == "linear":
        return p1 * x + p2
    if family == "square":
        return x * x + p2
    return p2


def _assumption_status(seq: Sequence[int], bound: int, step: int) -> dict[str, bool]:
    return {
        "positivity": all(v > 0 for v in seq),
        "monotonicity": all(seq[i + 1] > seq[i] for i in range(len(seq) - 1)),
        "boundedness": all(v < bound for v in seq),
        "small_steps": all(abs(seq[i + 1] - seq[i]) <= step for i in range(len(seq) - 1)),
    }


def _rigid_pair(rng: random.Random, n: int) -> tuple[list[tuple[int, int]], list[int]]:
    """A digraph with no non-trivial automorphism, plus a random relabelling.

    Every node must appear in some edge: a node with no edges is not written into
    the observation at all, so its correspondence would be unrecoverable from
    what the learner sees even though the generator "knows" it.
    """
    nodes = list(range(n))
    for _ in range(400):
        edges = sorted({(rng.randrange(n), rng.randrange(n)) for _ in range(n + 3)})
        edges = [e for e in edges if e[0] != e[1]]
        if len(edges) < n or len({x for e in edges for x in e}) < n:
            continue
        es = set(edges)
        autos = sum(1 for p in permutations(nodes)
                    if {(p[u], p[v]) for u, v in es} == es)
        if autos == 1:
            return edges, list(rng.sample(nodes, n))
    # pragma: no cover - a directed path is rigid and covers every node
    return [(i, i + 1) for i in range(n - 1)], list(rng.sample(nodes, n))


def _render_claim(quant: str, neg: bool, subj: str, pred: str) -> str:
    if quant == "all":
        return f"no {subj} is {pred}" if neg else f"every {subj} is {pred}"
    return f"some {subj} is not {pred}" if neg else f"some {subj} is {pred}"


def _claim_term(quant: str, neg: bool, subj: str, pred: str) -> Term:
    """The claim as **structure**, not as a finished English sentence.

    ``_render_claim`` builds the English directly, which is the right thing for
    a resource with one language and the wrong thing for one with four hundred:
    a finished string carries no parts, so nothing downstream can say it in
    anything else. Handing over the quantifier, the polarity, the restriction
    and the scope lets each grammar assemble them the way its language does —
    *no prism is yellow*, *kein Prisma ist gelb*, *bütün prizma sarı değil*.
    """
    return Pred("nl_claim", Ident(quant), Ident("neg" if neg else "pos"),
                Ident(subj), Ident(pred))


def _formal_claim(quant: str, neg: bool, subj: str, pred: str) -> Term:
    x = Ident("x")
    body = Pred("not", Pred(pred, x)) if neg else Pred(pred, x)
    if quant == "all":
        return Pred("forall", x, Pred("implies", Pred(subj, x), body))
    return Pred("exists", x, Pred("and", Pred(subj, x), body))


def _claim_pool(rng: random.Random) -> tuple[tuple, list[tuple]]:
    sh = rng.choice(SHAPES)
    col, col2 = rng.sample(COLORS, 2)
    quant = rng.choice(["all", "some"])
    neg = rng.random() < 0.5
    other = "some" if quant == "all" else "all"
    true = (quant, neg, sh, col)
    pool = [(other, neg, sh, col), (quant, not neg, sh, col), (quant, neg, col, sh),
            (quant, neg, sh, col2), (other, not neg, sh, col)]
    picks = rng.sample(pool, 3)
    return true, picks
