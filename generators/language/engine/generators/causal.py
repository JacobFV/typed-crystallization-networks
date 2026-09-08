"""Support for analogy, causality, planning and program lessons.

Private. Ported alongside the lessons that use it; every function here is
called by at least one generator in :mod:`langcurriculum.lessons`.
"""

from __future__ import annotations

import random
from typing import Any, Mapping, Sequence

from .._structure import Ident, Lst, Num, Pred, Term

_CONS = "kmtszlpvrbdgn"


_VOWELS = "aeiou"


def _nonce(rng: random.Random) -> str:
    return rng.choice(_CONS) + rng.choice(_VOWELS) + rng.choice(_CONS)


def _nonce_names(rng: random.Random, n: int) -> list[str]:
    """``n`` distinct pronounceable nonce words, fresh every episode."""
    out: list[str] = []
    while len(out) < n:
        w = _nonce(rng)
        if w not in out:
            out.append(w)
    return out


def _options(rng: random.Random, correct: Any, wrong: Sequence[Any]) -> tuple[list[Any], int]:
    """Shuffle ``[correct, *wrong]``; return the order and the correct index.

    Every multiple-choice lesson here goes through this, which is what makes the
    correct label uniform and a constant guesser worth 1/k.
    """
    items = [correct, *wrong]
    order = list(range(len(items)))
    rng.shuffle(order)
    return [items[i] for i in order], order.index(0)


def _labels(prefix: str, n: int) -> list[str]:
    return [f"{prefix}{i}" for i in range(n)]


def _qg_colors(at: Mapping[str, str], contains: Mapping[str, str], color: Mapping[str, str],
               target: str, rooms: Sequence[str], boxes: Sequence[str],
               palette: Sequence[str]) -> set[str]:
    """Every colour the goal could take, over the worlds consistent with the
    *visible* facts. ``|result| == 1`` is derivability; anything else is not."""
    if target in at:
        cand_rooms = [at[target]]
    else:                                    # any room nobody else occupies
        taken = {r for p, r in at.items()}
        cand_rooms = [r for r in rooms if r not in taken]
    out: set[str] = set()
    for r in cand_rooms:
        if r in contains:
            cand_boxes = [contains[r]]
        else:                                # any box not already placed elsewhere
            placed = {b for rr, b in contains.items() if rr != r}
            cand_boxes = [b for b in boxes if b not in placed]
        for b in cand_boxes:
            out |= {color[b]} if b in color else set(palette)
    return out


def _tree_values(parent: Mapping[str, str], neg: Mapping[str, int], roots: Mapping[str, int],
                 order: Sequence[str], forced: Mapping[str, int] | None = None) -> dict[str, int]:
    """Run the boolean SCM. ``forced`` is a do(): it overrides the mechanism."""
    forced = forced or {}
    vals: dict[str, int] = {}
    for v in order:
        if v in forced:
            vals[v] = forced[v]
        elif v in parent:
            vals[v] = vals[parent[v]] ^ neg[v]
        else:
            vals[v] = roots[v]
    return vals


def _plan_domain(rng: random.Random) -> dict[str, Any] | None:
    """A locations/doors/key domain, or None if this draw is inadmissible."""
    locs = _nonce_names(rng, 5)
    key = _nonce(rng)
    while key in locs:
        key = _nonce(rng)
    edges: set[tuple[str, str]] = set()
    order = list(locs)
    rng.shuffle(order)
    for i in range(1, len(order)):                       # random spanning tree
        j = rng.randrange(i)
        edges.add(tuple(sorted((order[i], order[j]))))
    for _ in range(rng.randint(1, 2)):                   # a couple of extra ways round
        a, b = rng.sample(locs, 2)
        edges.add(tuple(sorted((a, b))))
    edges_l = sorted(edges)
    nbrs: dict[str, list[str]] = {l: [] for l in locs}
    for a, b in edges_l:
        nbrs[a].append(b)
        nbrs[b].append(a)

    locked = tuple(rng.choice(edges_l)) if rng.random() < 0.75 else None
    key_at = rng.choice(locs)
    start = rng.choice([l for l in locs if len(nbrs[l]) >= 3] or locs)
    goal = rng.choice([l for l in locs if l != start])

    def acts(state: tuple[str, bool]) -> list[tuple[str, str]]:
        loc, has = state
        out = [("move", n) for n in sorted(nbrs[loc])
               if locked is None or has or tuple(sorted((loc, n))) != locked]
        if loc == key_at and not has:
            out.append(("take", key))
        return out

    def step(state: tuple[str, bool], a: tuple[str, str]) -> tuple[str, bool]:
        return (a[1], state[1]) if a[0] == "move" else (state[0], True)

    def dist(state: tuple[str, bool]) -> int | None:
        seen, frontier, d = {state}, [state], 0
        while frontier:
            if any(s[0] == goal for s in frontier):
                return d
            nxt = []
            for s in frontier:
                for a in acts(s):
                    t = step(s, a)
                    if t not in seen:
                        seen.add(t)
                        nxt.append(t)
            frontier, d = nxt, d + 1
        return None

    s0 = (start, False)
    d0 = dist(s0)
    applicable = acts(s0)
    if d0 is None or not (2 <= d0 <= 6) or len(applicable) < 3:
        return None
    best = [a for a in applicable if (lambda d: d is not None and d == d0 - 1)(dist(step(s0, a)))]
    if len(best) != 1:
        return None
    return {"locs": locs, "edges": edges_l, "locked": locked, "key": key, "key_at": key_at,
            "start": start, "goal": goal, "actions": applicable, "best": best[0], "length": d0}


def _plan_fallback(rng: random.Random) -> dict[str, Any]:
    """A hand-built domain with a provably unique optimal first action, used only
    if the random draw fails repeatedly. Names are still fresh per episode."""
    s, x, y, z, g = _nonce_names(rng, 5)
    key = _nonce(rng)
    edges = sorted({tuple(sorted(e)) for e in [(s, x), (s, y), (s, z), (x, g), (y, z)]})
    return {"locs": [s, x, y, z, g], "edges": edges, "locked": None, "key": key, "key_at": y,
            "start": s, "goal": g, "actions": [("move", n) for n in sorted([x, y, z])],
            "best": ("move", x), "length": 2}


_PROC_MOD = 10


def _proc_simple(rng: random.Random, vs: Sequence[str]) -> tuple[str, str, int]:
    op = rng.choice(["set", "add", "add", "mul"])
    k = {"set": lambda: rng.randrange(_PROC_MOD),
         "add": lambda: rng.randint(1, 9),
         "mul": lambda: rng.choice([3, 7, 9])}[op]()
    return (op, rng.choice(list(vs)), k)


def _proc_body(rng: random.Random, vs: Sequence[str], n: int) -> list[tuple]:
    return [_proc_simple(rng, vs) for _ in range(n)]


def _proc_exec(prog: Sequence[tuple], state: dict[str, int]) -> dict[str, int]:
    st = dict(state)
    for stmt in prog:
        head = stmt[0]
        if head == "set":
            st[stmt[1]] = stmt[2] % _PROC_MOD
        elif head == "add":
            st[stmt[1]] = (st[stmt[1]] + stmt[2]) % _PROC_MOD
        elif head == "mul":
            st[stmt[1]] = (st[stmt[1]] * stmt[2]) % _PROC_MOD
        elif head == "repeat":
            for _ in range(stmt[1]):
                st = _proc_exec(stmt[2], st)
        elif head == "ifgt":
            branch = stmt[2] if st[stmt[1]] > stmt[3] else stmt[4]
            st = _proc_exec(branch, st)
    return st


def _proc_symbol(stmt: tuple) -> Term:
    head = stmt[0]
    if head in ("set", "add", "mul"):
        return Pred(head, Ident(stmt[1]), Num(stmt[2]))
    if head == "repeat":
        return Pred("repeat", Num(stmt[1]), Lst([_proc_symbol(s) for s in stmt[2]]))
    return Pred("if_greater", Ident(stmt[1]), Num(stmt[3]),
                Lst([_proc_symbol(s) for s in stmt[2]]),
                Lst([_proc_symbol(s) for s in stmt[4]]))


# `drop_first` rather than `drop`: the gloss table is keyed by head and read
# by every lesson at once, and `drop` is already a robot action meaning *put
# down* in conceptual_chunking. Glossing it "remove" for this DSL rewrote that
# lesson's action sequence too. The list operation gets its own name.
_DSL_OPS = ("add", "mul", "reverse", "sort", "take", "drop_first", "keep_greater", "keep_even")


_DSL_TEXT = {
    "add": "add {k} to each element",
    "mul": "multiply each element by {k}",
    "reverse": "reverse the list",
    "sort": "sort the list into increasing order",
    "take": "keep only the first {k} elements",
    "drop_first": "remove the first {k} elements",
    "keep_greater": "keep only the elements greater than {k}",
    "keep_even": "keep only the even elements",
}


_DSL_ARG = {"add": (1, 4), "mul": (2, 3), "take": (2, 3), "drop_first": (1, 2), "keep_greater": (2, 6)}


def _dsl_apply(op: tuple[str, int], xs: Sequence[int]) -> list[int]:
    name, k = op
    if name == "add":
        return [x + k for x in xs]
    if name == "mul":
        return [x * k for x in xs]
    if name == "reverse":
        return list(reversed(list(xs)))
    if name == "sort":
        return sorted(xs)
    if name == "take":
        return list(xs)[:k]
    if name == "drop_first":
        return list(xs)[k:]
    if name == "keep_greater":
        return [x for x in xs if x > k]
    return [x for x in xs if x % 2 == 0]


def _dsl_run(prog: Sequence[tuple[str, int]], xs: Sequence[int]) -> list[int]:
    out = list(xs)
    for op in prog:
        out = _dsl_apply(op, out)
    return out


def _dsl_op(rng: random.Random) -> tuple[str, int]:
    name = rng.choice(_DSL_OPS)
    lo_hi = _DSL_ARG.get(name)
    return (name, rng.randint(*lo_hi) if lo_hi else 0)


def _dsl_program(rng: random.Random) -> list[tuple[str, int]]:
    n = rng.randint(2, 3)
    prog: list[tuple[str, int]] = []
    while len(prog) < n:
        op = _dsl_op(rng)
        if not prog or op[0] != prog[-1][0]:          # two sorts in a row is not a program
            prog.append(op)
    return prog


def _dsl_mutate(rng: random.Random, prog: Sequence[tuple[str, int]]) -> list[tuple[str, int]]:
    """One operator or one bound changed — never a wholesale rewrite."""
    out = list(prog)
    i = rng.randrange(len(out))
    name, k = out[i]
    if name in _DSL_ARG and rng.random() < 0.55:
        lo, hi = _DSL_ARG[name]
        choices = [j for j in range(lo, hi + 2) if j != k]
        out[i] = (name, rng.choice(choices))
    else:
        out[i] = _dsl_op(rng)
    return out


def _dsl_symbol(prog: Sequence[tuple[str, int]], label: str | None = None) -> list[Term]:
    """The program as symbols, one term per step.

    The operation is the head of its own term rather than an ``Ident``
    argument, because that is what it is: a step applies an operation to a
    number. Written as an identifier it went out as one -- Polish read
    "krok p0: 0, keep_greater i 2", underscore and all -- since an ``Ident``
    is a name and names must pass through untouched or the nonce words this
    curriculum invents would be translated away.

    As a head it goes through the gloss table, which also fixes the sense.
    A head is looked up as a verb, and the verb is the reading these ops
    want: `keep` as a noun is a castle keep, and Spanish said *torreón*
    where it means *guardar*; `sort` as a noun is a kind or a type, and four
    languages said so.
    """
    if label is None:
        return [Pred("step", Num(i), Pred(op, Num(k))) for i, (op, k) in enumerate(prog)]
    return [Pred("step", Ident(label), Num(i), Pred(op, Num(k)))
            for i, (op, k) in enumerate(prog)]


def _dsl_text(prog: Sequence[tuple[str, int]]) -> str:
    """The English description, kept for the answer key and for telling two
    descriptions apart. What the reader sees is :func:`_dsl_desc`."""
    parts = [_DSL_TEXT[op].format(k=k) for op, k in prog]
    return "first " + ", then ".join(parts)


def _dsl_desc(prog: Sequence[tuple[str, int]]) -> Term:
    """The description as structure, so the grammar can say it.

    It was `Str("first keep only the elements greater than 2, then ...")`, and
    a `Str` is a literal: the whole lesson reached Polish, Finnish and four
    hundred others in English. Each step is now a construction the grammar
    builds -- "zachować liczby > 4", "luvut järjestää", "düzen aksi" -- with
    the verb where the language puts verbs.

    Numbered, because a program is ordered and a plain list joined with *and*
    would not say so.
    """
    return Lst([Pred("step", Num(i), Pred(f"desc_{op}", Num(k)))
                for i, (op, k) in enumerate(prog)])


def _dsl_lists(rng: random.Random, n: int) -> list[list[int]]:
    return [[rng.randint(0, 9) for _ in range(rng.randint(4, 5))] for _ in range(n)]
