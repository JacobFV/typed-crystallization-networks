"""Support for the epistemics, argument and problem-formulation lessons.

Private. Ported alongside the lessons that use it; every function here is
called by at least one generator in :mod:`langcurriculum.lessons`.
"""

from __future__ import annotations

import random
from typing import Any, Callable, Mapping, Sequence


_NONCE_LETTERS = "kmtszlpvrgnd"


SOURCES = ["orin", "vela", "kesh", "tamm", "ryn", "dola", "ferro", "idris"]


DOMAINS = ["geology", "medicine", "navigation", "metallurgy", "botany", "optics"]


OPTION_LABELS = ["option_a", "option_b", "option_c", "option_d"]


OPERATORS = ["search", "proof", "simulation", "memory_lookup", "experiment",
             "delegation", "act"]


def _shuffled(rng: random.Random, xs: Sequence[Any]) -> list[Any]:
    """A shuffled copy. Every answer vocabulary in this module goes through it."""
    ys = list(xs)
    rng.shuffle(ys)
    return ys


def _nonce(rng: random.Random, n: int = 3) -> str:
    return "".join(rng.choice(_NONCE_LETTERS) for _ in range(n))


def _nonces(rng: random.Random, k: int, n: int = 4, avoid: Sequence[str] = ()) -> list[str]:
    """``k`` distinct nonce words — every episode invents its own vocabulary."""
    out: list[str] = []
    seen = set(avoid)
    while len(out) < k:
        w = _nonce(rng, n)
        if w not in seen:
            seen.add(w)
            out.append(w)
    return out


def _labelled(rng: random.Random, options: Sequence[Any], correct_index: int) -> tuple[list[str], str]:
    """Attach the four option labels to options in a random order.

    Returns the label per option position and the label of the correct one, so
    the *name* of the answer is uniform even though the generator always builds
    the correct option first.
    """
    labels = _shuffled(rng, OPTION_LABELS[: len(options)])
    return labels, labels[correct_index]


def _provenance_status(reliable: set[str], unreliable: set[str],
                       reports: Sequence[tuple[str, str, str]], q: str) -> str:
    """What the *reports* establish about ``q`` — never what happens to be true."""
    r_yes = [s for s, p, pol in reports if p == q and pol == "yes" and s in reliable]
    r_no = [s for s, p, pol in reports if p == q and pol == "no" and s in reliable]
    u_any = [s for s, p, _ in reports if p == q and s in unreliable]
    if r_yes and r_no:
        return "contested"
    if r_yes:
        return "established"
    if r_no:
        return "refuted"
    if u_any:
        return "reported_only"
    return "unmentioned"


PROVENANCE_LABELS = ["established", "refuted", "contested", "reported_only", "unmentioned"]


def _grounded(nodes: Sequence[str], attacks: Sequence[tuple[str, str]]) -> tuple[set, set, set]:
    """The grounded extension: the least fixed point of "defended by".

    An argument is IN when every one of its attackers is OUT, OUT when some
    attacker is IN, and UNDECIDED otherwise — which is what an odd attack cycle
    produces, and is why "survives criticism" is a three-valued question.
    """
    incoming: dict[str, list[str]] = {n: [] for n in nodes}
    for a, b in attacks:
        incoming[b].append(a)
    inn: set[str] = set()
    out: set[str] = set()
    changed = True
    while changed:
        changed = False
        for n in nodes:
            if n in inn or n in out:
                continue
            if all(a in out for a in incoming[n]):
                inn.add(n)
                changed = True
        for n in nodes:
            if n in inn or n in out:
                continue
            if any(a in inn for a in incoming[n]):
                out.add(n)
                changed = True
    return inn, out, {n for n in nodes if n not in inn and n not in out}


ARG_LABELS = ["accepted", "rejected", "undecided"]


def _concept_graph(rng: random.Random, n: int = 8) -> tuple[list[str], dict[str, list[str]]]:
    """A prerequisite DAG; concepts are index-ordered, so index order is topological."""
    concepts = _nonces(rng, n, 4)
    prereqs: dict[str, list[str]] = {}
    for i, c in enumerate(concepts):
        k = min(i, rng.choice([0, 1, 1, 2]))
        prereqs[c] = sorted(rng.sample(concepts[:i], k)) if k else []
    return concepts, prereqs


def _need(c: str, known: set[str], prereqs: Mapping[str, list[str]]) -> set[str]:
    """Everything that must be introduced before ``c`` can be understood."""
    if c in known:
        return set()
    out = {c}
    for p in prereqs[c]:
        out |= _need(p, known, prereqs)
    return out


def _follow(expl: Sequence[str], known: set[str],
            prereqs: Mapping[str, list[str]]) -> tuple[set[str], list[tuple[str, list[str]]]]:
    """The listener takes the steps in order; a step whose prerequisites are not
    yet available is not understood and is *not* revisited."""
    derived = set(known)
    stuck: list[tuple[str, list[str]]] = []
    for c in expl:
        missing = [p for p in prereqs.get(c, ()) if p not in derived]
        if missing:
            stuck.append((c, missing))
        else:
            derived.add(c)
    return derived, stuck


def _closure(known: set[str], rules: Sequence[tuple[tuple[str, ...], str]]) -> set[str]:
    """Forward chaining to a fixed point: what the student can already derive."""
    have = set(known)
    changed = True
    while changed:
        changed = False
        for prem, concl in rules:
            if concl not in have and all(p in have for p in prem):
                have.add(concl)
                changed = True
    return have


GAP_LABELS = ["known", "inferable", "unknown", "malformed"]


ATTRS = ["weight", "value", "time"]


def _spec_accepts(items: Mapping[str, Mapping[str, int]], plan: Sequence[str],
                  spec: Sequence[tuple[str, str, int]]) -> bool:
    for sense, attr, k in spec:
        tot = sum(items[i][attr] for i in plan)
        if sense == "at_least" and tot < k:
            return False
        if sense == "at_most" and tot > k:
            return False
    return True


def _consistent(v: str, asg: Mapping[str, int], cons: Sequence[tuple[str, str, str]]) -> bool:
    for a, rel, b in cons:
        if a not in asg or b not in asg:
            continue
        if a != v and b != v:
            continue
        x, y = asg[a], asg[b]
        if rel == "!=" and x == y:
            return False
        if rel == "<" and not x < y:
            return False
        if rel == ">" and not x > y:
            return False
    return True


def _backtrack(order: Sequence[str], domains: Mapping[str, Sequence[int]],
               cons: Sequence[tuple[str, str, str]]) -> tuple[int, bool]:
    """Chronological backtracking, values ascending, stop at the first solution.
    Returns the number of assignments tried — the cost the reformulation pays."""
    nodes = 0

    def rec(i: int, asg: dict[str, int]) -> bool:
        nonlocal nodes
        if i == len(order):
            return True
        v = order[i]
        for val in sorted(domains[v]):
            nodes += 1
            asg[v] = val
            if _consistent(v, asg, cons) and rec(i + 1, asg):
                return True
            del asg[v]
        return False

    solved = rec(0, {})
    return nodes, solved


def _topo_ok(order: Sequence[str], edges: Sequence[tuple[str, str]]) -> bool:
    pos = {x: i for i, x in enumerate(order)}
    return all(pos[a] < pos[b] for a, b in edges)


def _run_method(method: Sequence[str], expansions: Mapping[str, list[str]],
                prims: Mapping[str, tuple[list[str], list[str]]],
                state0: Sequence[str], goal: str) -> bool:
    """Expand abstract subtasks to primitives and execute; a violated
    precondition aborts the whole plan, which is what makes ordering matter."""
    s = set(state0)
    for st in method:
        for p in expansions[st]:
            pre, add = prims[p]
            if not set(pre) <= s:
                return False
            s |= set(add)
    return goal in s


def _game_analysis(n: int, moves: Sequence[int]) -> list[list[int]]:
    """Normal-play take-away: winning moves at every position, by exact DP."""
    win = [False] * (n + 1)
    best: list[list[int]] = [[] for _ in range(n + 1)]
    for k in range(1, n + 1):
        wm = [m for m in moves if m <= k and not win[k - m]]
        win[k] = bool(wm)
        best[k] = wm
    return best


def _game_pools(moves: Sequence[int], hi: int, pass_token: str,
                naming: Mapping[int, str]) -> dict[str, list[int]]:
    """Positions with a *unique* winning move (or none at all), grouped by the
    answer they license — so the demonstrated answer is never ambiguous and the
    query position can be drawn to keep the answer distribution flat."""
    best = _game_analysis(hi, moves)
    pools: dict[str, list[int]] = {}
    for k in range(2, hi + 1):
        if len(best[k]) > 1:
            continue
        lab = naming[best[k][0]] if best[k] else pass_token
        pools.setdefault(lab, []).append(k)
    return pools


def _game_draw(rng: random.Random, pools: Mapping[str, list[int]],
               used: Sequence[int]) -> tuple[int, str]:
    """Draw a label uniformly first, then a position realizing it."""
    avail = sorted(lab for lab, ks in pools.items() if any(k not in used for k in ks))
    lab = rng.choice(avail)
    return rng.choice([k for k in pools[lab] if k not in used]), lab


def _running_sum(xs: Sequence[int]) -> list[int]:
    out, t = [], 0
    for x in xs:
        t += x
        out.append(t)
    return out


def _running_max(xs: Sequence[int]) -> list[int]:
    out, m = [], xs[0]
    for x in xs:
        m = max(m, x)
        out.append(m)
    return out


def _dedupe(xs: Sequence[int]) -> list[int]:
    out: list[int] = []
    for x in xs:
        if not out or out[-1] != x:
            out.append(x)
    return out


PROCEDURES: dict[str, Callable[[Sequence[int]], list[int]]] = {
    "sort_ascending": lambda xs: sorted(xs),
    "sort_descending": lambda xs: sorted(xs, reverse=True),
    "reverse": lambda xs: list(xs)[::-1],
    "rotate_left": lambda xs: list(xs)[1:] + list(xs)[:1],
    "running_sum": _running_sum,
    "running_max": _running_max,
    "double_each": lambda xs: [2 * x for x in xs],
    "drop_adjacent_repeats": _dedupe,
}


def _forward_scan(arr: Sequence[int], t: int) -> int:
    for i, v in enumerate(arr):
        if v == t:
            return i + 1
    return len(arr)


def _backward_scan(arr: Sequence[int], t: int) -> int:
    for k, i in enumerate(range(len(arr) - 1, -1, -1)):
        if arr[i] == t:
            return k + 1
    return len(arr)


def _binary_search(arr: Sequence[int], t: int) -> int:
    lo, hi, steps = 0, len(arr) - 1, 0
    while lo <= hi:
        mid = (lo + hi) // 2
        steps += 1
        if arr[mid] == t:
            return steps
        if arr[mid] < t:
            lo = mid + 1
        else:
            hi = mid - 1
    return steps


def _jump_search(arr: Sequence[int], t: int, b: int) -> int:
    n, steps, start = len(arr), 0, 0
    while start < n:
        end = min(start + b, n) - 1
        steps += 1
        if arr[end] >= t:
            for i in range(start, end + 1):
                steps += 1
                if arr[i] == t:
                    return steps
            return steps
        start += b
    return steps

