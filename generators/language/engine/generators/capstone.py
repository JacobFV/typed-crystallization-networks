"""Support for the civilization-scale and capstone lessons.

Private. Ported alongside the lessons that use it; every function here is
called by at least one generator in :mod:`langcurriculum.lessons`.
"""

from __future__ import annotations

import random
from collections import deque
from typing import Any, Mapping, Sequence


_CONS = "kmtszlpvrdgbn"


_VOWS = "aeiou"


def _shuffled(rng: random.Random, xs: Sequence[Any]) -> list[Any]:
    ys = list(xs)
    rng.shuffle(ys)
    return ys


def _nonce(rng: random.Random, syllables: int = 2) -> str:
    return "".join(rng.choice(_CONS) + rng.choice(_VOWS) for _ in range(syllables))


def _nonce_pool(rng: random.Random, k: int, syllables: int = 2) -> list[str]:
    """``k`` distinct nonce words. Fresh every episode, so no token carries
    meaning between episodes and a constant guesser cannot exist."""
    out: list[str] = []
    seen: set[str] = set()
    guard = 0
    while len(out) < k and guard < 4000:
        guard += 1
        w = _nonce(rng, syllables)
        if w not in seen:
            seen.add(w)
            out.append(w)
    while len(out) < k:                                   # pragma: no cover
        out.append(f"w{len(out)}")
    return out


def _mode(xs: Sequence[Any]) -> tuple[Any, bool]:
    """(most common element, whether it is a strict plurality). Order-stable."""
    counts: dict[Any, int] = {}
    for x in xs:
        counts[x] = counts.get(x, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], str(kv[0])))
    top = ranked[0]
    strict = len(ranked) == 1 or ranked[1][1] < top[1]
    return top[0], strict


def _dedup(xs: Sequence[Any]) -> list[Any]:
    out: list[Any] = []
    seen: set[Any] = set()
    for x in xs:
        key = x if isinstance(x, (str, int, float, bool)) else str(x)
        if key not in seen:
            seen.add(key)
            out.append(x)
    return out


def _transmit(state: list[str], prestige: list[int], rule: str) -> list[str]:
    """One generation of cultural transmission on a ring, under a stated rule."""
    n = len(state)
    new: list[str] = []
    for i in range(n):
        nb = [(i - 1) % n, i, (i + 1) % n]
        if rule == "prestige":
            new.append(state[max(nb, key=lambda j: prestige[j])])
            continue
        counts: dict[str, int] = {}
        for j in nb:
            counts[state[j]] = counts.get(state[j], 0) + 1
        best = max(counts.values())
        tied = [w for w in _dedup([state[j] for j in nb]) if counts[w] == best]
        if len(tied) == 1:
            new.append(tied[0])
        else:                                   # prestige breaks a tie; prestige is distinct
            new.append(state[max((j for j in nb if state[j] in tied), key=lambda j: prestige[j])])
    return new


def _adopt(known: Sequence[str], current: str, theories: Sequence[str],
           preds: Mapping[str, Mapping[str, str]], results: Mapping[str, str]) -> str:
    """Adopt the best-supported theory over the evidence this lab has seen.

    Tiebreak is stated in the observation: keep what you hold if it is among the
    best, otherwise take the first-listed of the best. Stickiness is what makes
    the outcome depend on the *path* of evidence diffusion rather than on the
    final evidence set alone.
    """
    score = {t: sum(1 for e in known if preds[t][e] == results[e]) for t in theories}
    best = max(score.values()) if score else 0
    tied = [t for t in theories if score[t] == best]
    return current if current in tied else tied[0]


def _grammar(rng: random.Random, nts: Sequence[str], terms: Sequence[str],
             varnames: Sequence[str]) -> list[tuple[str, tuple[str, ...]]]:
    """A small rule system: ``head -> body`` over nonterminals, terminals and
    variables. Nothing about the properties below is imposed here; they are
    measured afterwards by the checkers."""
    rules: list[tuple[str, tuple[str, ...]]] = []
    for h in nts:
        for _ in range(2):
            n = rng.randint(1, 3)
            body = []
            for _ in range(n):
                r = rng.random()
                if r < 0.45:
                    body.append(rng.choice(terms))
                elif r < 0.85:
                    body.append(rng.choice(nts))
                else:
                    body.append(rng.choice(varnames))
            rules.append((h, tuple(body)))
    return rules


def _productive(rules, nts) -> bool:
    """Every nonterminal derives some finite string (no rule system that only
    loops), and none of them is unreachable from the start symbol."""
    prod: set[str] = set()
    for _ in range(len(nts) + 2):
        for h, body in rules:
            if all((s not in nts) or (s in prod) for s in body):
                prod.add(h)
    if set(nts) - prod:
        return False
    reach = {nts[0]}
    for _ in range(len(nts) + 2):
        for h, body in rules:
            if h in reach:
                reach |= {s for s in body if s in nts}
    return not (set(nts) - reach)


def _self_embedding(rules, nts) -> bool:
    """``A =>* u A v`` with u and v both non-empty: the exact criterion (Chomsky)
    for a context-free grammar to be genuinely non-regular, i.e. for recognizing
    its language to need an unbounded stack rather than finite state.

    Computed as reachability with two accumulated flags (something to the left,
    something to the right) over the nonterminal graph. No epsilon rules exist
    in the generated systems, so every symbol contributes a non-empty string.
    """
    for start in nts:
        seen = {(start, False, False)}
        q = deque([(start, False, False)])
        while q:
            a, l, r = q.popleft()
            if a == start and l and r:
                return True
            for h, body in rules:
                if h != a:
                    continue
                for i, s in enumerate(body):
                    if s not in nts:
                        continue
                    nxt = (s, l or i > 0, r or i < len(body) - 1)
                    if nxt not in seen:
                        seen.add(nxt)
                        q.append(nxt)
    return False


def _repeated_variable(rules, varnames) -> bool:
    """Some rule mentions the same variable twice: a rule that cannot be applied
    without *binding* the variable and checking the two occurrences agree."""
    for _h, body in rules:
        vs = [s for s in body if s in varnames]
        if len(vs) != len(set(vs)):
            return True
    return False


def _curriculum_valid(stages: Sequence[tuple[str, tuple[str, ...], str]],
                      have: Sequence[str], target: str) -> bool:
    """A curriculum works iff every stage's prerequisites are already held when
    it is reached, and the target capability is produced by the end. Pure
    prerequisite closure — exactly computable, and the whole point of ordering."""
    cur = set(have)
    for _name, req, prod in stages:
        if not set(req) <= cur:
            return False
        cur.add(prod)
    return target in cur


def _bfs_dist(delta: Mapping[tuple[int, int], int], n_states: int, n_acts: int,
              goal: int) -> list[int]:
    INF = 10 ** 6
    dist = [INF] * n_states
    dist[goal] = 0
    for _ in range(n_states):
        for s in range(n_states):
            for a in range(n_acts):
                t = delta[(s, a)]
                if dist[t] + 1 < dist[s]:
                    dist[s] = dist[t] + 1
    return dist


_OPEN_WORLD_NOTE = (
    "Not reducible to a one-step exactly-graded episode, and not faked. The "
    "lesson's content is a *loop*: discover an ontology, acquire the language "
    "that describes it, propose competing theories, DESIGN an experiment, run "
    "it, revise under criticism, and report with provenance. Everything that "
    "distinguishes it from lessons 150-169 lives in the parts a single "
    "question cannot contain: (1) the agent must choose interventions, so the "
    "environment has to accept experiment programs as actions and return "
    "outcomes, i.e. a multi-step interactive environment with a hidden generative world "
    "model rather than a generate()->answer pair; (2) the score is a "
    "trajectory functional — sample efficiency, whether each claim is "
    "supported by an experiment the agent itself ran, calibration of stated "
    "confidence, and whether a refuted theory is actually abandoned — none of "
    "which is a function of one answer; (3) grading requires an adversarial "
    "critic and a provenance ledger, both of which are additional environments. "
    "A real implementation needs: a parameterized hidden world (latent laws + "
    "nuisance parameters) with an exact simulator; an action language for "
    "experiments, assertions and retractions; an evaluator that checks each "
    "asserted law against the true one and each claim against the agent's own "
    "evidence ledger; and a scalar built from (discovery completeness x "
    "provenance validity x calibration) over the whole trajectory. Until that "
    "harness exists, any single-step version would grade a different, easier "
    "lesson while wearing this one's name."
)


#: the lessons the capstone section implements. ``symbolic_generalist`` mixes
#: two *other* lessons per episode and must exclude these, or it recurses.
_OWN_IDS = frozenset({
    "civilization_simulator", "scientific_civilization", "symbolic_world_builder",
    "curriculum_invention", "universal_interface_transfer", "unknown_game",
    "symbolic_generalist", "open_world_research_agent",
})
