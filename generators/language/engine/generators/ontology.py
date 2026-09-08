"""Support for the ontology and representation lessons.

Private. Ported alongside the lessons that use it; every function here is
called by at least one generator in :mod:`langcurriculum.lessons`.
"""

from __future__ import annotations

import random
from typing import Any, Iterable, Mapping, Sequence

from ..binding import SlotVocabulary
from .._structure import Ident, Lst, Pred, Term

OPTION_LABELS = ("opt_a", "opt_b", "opt_c", "opt_d")


ENCODINGS = ("graph", "sequence", "set", "table")


B_CONCEPT_NAMES = ("bex", "kirn", "zolt", "muva")


A_CONCEPT_NAMES = ("tavu", "nelo", "sogi", "wibra")


NEW_PRED_NAMES = ("p_kor", "p_vim", "p_sel", "p_dun")


MACRO_NAMES = ("m_kor", "m_vim", "m_sel", "m_dun")


PROPS = SlotVocabulary("property", ("warm", "striped", "hollow", "rigid", "porous", "glossy", "heavy", "fibrous"))


TYPE_NAMES = ("kes", "mor", "tav", "zun", "pil", "reth", "sam", "dov", "quen", "lorn")


ENTITIES = tuple(f"x{i}" for i in range(9))


ITEMS = tuple(f"i{i}" for i in range(8))


REL_NAMES = ("near", "holds", "links", "feeds", "binds")


OPS = SlotVocabulary("act", ("lift", "move", "drop", "turn", "scan", "wait", "push", "open"))


VERB_ISA = {"give": "transfer", "lend": "transfer", "sell": "transfer", "hand": "transfer",
            "push": "manipulate", "pull": "manipulate", "lift": "manipulate",
            "transfer": "action", "manipulate": "action"}


ENTITY_ISA = {"alice": "person", "bob": "person", "carol": "person", "dave": "person",
              "apple": "food", "bread": "food", "coin": "money", "gem": "money",
              "person": "agent", "food": "object", "money": "object",
              "agent": "thing", "object": "thing"}


PERSONS = ("alice", "bob", "carol", "dave")


THINGS = SlotVocabulary("thing", ("apple", "bread", "coin", "gem"))


REL_WORDS = {"left_of": "to the left of", "right_of": "to the right of",
             "above": "above", "below": "below", "near": "next to", "inside": "inside"}


def _shuffled(rng: random.Random, xs: Iterable[Any]) -> list[Any]:
    ys = list(xs)
    rng.shuffle(ys)
    return ys


def _label_options(rng: random.Random, correct: Any, distractors: Sequence[Any]):
    """Assign shuffled labels to candidates and shuffle their presentation order.

    Returns ``(pairs, vocabulary, answer)`` where ``pairs`` is
    ``[(label, candidate), ...]`` in presentation order. The correct candidate's
    label is drawn uniformly, so the answer is uniform over the label alphabet
    *and* its index in the shuffled vocabulary is uniform too.
    """
    cands = [correct, *distractors]
    labels = _shuffled(rng, OPTION_LABELS[: len(cands)])
    answer = labels[0]
    pairs = _shuffled(rng, list(zip(labels, cands)))
    return pairs, _shuffled(rng, labels), answer


def _inherited(own: Mapping[str, set], parent: Mapping[str, Any], t: str) -> set:
    """Own properties plus every ancestor's, following ``sub`` links."""
    out: set = set()
    seen: set = set()
    cur: Any = t
    while cur is not None and cur in own and cur not in seen:
        seen.add(cur)
        out |= set(own[cur])
        cur = parent.get(cur)
    return out


def _entity_props(own, parent, inst, e, exempt=(), extra=()) -> set:
    p = _inherited(own, parent, inst[e])
    p = p - {q for (x, q) in exempt if x == e}
    return p | {q for (x, q) in extra if x == e}


def _consistent(own, parent, inst, observed, exempt=(), extra=()) -> bool:
    """Exact: every entity's observed property set equals its type's."""
    if set(inst) != set(observed):
        return False
    for e in inst:
        if inst[e] not in own:
            return False
        if _entity_props(own, parent, inst, e, exempt, extra) != set(observed[e]):
            return False
    return True


def _onto_key(own, parent, inst) -> str:
    return repr((sorted((t, sorted(v)) for t, v in own.items()),
                 sorted(parent.items(), key=lambda kv: kv[0]),
                 sorted(inst.items())))


def _onto_facts(rng: random.Random, own, parent, inst) -> Term:
    fs: list[Term] = [Pred("type", Ident(t)) for t in sorted(own)]
    fs += [Pred("sub", Ident(t), Ident(p)) for t, p in sorted(parent.items()) if p is not None]
    fs += [Pred("prop", Ident(t), Ident(p)) for t in sorted(own) for p in sorted(own[t])]
    fs += [Pred("inst", Ident(e), Ident(t)) for e, t in sorted(inst.items())]
    return Lst(_shuffled(rng, fs))


def _random_hierarchy(rng: random.Random):
    """A small type tree with disjoint own-properties, plus its leaves."""
    props = rng.sample(list(PROPS), 5)
    types = rng.sample(list(TYPE_NAMES), 5)
    root = types[0]
    own = {root: {props[0]}}
    parent: dict[str, Any] = {root: None}
    if rng.random() < 0.5:
        own[types[1]] = {props[1]}
        own[types[2]] = {props[2]}
        parent[types[1]] = parent[types[2]] = root
        leaves = [types[1], types[2]]
    else:
        own[types[1]] = {props[1]}
        own[types[2]] = {props[2]}
        parent[types[1]] = parent[types[2]] = root
        own[types[3]] = {props[3]}
        own[types[4]] = {props[4]}
        parent[types[3]] = parent[types[4]] = types[1]
        leaves = [types[3], types[4], types[2]]
    return own, parent, leaves, props, types


def _assign_entities(rng: random.Random, leaves: Sequence[str], n_extra: int = 2):
    ents = rng.sample(list(ENTITIES), len(leaves) + n_extra)
    inst = {e: leaves[i] for i, e in enumerate(ents[: len(leaves)])}
    for e in ents[len(leaves):]:
        inst[e] = rng.choice(list(leaves))
    return inst


def _mutations(rng: random.Random, own, parent, inst):
    """Perturbations of a hierarchy; each is *checked* for inconsistency later."""
    out = []
    types = sorted(own)
    leaves = sorted({t for t in types if t not in set(parent.values())})
    fresh = [p for p in PROPS if all(p not in own[t] for t in types)]

    def copy():
        return ({t: set(v) for t, v in own.items()}, dict(parent), dict(inst))

    for t in types:                                  # add a property to a type
        for p in fresh[:2]:
            o, pa, ins = copy()
            o[t] = o[t] | {p}
            out.append((o, pa, ins))
    for t in types:                                  # drop a property from a type
        for p in sorted(own[t]):
            o, pa, ins = copy()
            o[t] = o[t] - {p}
            out.append((o, pa, ins))
    for t in types:                                  # push a property up to the root
        for p in sorted(own[t]):
            root = next(r for r in types if parent[r] is None)
            if t == root:
                continue
            o, pa, ins = copy()
            o[t] = o[t] - {p}
            o[root] = o[root] | {p}
            out.append((o, pa, ins))
    for a in leaves:                                 # swap two entities' types
        for b in leaves:
            if a >= b:
                continue
            ea = [e for e in inst if inst[e] == a]
            eb = [e for e in inst if inst[e] == b]
            if not ea or not eb:
                continue
            o, pa, ins = copy()
            ins[ea[0]], ins[eb[0]] = b, a
            out.append((o, pa, ins))
    for t in types:                                  # re-parent a type
        for u in types:
            if t == u or parent[t] is None or parent[t] == u or u == t:
                continue
            o, pa, ins = copy()
            pa[t] = u
            if _inherited(o, pa, t) == _inherited(own, parent, t):
                continue
            out.append((o, pa, ins))
    return out


def _apply_op(own, parent, inst, exempt, extra, op):
    """Apply a revision. Returns a fresh ontology (never mutates the input)."""
    own = {t: set(v) for t, v in own.items()}
    parent, inst = dict(parent), dict(inst)
    exempt, extra = set(exempt), set(extra)
    kind = op[0]
    if kind == "exempt":
        exempt.add((op[1], op[2]))
    elif kind == "attribute":
        extra.add((op[1], op[2]))
    elif kind == "generalize":
        if op[1] in own:
            own[op[1]] = own[op[1]] | {op[2]}
    elif kind == "reassign":
        if op[1] in inst:
            inst[op[1]] = op[2]
    elif kind == "merge":
        t1, t2 = op[1], op[2]
        if t2 in own and t1 in own and parent.get(t2) is not None:
            for e in list(inst):
                if inst[e] == t2:
                    inst[e] = t1
            for t in list(parent):
                if parent[t] == t2:
                    parent[t] = t1
            del own[t2]
            del parent[t2]
    elif kind == "split":
        t, p, t1, m1, t2, m2 = op[1], op[2], op[3], op[4], op[5], op[6]
        if t in own:
            own[t] = own[t] - {p}
            own[t1], own[t2] = {p}, set()
            parent[t1] = parent[t2] = t
            for e in m1:
                if e in inst:
                    inst[e] = t1
            for e in m2:
                if e in inst:
                    inst[e] = t2
    return own, parent, inst, exempt, extra


def _op_symbol(op) -> Term:
    if op[0] == "split":
        return Pred("split", Ident(op[1]), Ident(op[2]), Ident(op[3]),
                    Lst([Ident(e) for e in op[4]]), Ident(op[5]),
                    Lst([Ident(e) for e in op[6]]))
    return Pred(op[0], Ident(op[1]), Ident(op[2]))


def _revision_pool(own, parent, inst, observed, e_new, t_new, props_seen):
    """Every revision worth considering, as ``(kind, ...)`` tuples."""
    types = sorted(own)
    ops = []
    for p in sorted(props_seen):
        ops.append(("exempt", e_new, p))
        ops.append(("attribute", e_new, p))
    for t in types:
        if t != t_new:
            ops.append(("reassign", e_new, t))
        for p in sorted(props_seen):
            ops.append(("generalize", t, p))
    for a in types:
        for b in types:
            if a != b:
                ops.append(("merge", a, b))
    fresh = [t for t in TYPE_NAMES if t not in own][:2]
    if len(fresh) == 2:
        for t in types:
            members = sorted(e for e in inst if inst[e] == t)
            if len(members) < 2:
                continue
            for p in sorted(_inherited(own, parent, t)):
                have = [e for e in members if p in observed[e]]
                lack = [e for e in members if p not in observed[e]]
                ops.append(("split", t, p, fresh[0], tuple(have), fresh[1], tuple(lack)))
                ops.append(("split", t, p, fresh[0], tuple(lack), fresh[1], tuple(have)))
                ops.append(("split", t, p, fresh[0], tuple(members), fresh[1], ()))
    return ops


def _extension(entities, bits, spec) -> frozenset:
    """``spec`` is a tuple of (attribute index, required value)."""
    return frozenset(e for e, b in zip(entities, bits)
                     if all(b[i] == v for i, v in spec))


def _selection_costs(chain, edges, u, h, b, l, x, k):
    """Exact cost of answering the query under each of the four encodings."""
    m = len(edges)
    idx = chain.index(x)
    seq = u * (idx + 1) + h * k
    if k == 0:
        gpos = 1 + min(i for i, (a, c) in enumerate(edges) if a == x or c == x)
        return {"sequence": seq, "graph": u * gpos, "set": l, "table": b * m + l}
    used = [(chain[idx + j], chain[idx + j + 1]) for j in range(k)]
    graph = u * sum(1 + edges.index(e) for e in used)
    return {"sequence": seq, "graph": graph, "set": None, "table": b * m + l * k}


def _unique_argmin(costs):
    live = {k: v for k, v in costs.items() if v is not None}
    best = min(live.values())
    winners = [k for k, v in live.items() if v == best]
    return winners[0] if len(winners) == 1 else None


def _pattern_instances(facts, pat):
    """Ordered entity pairs matching ``(r1, r2, orientation)``."""
    r1, r2, orient = pat
    a = {(x, y) for (r, x, y) in facts if r == r1}
    out = []
    for (x, y) in sorted(a):
        need = (r2, x, y) if orient == "same" else (r2, y, x)
        if need in facts:
            out.append((x, y))
    return out


def _pattern_cover(facts, pat):
    r1, r2, orient = pat
    cov = set()
    for (x, y) in _pattern_instances(facts, pat):
        cov.add((r1, x, y))
        cov.add((r2, x, y) if orient == "same" else (r2, y, x))
    return cov


def _description_length(facts, pat):
    """Symbols to write the corpus down: ``1 + arity`` per atom, definition included."""
    inst = _pattern_instances(facts, pat)
    cov = _pattern_cover(facts, pat)
    definition = 3 + 2 * 3                           # head P(X,Y) plus a two-literal body
    return definition + 3 * (len(facts) - len(cov)) + 3 * len(inst)


def _ancestors(isa: Mapping[str, str], x: str) -> list[str]:
    out, cur, seen = [], isa.get(x), set()
    while cur is not None and cur not in seen:
        out.append(cur)
        seen.add(cur)
        cur = isa.get(cur)
    return out


def _slot_matches(isa, pat, val) -> bool:
    return pat == "_" or pat == val or pat in _ancestors(isa, val)


def _schema_covers(pat, ev) -> bool:
    isas = (VERB_ISA, ENTITY_ISA, ENTITY_ISA, ENTITY_ISA)
    return all(_slot_matches(i, p, v) for i, p, v in zip(isas, pat, ev))


def _more_specific(a, b) -> bool:
    """``a`` denotes a subset of ``b``: every slot of b covers a's slot."""
    isas = (VERB_ISA, ENTITY_ISA, ENTITY_ISA, ENTITY_ISA)
    if a == b:
        return False
    return all(pb == "_" or pa == pb or pb in _ancestors(i, pa)
               for i, pa, pb in zip(isas, a, b))


def _generalize_slot(isa, val):
    parent = isa.get(val)
    return parent if parent is not None else "_"


def _slot_members(isa, pat, domain):
    if pat == "_":
        return list(domain)
    return [d for d in domain if d == pat or pat in _ancestors(isa, d)]


def _occurrences(seq: Sequence[str], macro: Sequence[str]) -> int:
    """Greedy, non-overlapping, left to right — a deterministic count."""
    i = c = 0
    while i + len(macro) <= len(seq):
        if tuple(seq[i:i + len(macro)]) == tuple(macro):
            c += 1
            i += len(macro)
        else:
            i += 1
    return c


def _macro_saving(plans, macro) -> int:
    used = sum(_occurrences(p, macro) for p in plans)
    return used * (len(macro) - 1) - len(macro)


_LOCAL_GENERATORS: tuple = ()


_COMPOSING = False


def _english(world, struct) -> list[str]:
    """Realize a structure as English. The *only* source of the sentence."""
    def np(oid):
        o = world[oid]
        return ["the", o["color"], o["shape"]]

    head = struct.value[0] if isinstance(struct.value, tuple) else str(struct.value)
    if head == "and":
        a, b = struct.children
        left, right = _english(world, a), _english(world, b)
        return left + ["and"] + right[3:]            # share the subject NP
    if head == "not":
        inner = struct.children[0]
        rel = inner.value[0]
        x, y = [c.value for c in inner.children]
        return np(x) + ["is", "not"] + REL_WORDS[rel].split() + np(y)
    x, y = [c.value for c in struct.children]
    return np(x) + ["is"] + REL_WORDS[head].split() + np(y)


#: the lessons this module implements. ``general_language_agent`` composes a
#: *different* lesson each episode and must exclude these, or it recurses.
_OWN_IDS = frozenset({
    "general_language_agent", "natural_language_bridge", "ontology_construction",
    "ontology_revision", "ontology_alignment", "representation_selection",
    "representation_invention", "abstraction_ladder", "conceptual_chunking",
})

#: what ``general_language_agent`` falls back to when it cannot (or must not)
#: compose. Resolved through the registry at call time, because these lessons
#: live in modules that import this one.
_LOCAL_LESSON_IDS = ("ontology_construction", "ontology_revision", "ontology_alignment",
                     "representation_selection", "representation_invention",
                     "abstraction_ladder", "conceptual_chunking", "natural_language_bridge")

#: mutable because the re-entrancy guard has to be visible to the generator that
#: sets it and to any composer that reads it, and those are now separate modules.
_STATE = {"composing": False}


def _local_episode(rng: random.Random):
    """A non-composing episode from this section, for the fallback paths."""
    from ..registry import get

    lid = _LOCAL_LESSON_IDS[rng.randrange(len(_LOCAL_LESSON_IDS))]
    return get(lid).invoke(rng)
