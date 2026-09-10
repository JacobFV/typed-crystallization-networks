"""Everything the objectives need beyond the pool itself.

Two of the seven objectives need more than the instrumented pass provides:

* **O4** needs, for each task `u` of the mining corpus, a pool mined from the
  corpus *minus* `u`, and each of its classes scored on `u`'s entries -- which
  were withheld from that pool's derivation and from its scoring.  `cv_folds`
  builds exactly that, by the same rule, with the same parameters.
* **O5** needs a *measured* execution cost, not `Program.execution_cost` (§41
  measured that as constant, 148.0, across programs differing by 48 executed
  bytecodes).  `o5_scores` uses `research/lazy-guard/cost.py`'s `NativeEval`
  meter -- §51's instrument -- to count executed primitive `tcn` leaf operations
  exhaustively over the complete 16-row Boolean domain, so the figure is an
  exact total and not a sample.
"""
from __future__ import annotations

import itertools

import _paths  # noqa: F401

from tcn.operators import Registry
from tcn.types import BOOL, Value

import mine
import mine_semantic
from mine_multi import MAX_HOLES, MAX_NODES, _conforms  # noqa: F401

import cost as lazy_cost            # research/lazy-guard/cost.py, §51's instrument
import evaltasks
import family
import pool

DOMAIN = list(itertools.product((False, True), repeat=4))


# ------------------------------------------------------------------ O4 folds

def saving_on_entries(canon, key, programs, examples_by_entry,
                      max_nodes=MAX_NODES, max_holes=MAX_HOLES):
    """`{entry: s_e}` for rewriting each program with `canon`, verified.

    Same rewriting and same verification as the rule: disjoint occurrences in
    root order, one module call per site, and the rewritten program executed
    against the task's own rows before its saving is counted.
    """
    registry = Registry()
    name = registry.register_module(canon)
    if name != "module:" + canon.digest:
        return None
    cache, out = {}, {}
    for entry, program in programs.items():
        occs = []
        for root, S, cnd, holes in mine.fragments(program, max_nodes, max_holes):
            if (len(cnd.inputs), mine_semantic.truth_table(cnd, cache)) == key:
                occs.append({"root": root, "nodes": set(S), "holes": holes})
        if not occs:
            continue
        chosen = mine.disjoint_occurrences(sorted(occs, key=lambda o: o["root"]))
        try:
            rw = mine.rewrite(program, chosen, name, registry)
        except (TypeError, ValueError, KeyError):
            continue
        if not _conforms(rw, examples_by_entry[entry], registry):
            continue
        out[entry] = program.description_bits() - rw.description_bits()
    return out


def cv_folds(family_name, variant, exclude, corpus, task_of, params=None):
    """`{held-out task u: {pooling key: (saving on u's entries, D)}}`.

    For each task `u` of the mining corpus the pool is re-mined on the corpus
    minus `u` under the identical rule; each surviving class is then scored on
    `u`'s entries with its *own* elected representative.
    """
    fns = dict(family.FAMILIES[family_name])
    tasks = sorted(set(task_of.values()))
    folds = {}
    for u in tasks:
        sub = pool.build(family_name, variant, tuple(sorted(set(exclude) | {u})),
                         **(params or {}))[0]
        held = {e: p for e, p in corpus.items() if task_of[e] == u}
        ex = {e: evaltasks.examples_for(fns[u]) for e in held}
        fold = {}
        for k in sub:
            sav = saving_on_entries(k.canonical, k.key, held, ex,
                                    (params or {}).get('max_nodes', MAX_NODES),
                                    (params or {}).get('max_holes', MAX_HOLES))
            if sav is None:
                continue
            fold[k.key] = (sum(sav.values()), k.definition_bits)
        folds[u] = fold
    return folds


# ------------------------------------------------------- O5 measured cost

def _selected(program):
    """A copy with `selected = 0` everywhere, so the native evaluator can run it.

    Corpus programs and rewrites carry exactly one candidate per node, so this
    fixes no choice that was ever open.
    """
    if all(n.selected is not None for n in program.nodes):
        return program
    return program.harden([0] * len(program.nodes))


def _ops(program, registry, domain=None, inputs=None):
    """Executed primitive leaf operations, exhaustive over the domain."""
    ev = lazy_cost.NativeEval(registry)
    keys = [k for k, _ in program.inputs] if inputs is None else inputs
    dom = domain or list(itertools.product((False, True), repeat=len(keys)))
    total = 0
    for bits in dom:
        ev.meter.reset()
        ev.run(_selected(program), dict(zip(keys, bits)))
        total += ev.meter.n
    return total, len(dom)


def _bytecodes(program, registry, domain=None, inputs=None):
    keys = [k for k, _ in program.inputs] if inputs is None else inputs
    dom = domain or list(itertools.product((False, True), repeat=len(keys)))
    p = _selected(program)
    ev = lazy_cost.NativeEval(registry)

    def run():
        for bits in dom:
            ev.run(p, dict(zip(keys, bits)))
    return lazy_cost.count_opcodes(run)


def o5_scores(classes, corpus, want_bytecodes=True):
    """`{key: score}` where score = ops saved over the corpus, definition once.

    `X(p)` is the total executed primitive operations of `p` over its complete
    input domain -- 16 rows for a corpus program, `2**arity` for a definition.
    Rewriting a call site does not remove the leaves it calls, so this objective
    is expected to carry little signal; that is a measurement, not a defect of
    the arm, and it is reported either way.
    """
    r0 = Registry()
    flat = {e: _ops(p, r0)[0] for e, p in corpus.items()}
    flat_total = sum(flat.values())
    scores, detail = {}, {}
    for k in classes:
        registry = Registry()
        name = registry.register_module(k.canonical)
        assert name == "module:" + k.canonical.digest
        rw_total = 0
        for e, p in corpus.items():
            rw = k.rewritten.get(e)
            rw_total += flat[e] if rw is None else _ops(rw, registry)[0]
        d_ops, _ = _ops(k.canonical, Registry())
        scores[k.key] = float(flat_total - rw_total - d_ops)
        row = {"digest": k.digest, "flat_ops": flat_total, "rewritten_ops": rw_total,
               "definition_ops": d_ops, "score": scores[k.key]}
        if want_bytecodes:
            rb = 0
            for e, p in corpus.items():
                rw = k.rewritten.get(e)
                rb += _bytecodes(p if rw is None else rw,
                                 r0 if rw is None else registry)
            row["rewritten_bytecodes"] = rb
        detail[k.digest] = row
    if want_bytecodes:
        fb = sum(_bytecodes(p, r0) for p in corpus.values())
        for row in detail.values():
            row["flat_bytecodes"] = fb
            row["bytecode_score"] = float(fb - row["rewritten_bytecodes"])
    return scores, detail


# ------------------------------------------------------------------- assemble

def build_context(family_name, variant, exclude, classes, corpus, task_of,
                  need=("O3", "O4", "O5"), params=None):
    entries_of_task = {}
    for e, t in task_of.items():
        entries_of_task.setdefault(t, []).append(e)
    ctx = {"family": family_name,
           "tasks_in_corpus": sorted(entries_of_task),
           "entries_of_task": entries_of_task}
    ctx["cv"] = (cv_folds(family_name, variant, exclude, corpus, task_of, params)
                 if "O4" in need else {})
    if "O5" in need:
        ctx["o5_score"], ctx["o5_detail"] = o5_scores(classes, corpus)
    else:
        ctx["o5_score"], ctx["o5_detail"] = {}, {}
    return ctx


def fixture_for(canon):
    return [{k: Value.of(BOOL, v) for (k, _), v in zip(canon.inputs, bits)}
            for bits in itertools.product((False, True), repeat=len(canon.inputs))]
