"""Exhaustive family decision for the all-Boolean scaffolds of §44/§46.

The tight scaffold is three BOOL nodes over six BOOL inputs on the complete
64-row truth table of `maj(a,b,c) xor maj(d,e,f)`, so a whole family fits in a
bitmask: one 64-bit integer per node value, one bitwise operation per candidate.

Semantics are read off `candidate.operator.name`, and a module candidate is
evaluated by executing the **module itself** through the registry over its
2^arity input combinations -- nothing about `MAJ3` or the distractor is
hard-coded here.  `validate()` checks the whole thing against `Program.execute`
on random members, per §45's discipline, and reports members checked and
mismatches.
"""
from __future__ import annotations
import itertools, random, sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
ROOT = os.path.dirname(os.path.dirname(HERE)); sys.path.insert(0, ROOT)
from tcn.types import BOOL, Value
from tcn.graph import Candidate

BITWISE = {
    'and': lambda f, a, b: a & b,
    'or': lambda f, a, b: a | b,
    'xor': lambda f, a, b: a ^ b,
    'nand': lambda f, a, b: ~(a & b) & f,
    'nor': lambda f, a, b: ~(a | b) & f,
    'xnor': lambda f, a, b: ~(a ^ b) & f,
    'eq': lambda f, a, b: ~(a ^ b) & f,
}
UNARY_BITWISE = {
    'not': lambda f, a: ~a & f,
    'identity': lambda f, a: a,
}


def input_masks(examples, names):
    """One 64-bit mask per input port over the example rows."""
    out = {}
    for k in names:
        m = 0
        for i, ex in enumerate(examples):
            if bool(ex['inputs'][k].decoded):
                m |= 1 << i
        out[k] = m
    return out


def label_mask(examples, target='out'):
    m = 0
    for i, ex in enumerate(examples):
        if bool(ex['targets'][target].decoded):
            m |= 1 << i
    return m


def module_table(registry, name, arity):
    """Truth table of a registered module, obtained by executing it."""
    m = registry.modules[name]
    keys = [k for k, _ in m.inputs]
    tab = {}
    for bits in itertools.product((False, True), repeat=arity):
        out, _ = m.run({k: Value.of(BOOL, b) for k, b in zip(keys, bits)}, registry=registry)
        tab[bits] = bool(list(out.values())[0].decoded)
    return tab


def _module_mask(tab, arity, srcs, full):
    res = 0
    for bits in itertools.product((False, True), repeat=arity):
        if not tab[bits]:
            continue
        sel = full
        for b, s in zip(bits, srcs):
            sel &= s if b else (~s & full)
        res |= sel
    return res


def compile_candidates(cands, registry, full, tables):
    """Return a list of (fn, source_names) closures, one per candidate."""
    out = []
    for c in cands:
        name = c.operator.name
        srcs = c.sources
        if len(srcs) == 2 and name in BITWISE:
            f = BITWISE[name]
            out.append((lambda pm, f=f, s=srcs: f(full, pm[s[0]], pm[s[1]]), srcs))
        elif len(srcs) == 1 and name in UNARY_BITWISE:
            f = UNARY_BITWISE[name]
            out.append((lambda pm, f=f, s=srcs: f(full, pm[s[0]]), srcs))
        else:
            arity = len(srcs)
            if name not in tables:
                tables[name] = module_table(registry, name, arity)
            tab = tables[name]
            out.append((lambda pm, t=tab, a=arity, s=srcs:
                        _module_mask(t, a, [pm[x] for x in s], full), srcs))
    return out


def sweep(program, registry, examples, override=None, target='out'):
    """Exhaust the family.  `override` replaces one node's candidate tuple.

    Enumeration order: nodes in program order, the first node the most
    significant digit.  Ties at the best training accuracy break by that order.
    """
    n = len(examples)
    full = (1 << n) - 1
    lab = label_mask(examples, target)
    names = [k for k, _ in program.inputs]
    base = input_masks(examples, names)
    tables = {}
    ov = override or {}
    nodes = list(program.nodes)
    compiled = [compile_candidates(ov.get(nd.name, nd.candidates), registry, full, tables)
                for nd in nodes]
    out_node = program.outputs[0][1]
    best = [-1, None, 0, 0, 0]      # acc, member, tied, conforming, members

    def rec(k, pm, path):
        if k == len(nodes):
            m = pm[out_node]
            a = ((~(m ^ lab)) & full).bit_count()
            best[4] += 1
            if a > best[0]:
                best[0], best[1], best[2] = a, tuple(path), 1
            elif a == best[0]:
                best[2] += 1
            best[3] += (a == n)
            return
        nd = nodes[k]
        for i, (fn, _) in enumerate(compiled[k]):
            pm[nd.name] = fn(pm)
            path.append(i)
            rec(k + 1, pm, path)
            path.pop()
        pm.pop(nd.name, None)

    rec(0, dict(base), [])
    return {'space_size': best[4], 'evaluated': best[4], 'exhausted': True,
            'certificate': 'complete',
            'best_train_accuracy': best[0] / n, 'best_train_hits': best[0],
            'members_tied_at_best_train': best[2],
            'conforming_on_train': best[3],
            'best_member': list(best[1]) if best[1] else None,
            'candidates_per_node': [len(c) for c in compiled]}


def selections(program, member):
    return {nd.name: member[i] for i, nd in enumerate(program.nodes)}


def validate(program, registry, examples, n=150, seed=0, override=None, target='out'):
    """Check the bitmask simulator against `Program.execute`, member by member."""
    ncase = len(examples)
    full = (1 << ncase) - 1
    names = [k for k, _ in program.inputs]
    base = input_masks(examples, names)
    tables = {}
    ov = override or {}
    nodes = list(program.nodes)
    compiled = [compile_candidates(ov.get(nd.name, nd.candidates), registry, full, tables)
                for nd in nodes]
    out_node = program.outputs[0][1]
    rnd = random.Random(seed)
    checked = mismatches = 0
    detail = []
    progs = program
    if ov:
        from tcn.graph import Node, Program
        progs = Program(program.inputs,
                        tuple(nd if nd.name not in ov else
                              Node(nd.name, nd.output, tuple(ov[nd.name]), nd.region, nd.depth)
                              for nd in program.nodes),
                        program.outputs, program.constants).validate(registry)
    for _ in range(n):
        member = [rnd.randrange(len(compiled[k])) for k in range(len(nodes))]
        pm = dict(base)
        for k, nd in enumerate(nodes):
            pm[nd.name] = compiled[k][member[k]][0](pm)
        sim = pm[out_node]
        sel = {nd.name: member[i] for i, nd in enumerate(progs.nodes)}
        real = 0
        for i, ex in enumerate(examples):
            try:
                _, _, trace = progs.execute(ex['inputs'], registry=registry, selections=sel)
                v = bool(round(trace[out_node].flat()[0]))
            except Exception:
                v = None
            if v is None:
                real = None
                break
            if v:
                real |= 1 << i
        checked += 1
        if sim != real:
            mismatches += 1
            if len(detail) < 5:
                detail.append({'member': member, 'simulator': sim, 'real_program': real})
    return {'members_checked': checked, 'mismatches': mismatches,
            'examples': ncase, 'episode_evaluations_checked': checked * ncase,
            'compared_via': 'Program.execute, per example', 'mismatch_detail': detail}


def extend_site(program, registry, site, op_name):
    """Every wiring of `op_name` over the site's own source pool.

    The pool is read off the node's existing candidates, so nothing is
    hand-wired; the operator is resolved through the registry, so nothing is
    added to core.
    """
    nd = next(x for x in program.nodes if x.name == site)
    pool = []
    for c in nd.candidates:
        for s in c.sources:
            if s not in pool:
                pool.append(s)
    types = program.port_types()
    for arity in (2, 1):
        try:
            op = registry.resolve(op_name, tuple([types[pool[0]]] * arity), nd.output, None)
        except Exception:
            continue
        return [Candidate(op, tuple(t)) for t in itertools.product(pool, repeat=arity)]
    return []
