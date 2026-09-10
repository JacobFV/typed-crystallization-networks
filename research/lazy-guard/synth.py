"""Oracle-guided resynthesis: enumerative proposal, exhaustive equivalence.

DESIGN sec 2 recommends B + D -- two languages, with enumerative/CEGIS synthesis
in the algorithm IR and the frozen program as the oracle -- and sec 4 says the
equivalence backend should be exhaustive execution first.  That is what this is.
No e-graph, no Lean, no rule set that names the answer.

Two proposal engines, both generic over the grammar they are handed:

  `rewrite_search`   bottom-up typed enumeration of a *replacement* for one
                     binding of the transcribed specification, chosen by
                     expected cost among the exact candidates, iterated to a
                     fixed point.  Nothing about `mux` or `guard` is special
                     cased: the enumerator produces every well-typed expression
                     of the allowed constructs up to the size bound.

  `toplevel_search`  bottom-up typed enumeration of a *whole output expression*
                     over components obtained by anti-unifying the
                     specification's repeated sub-DAGs.  The anti-unifier is
                     generic (DESIGN sec 6 items 2 and 3): it groups structurally
                     isomorphic sub-DAGs and abstracts the positions where they
                     differ.  It is not told that the miniature has positions.

`ABLATABLE` names the constructs a run may be denied, which is what criterion 3
is checked with.
"""
from __future__ import annotations

import itertools
import json
import pathlib
import random
import sys

HERE = pathlib.Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cost as _cost
import ir
from tcn.types import BOOL, Type, decode

ABLATABLE = ("Guard", "Find", "Scan", "Let")


def tkey(t):
    return json.dumps(t.to_dict(), sort_keys=True, separators=(",", ":"))


_WIDE = ("identity", "not", "and", "or", "xor", "nand", "nor", "xnor",
         "eq", "lt", "le", "gt", "ge", "add", "sub", "mul", "min", "max",
         "mod", "idiv", "mux", "insert", "remove", "member", "union",
         "intersection", "count", "index", "tuple")


def wide_vocabulary(registry, types, spec_ops=()):
    """Every universal operator the available types admit, plus the modules.

    This is deliberately *wider* than the specification's own operator multiset:
    a search that can only reuse the operators the specification happens to
    contain has been given a hint.  Nothing domain specific enters here; the
    signatures come from `Registry.resolve` alone.
    """
    ts = list({tkey(t): t for t in types}.values())
    out = {}
    for op in spec_ops:
        out[(op.name, tkey(op.output), tuple(tkey(t) for t in op.inputs), op.parameters)] = op
    names = list(_WIDE) + [m for m in registry.modules]
    for name in names:
        for arity in (1, 2, 3):
            for combo in itertools.product(ts, repeat=arity):
                try:
                    op = registry.resolve(name, combo)
                except Exception:
                    continue
                out[(op.name, tkey(op.output), tuple(tkey(t) for t in op.inputs), op.parameters)] = op
    return list(out.values())


# ---------------------------------------------------------------------------
# typed bottom-up enumeration
# ---------------------------------------------------------------------------
class Grammar:
    """Typed terminals + operator signatures + which IR constructs are allowed."""

    def __init__(self, registry, terminals, operators, constructs=ABLATABLE):
        self.registry = registry
        self.terminals = list(terminals)          # [(expr, Type)]
        self.operators = list(operators)          # [Operator]  (leaf or module)
        self.constructs = set(constructs)

    def enumerate(self, target, max_size, extra=()):
        """Every well-typed expression of type `target` with size <= max_size.

        Bottom-up by size, so a size-k expression is built only from parts
        already enumerated.  Deterministic order; no heuristic ranking.
        """
        pool = {}                                  # size -> {tkey: [expr]}
        allt = list(self.terminals) + list(extra)
        pool[1] = {}
        for e, t in allt:
            pool[1].setdefault(tkey(t), []).append(e)
        types = {tkey(t): t for _, t in allt}
        for op in self.operators:
            types.setdefault(tkey(op.output), op.output)
            for t in op.inputs:
                types.setdefault(tkey(t), t)
        for size in range(2, max_size + 1):
            cur = {}
            for op in self.operators:
                need = len(op.inputs)
                for parts in self._splits(pool, op.inputs, size - 1):
                    e = (ir.Call(op.name, tuple(parts)) if op.name.startswith("module:")
                         else ir.Prim(op, tuple(parts)))
                    cur.setdefault(tkey(op.output), []).append(e)
            if "Guard" in self.constructs:
                bools = self._upto(pool, BOOL, size - 1)
                for tk, t in types.items():
                    for c in bools:
                        rest = size - 1 - ir._size(c)
                        for a, b in self._pairs(pool, t, t, rest):
                            cur.setdefault(tk, []).append(ir.Guard(c, a, b))
            pool[size] = cur
        out = []
        for size in range(1, max_size + 1):
            out += pool.get(size, {}).get(tkey(target), [])
        return out

    def _upto(self, pool, t, budget):
        k = tkey(t)
        return [e for s in range(1, budget + 1) for e in pool.get(s, {}).get(k, [])]

    def _pairs(self, pool, ta, tb, budget):
        for sa in range(1, budget):
            for a in pool.get(sa, {}).get(tkey(ta), []):
                for sb in range(1, budget - sa + 1):
                    for b in pool.get(sb, {}).get(tkey(tb), []):
                        yield a, b

    def _splits(self, pool, types, budget):
        if not types:
            if budget == 0:
                yield []
            return
        head, rest = types[0], types[1:]
        for s in range(1, budget + 1):
            for e in pool.get(s, {}).get(tkey(head), []):
                for tail in self._splits(pool, rest, budget - s):
                    yield [e] + tail


# ---------------------------------------------------------------------------
# the oracle
# ---------------------------------------------------------------------------
class Oracle:
    """The frozen program: unlimited labelled I/O and a counterexample check."""

    def __init__(self, fx):
        self.fx = fx
        self.program = fx["program"]
        self.registry = fx["registry"]
        self.port = fx["port"]
        self.meter = _cost.Meter()
        self.ev = _cost.NativeEval(fx["registry"], self.meter)
        self._memo = {}

    def __call__(self, case):
        key = case[self.port]
        v = self._memo.get(key)
        if v is None:
            v = self.ev.run(self.program, {self.port: key})[0]
            v = next(iter(v.values()))
            self._memo[key] = v
        return v


def check(evaluator, prog, oracle, cases, port):
    """First disagreement, or None."""
    for c in cases:
        try:
            got = evaluator.run(prog, {port: c[port]})
        except Exception:
            return c
        if got != oracle(c):
            return c
    return None


def verify_exhaustive(evaluator, prog, oracle, all_cases, port):
    """The equivalence backend: execution over **every** input of the domain.

    DESIGN sec 4: exhaustion first, SMT second, Lean only where parametricity
    forces it.  The carrier here is finite and small, so this is both cheaper
    than proving and produces the certificate this repository already uses.
    Returns `(ok, counterexample, n_checked)`.
    """
    n = 0
    for c in all_cases:
        try:
            got = evaluator.run(prog, {port: c[port]})
        except Exception:
            return False, c, n
        if got != oracle(c):
            return False, c, n
        n += 1
    return True, None, n


def expected_ops(evaluator, prog, cases, port):
    tot = 0
    for c in cases:
        evaluator.meter.reset()
        evaluator.run(prog, {port: c[port]})
        tot += evaluator.meter.n
    return tot / len(cases)


# ---------------------------------------------------------------------------
# STAGE 1 engine: replacement enumeration to a fixed point
# ---------------------------------------------------------------------------
def rewrite_search(fx, grammar_constructs=ABLATABLE, max_size=4, seed=0,
                   sample=192, domain=None, verbose=True, seed_candidates=(),
                   verify_sample=None):
    """Greedy CEGIS over per-binding replacements.  Returns the final program.

    `verify_sample` limits the *search-time* equivalence check to that many
    inputs; the final certificate is always taken over the whole domain by the
    caller.  It is only ever used where the domain is too large to exhaust once
    per accepted rewrite.
    """
    p, r = fx["program"], fx["registry"]
    port = fx["port"]
    types = p.port_types()
    prog = ir.from_program(p, r)
    oracle = Oracle(fx)
    ev = ir.Evaluator(r)

    rng = random.Random(seed)
    full = list(domain(fx))
    cases = rng.sample(full, min(sample, len(full)))
    all_cases = full if verify_sample is None else rng.sample(full, min(verify_sample, len(full)))

    spec_ops = [node.candidates[node.selected].operator for node in p.nodes]
    tof0 = p.port_types()
    ops = wide_vocabulary(r, list(tof0.values()), spec_ops)
    if verbose:
        print("   operator vocabulary: %d signatures" % len(ops))

    names = [k for k, _ in p.constants] + [n.name for n in p.nodes]
    order = {k: i for i, k in enumerate(names)}
    const_types = {k: v.type for k, v in p.constants}
    tof = dict(const_types)
    for n in p.nodes:
        tof[n.name] = n.output

    trace = []
    base_cost = expected_ops(ev, prog, cases, port)
    considered = 0
    counterexamples = []
    rounds = 0
    while True:
        rounds += 1
        ranked = []
        for target_name in reversed(names):
            if target_name in const_types:
                continue
            avail = [(ir.Ref(k), tof[k]) for k in names if order[k] < order[target_name]]
            g = Grammar(r, avail, ops, grammar_constructs)
            cands = list(g.enumerate(tof[target_name], max_size)) + list(seed_candidates)
            considered += len(cands)
            for cand in cands:
                trial = prog.rebind(target_name, cand)
                try:
                    if check(ev, trial, oracle, cases, port) is not None:
                        continue
                    c = expected_ops(ev, trial, cases, port)
                except (ValueError, RecursionError, KeyError):
                    continue
                if c < base_cost - 1e-9:
                    ranked.append((c, target_name, cand))
        ranked.sort(key=lambda z: z[0])
        accepted = False
        for c, nm, cand in ranked:
            trial = prog.rebind(nm, cand)
            ok, bad, _ = verify_exhaustive(ev, trial, oracle, all_cases, port)
            if not ok:
                # CEGIS: the counterexample enters the example set permanently,
                # so every later proposal is filtered by it too.
                if bad is not None and bad not in cases:
                    cases.append(bad)
                    counterexamples.append(bad[port])
                continue
            prog = trial
            base_cost = expected_ops(ev, prog, cases, port)
            accepted = True
            trace.append({"binding": nm, "replacement": repr(cand),
                          "sampled_expected_ops": base_cost,
                          "verified_exhaustively": len(all_cases)})
            if verbose:
                print("   rewrote %-4s := %-58s sampled E[ops] %.3f (exhaustively verified)"
                      % (nm, repr(cand)[:58], base_cost))
            break
        if not accepted:
            break
    return {"program": prog, "trace": trace, "candidates_considered": considered,
            "sample_cases": len(cases), "rounds": rounds,
            "counterexamples_found": len(counterexamples),
            "counterexamples": counterexamples[:20]}


# ---------------------------------------------------------------------------
# anti-unification: discover the parametric family
# ---------------------------------------------------------------------------
def _shape(name, table, consts, depth=0):
    """A structural key for the sub-DAG rooted at `name`, plus its holes.

    Two kinds of hole: an operator's `parameters` dictionary, and a leaf that is
    a *constant*.  Both become anonymous in the key, so two members of an
    unrolled family map to the same key and their differences show up as holes.
    Nothing here mentions positions, indices, spans or either miniature.
    """
    if name not in table:
        if name in consts:
            return ("const",), [("const", name, consts[name])]
        return ("leaf", name), []
    op, srcs = table[name]
    holes = []
    parts = []
    if op.parameters:
        holes.append(("param", name, dict(op.parameters)))
    for s in srcs:
        sub, h = _shape(s, table, consts, depth + 1)
        parts.append(sub)
        holes += h
    return (op.name, tkey(op.output), tuple(parts)), holes


def _slot_values(hole):
    kind, _, payload = hole
    if kind == "const":
        return payload if isinstance(payload, int) and not isinstance(payload, bool) else None
    if len(payload) == 1 and "index" in payload:
        return payload["index"]
    return None


def antiunify(fx, max_arity=None):
    """Group isomorphic sub-DAGs and abstract the positions where they differ.

    DESIGN sec 6 lists port/constant parameterization (item 2) and
    anti-unification (item 3) as the successors to `mine.py`'s syntactic digest.
    This is both, in their smallest useful form:

      * members of a group must differ in slots that all carry the *same*
        integer sequence, and that sequence must be a contiguous run `a, a+1,
        ..., a+k-1` in some order.  Then every varying slot is bound to one loop
        variable and the run gives the loop's bounds.
      * a varying `project` index generalizes to the dynamic `index` operator,
        which the algebra already has, and only when the tuple is homogeneous;
        a varying constant leaf generalizes to the loop variable directly.

    Returns `[{'lambda', 'type', 'lo', 'hi', 'instances', 'arity',
    'abstracted'}]`.  The lambda's free variable is `Var("i")`.
    """
    p, r = fx["program"], fx["registry"]
    const_val = {k: decode(v.type, v.raw) for k, v in p.constants}
    table = {}
    for n in p.nodes:
        c = n.candidates[n.selected]
        table[n.name] = (c.operator, c.sources)
    tof = p.port_types()

    idx_t = None
    for k, v in p.constants:
        if v.type.kind == "int" and v.type.encoding.kind == "integer":
            idx_t = v.type
            break

    groups = {}
    for n in p.nodes:
        key, holes = _shape(n.name, table, const_val)
        groups.setdefault(key, []).append((n.name, holes))

    comps = []
    for key, members in groups.items():
        if len(members) < 2 or not members[0][1]:
            continue
        if max_arity is not None and len(members) > max_arity:
            continue
        slots = list(zip(*[h for _, h in members]))
        seqs = {}
        for i, s in enumerate(slots):
            vals = [_slot_values(x) for x in s]
            if any(v is None for v in vals):
                if len({json.dumps(x[2], sort_keys=True, default=str) for x in s}) > 1:
                    seqs = None
                    break
                continue
            if len(set(vals)) > 1:
                seqs.setdefault(tuple(vals), []).append(i)
        if seqs is None or len(seqs) != 1:
            continue
        seq, varying = next(iter(seqs.items()))
        lo = min(seq)
        if sorted(seq) != list(range(lo, lo + len(seq))):
            continue
        varying = set(varying)
        if idx_t is None:
            continue

        order = sorted(range(len(members)), key=lambda j: seq[j])
        first = members[order[0]][0]
        counter = [0]
        failed = []

        def walk(nm):
            if nm not in table:
                if nm in const_val:
                    k = counter[0]
                    counter[0] += 1
                    return ir.Var("i") if k in varying else ir.Ref(nm)
                return ir.Ref(nm)
            op, srcs = table[nm]
            if op.parameters:
                k = counter[0]
                counter[0] += 1
                if k in varying:
                    src_t = tof[srcs[0]]
                    if op.name != "project" or src_t.kind != "tuple" or len(set(src_t.items)) != 1:
                        failed.append(nm)
                        return ir.Ref(nm)
                    try:
                        idx_op = r.resolve("index", (src_t, idx_t), src_t.items[0])
                    except Exception:
                        failed.append(nm)
                        return ir.Ref(nm)
                    return ir.Prim(idx_op, (walk(srcs[0]), ir.Var("i")))
            args = tuple(walk(s) for s in srcs)
            return ir.Call(op.name, args) if op.name.startswith("module:") else ir.Prim(op, args)

        lam = walk(first)
        if failed:
            continue
        kinds = sorted({slots[i][0][0] for i in varying})
        comps.append({"lambda": lam, "type": tof[first], "lo": lo, "hi": lo + len(seq),
                      "instances": [members[j][0] for j in order], "arity": len(members),
                      "abstracted": "+".join(kinds) + " family %d..%d -> loop variable"
                                    % (lo, lo + len(seq) - 1)})
        # the loop variable itself is a component of the same family
        vt = None
        for i in varying:
            kind, nm2, _ = slots[i][0]
            if kind == "const":
                vt = tof[nm2]
                break
        if vt is not None:
            comps.append({"lambda": ir.Var("i"), "type": vt, "lo": lo, "hi": lo + len(seq),
                          "instances": [], "arity": len(members),
                          "abstracted": "the loop variable of the %d..%d family"
                                        % (lo, lo + len(seq) - 1)})

    # a family of constants forming a contiguous run is the loop variable itself
    byt = {}
    for k, v in const_val.items():
        if isinstance(v, int) and not isinstance(v, bool):
            byt.setdefault(tkey(tof[k]), []).append((v, k))
    for tk, items in byt.items():
        items.sort()
        seq = [v for v, _ in items]
        if len(seq) < 2 or seq != sorted(set(seq)) or seq != list(range(seq[0], seq[0] + len(seq))):
            continue
        comps.append({"lambda": ir.Var("i"), "type": tof[items[0][1]],
                      "lo": seq[0], "hi": seq[-1] + 1,
                      "instances": [k for _, k in items], "arity": len(items),
                      "abstracted": "const family %d..%d -> loop variable" % (seq[0], seq[-1])})
    return comps


# ---------------------------------------------------------------------------
# STAGE 2 engine: whole-output enumeration over anti-unified components
# ---------------------------------------------------------------------------
def toplevel_search(fx, constructs=ABLATABLE, body_size=3, tail_size=2, seed=0,
                    sample=192, domain=None, verbose=True, extra_components=(),
                    seed_candidates=()):
    p, r = fx["program"], fx["registry"]
    port = fx["port"]
    out_type = p.port_types()[p.outputs[0][1]]
    oracle = Oracle(fx)
    ev = ir.Evaluator(r)
    rng = random.Random(seed)
    all_cases = list(domain(fx))
    cases = rng.sample(all_cases, min(sample, len(all_cases)))

    comps = antiunify(fx) + list(extra_components)
    if verbose:
        for c in comps:
            print("   anti-unified x%d : %r   [%s]" % (c["arity"], c["lambda"], c["abstracted"]))

    spec_ops = [node.candidates[node.selected].operator for node in p.nodes]
    const_val = {k: decode(v.type, v.raw) for k, v in p.constants}
    const_t = {k: v.type for k, v in p.constants}
    present = list(const_t.values()) + [c["type"] for c in comps] + [out_type] + \
        [t for _, t in p.inputs] + [fx["carrier"]]
    ops = wide_vocabulary(r, present, spec_ops)
    if verbose:
        print("   operator vocabulary: %d signatures over %d types"
              % (len(ops), len({tkey(t) for t in present})))

    # terminals visible inside a loop body: the loop variable and the components
    inner = [(c["lambda"], c["type"]) for c in comps]
    inner += [(ir.Lit(v), const_t[k]) for k, v in const_val.items()]
    g_in = Grammar(r, inner, ops, constructs)
    preds = g_in.enumerate(BOOL, body_size)
    tails = g_in.enumerate(out_type, tail_size)
    outer = [(ir.Lit(v), const_t[k]) for k, v in const_val.items()]
    g_out = Grammar(r, outer, ops, constructs)
    defaults = g_out.enumerate(out_type, 1)

    # the loop bounds are the runs the anti-unifier found, not a number the
    # fixture handed over.
    bounds = sorted({(c["lo"], c["hi"]) for c in comps})
    if verbose:
        print("   loop bounds discovered from the families: %r" % (bounds,))
    cands = []
    if "Find" in constructs:
        for lo, hi in bounds:
            for pr in preds:
                for th in tails:
                    for d in defaults:
                        cands.append(ir.Find("i", lo, hi, pr, th, d))
    if "Scan" in constructs:
        # a bounded scan with an early-stop `while`: the other way to express it
        acc_type = out_type
        inner_acc = inner + [(ir.Var("a"), acc_type)]
        g_acc = Grammar(r, inner_acc, ops, constructs)
        steps = g_acc.enumerate(acc_type, body_size)
        whiles = g_acc.enumerate(BOOL, body_size)
        for lo, hi in bounds:
            for st in steps:
                for w in whiles:
                    for i0 in defaults:
                        cands.append(ir.Scan("i", "a", lo, hi, i0, st, w))
    # the eager alternatives are in the space too
    cands += g_out.enumerate(out_type, 1)
    cands += list(seed_candidates)

    if verbose:
        print("   %d top-level candidates" % len(cands))

    # the specification's constants are always available and cost nothing; a
    # candidate that mentions one must be able to resolve it.
    base = [(k, ir.Lit(v, k)) for k, v in const_val.items()]
    survivors = []
    for cand in cands:
        prog = ir.IRProgram(list(base), cand, inputs=(port,))
        try:
            if check(ev, prog, oracle, cases, port) is not None:
                continue
            c = expected_ops(ev, prog, cases, port)
        except Exception:
            continue
        survivors.append((c, cand))
    survivors.sort(key=lambda z: (z[0], ir._size(z[1])))
    if verbose:
        print("   %d candidates, %d exact on %d sampled inputs" % (len(cands), len(survivors), len(cases)))
    best = None
    rejected = 0
    for c, cand in survivors:
        prog = ir.IRProgram(list(base), cand, inputs=(port,))
        ok, bad, _ = verify_exhaustive(ev, prog, oracle, all_cases, port)
        if ok:
            best = (c, cand)
            break
        rejected += 1
    return {"program": None if best is None else ir.IRProgram(list(base), best[1], inputs=(port,)),
            "sampled_expected_ops": None if best is None else best[0],
            "candidates_considered": len(cands),
            "exact_on_samples": len(survivors),
            "rejected_by_exhaustive_check": rejected,
            "components": [{"lambda": repr(c["lambda"]), "arity": c["arity"],
                            "abstracted": c["abstracted"]} for c in comps]}
