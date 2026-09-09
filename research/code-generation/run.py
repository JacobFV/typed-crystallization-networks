"""Stage A: learn the ISA's executor.  Stage B: learn the code generator.

Stage A is scored against `generators/logic`'s own `evaluate` on the canonical
netlist, so what it learns is the target instruction set's semantics.  It is then
frozen and registered as a module.

Stage B is scored *through* that frozen executor on the behaviour alone: an
emitter conforms only when the netlist it emits reproduces the specified truth
table.  No reference netlist is shown to stage B.  Held-out behaviours are then
checked outside the substrate by running the emitted gate list through
`generators.logic.generator.evaluate`.
"""
from __future__ import annotations
import argparse, json, pathlib, random, sys, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from common import (ADDR, ALL_SPECS, Builder, BOOL, NIB, Registry, SPEC, TWO, Value,
                    behaviour, canonical, nib, runs_correctly, spec_value)
from tcn.graph import Signal
from tcn.search import enumerate_prefix, space_size, viability

OUT = pathlib.Path(__file__).resolve().parent / 'out'
X = ('x0', 'x1', 'x2')


# --- stage A ---------------------------------------------------------------
def executor_scaffold(registry):
    """`(c0, c1, x0, x1, x2) -> bool`.  Skeleton declared; every convention searched."""
    b = Builder(registry, (('c0', NIB), ('c1', NIB)) + tuple((k, BOOL) for k in X))
    b.choice('jt', [('tuple', (p, q), None, None) for p in X for q in X])
    b.add('j', 'pack', ['jt'], out=TWO)
    b.choice('v0', [('index', (c, 'j'), None, None) for c in ('c0', 'c1')])
    b.choice('v1', [('index', (c, 'j'), None, None) for c in ('c0', 'c1')])
    b.choice('out', [('mux', (s, a, c), None, None) for s in X for a in ('v0', 'v1')
                     for c in ('v0', 'v1')])
    return b.program((('y', 'out'),))

def executor_examples(pairs):
    rows = []
    for t0, t1 in pairs:
        spec = behaviour(t0, t1)
        for m in range(8):
            bits = [bool((m >> i) & 1) for i in range(3)]
            rows.append({'inputs': {'c0': Value.of(NIB, nib(t0)), 'c1': Value.of(NIB, nib(t1)),
                                    **{k: Value.of(BOOL, v) for k, v in zip(X, bits)}},
                         'targets': {'y': Value.of(BOOL, spec[m])}})
    return rows


# --- stage B ---------------------------------------------------------------
def emitter_scaffold(registry, module, fixed_a=None, fixed_b=None):
    """`(spec, x0, x1, x2) -> bool`: eight free addresses into the specification.

    `fixed_a` / `fixed_b` harden one head to a known address tuple, which is how
    the two heads are staged; `None` leaves that head's four addresses free.
    """
    consts = tuple((f'k{i}', Value.of(ADDR, i)) for i in range(8))
    b = Builder(registry, (('spec', SPEC),) + tuple((k, BOOL) for k in X), consts)
    for tag, fixed in (('a', fixed_a), ('b', fixed_b)):
        for slot in range(4):
            pool = range(8) if fixed is None else (fixed[slot],)
            b.choice(f'{tag}{slot}', [('index', ('spec', f'k{i}'), None, None) for i in pool])
        b.add(f't{tag}', 'tuple', [f'{tag}{slot}' for slot in range(4)])
    b.add('out', module, ['ta', 'tb'] + list(X))
    return b.program((('y', 'out'),))

def emitter_examples(specs, restrict=None):
    """One example per (specification, assignment).  `restrict` is (input, value)."""
    rows = []
    for spec in specs:
        for m in range(8):
            bits = [bool((m >> i) & 1) for i in range(3)]
            if restrict is not None and bits[X.index(restrict[0])] != restrict[1]:
                continue
            rows.append({'inputs': {'spec': spec_value(spec),
                                    **{k: Value.of(BOOL, v) for k, v in zip(X, bits)}},
                         'targets': {'y': Value.of(BOOL, spec[m])}})
    return rows

def addresses(result, tag):
    """Read one head's four addresses out of a search result."""
    return tuple(int(result.selections[f'{tag}{s}']) for s in range(4))

def emit(spec, addr_a, addr_b, port_a='c0'):
    """Run the emitter by hand to produce the artifact's two variable fields."""
    ta = sum(int(spec[addr_a[i]]) << i for i in range(4))
    tb = sum(int(spec[addr_b[i]]) << i for i in range(4))
    return (ta, tb) if port_a == 'c0' else (tb, ta)


# --- experiment ------------------------------------------------------------
SIGNALS = (Signal('out', 'y', ('core',), BOOL),)

def held_out(specs, addr_a, addr_b, port_a):
    """Exact, outside the substrate: does the emitted gate list compute the spec?"""
    exact = 0; per_assignment = 0
    for spec in specs:
        t0, t1 = emit(spec, addr_a, addr_b, port_a)
        gates = canonical(t0, t1)
        from generators.logic.generator import evaluate
        hits = sum(evaluate(3, gates, m) == spec[m] for m in range(8))
        per_assignment += hits
        exact += hits == 8
    return {'behaviours': len(specs), 'exact': exact,
            'exact_rate': exact / len(specs),
            'assignment_accuracy': per_assignment / (8 * len(specs))}

def baselines(train, test):
    from generators.logic.generator import evaluate
    def score(fn, specs):
        e = a = 0
        for spec in specs:
            t0, t1 = fn(spec)
            g = canonical(t0, t1)
            h = sum(evaluate(3, g, m) == spec[m] for m in range(8))
            a += h; e += h == 8
        return {'exact_rate': e / len(specs), 'assignment_accuracy': a / (8 * len(specs))}
    best, best_score = None, -1
    for t0 in range(16):
        for t1 in range(16):
            s = score(lambda _s, t0=t0, t1=t1: (t0, t1), train)['assignment_accuracy']
            if s > best_score: best, best_score = (t0, t1), s
    rng = random.Random(0)
    rand = [score(lambda spec, a=[rng.randrange(8) for _ in range(4)],
                  b=[rng.randrange(8) for _ in range(4)]: emit(spec, a, b), test)
            for _ in range(64)]
    return {
        'best_constant_netlist': {'netlist': best, **score(lambda _s: best, test)},
        'identity_addresses': score(lambda spec: emit(spec, (0, 1, 2, 3), (4, 5, 6, 7)), test),
        'random_addresses_mean': {
            'exact_rate': sum(r['exact_rate'] for r in rand) / len(rand),
            'assignment_accuracy': sum(r['assignment_accuracy'] for r in rand) / len(rand)},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage-a-pairs', type=int, default=12)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    report = {}
    registry = Registry()
    rng = random.Random(args.seed)

    # ---- stage A: the executor -------------------------------------------
    scaffold = executor_scaffold(registry)
    pairs = [(rng.randrange(16), rng.randrange(16)) for _ in range(args.stage_a_pairs)]
    train_a = executor_examples(pairs)
    ra = enumerate_prefix(scaffold, train_a, SIGNALS, registry, tolerance=1e-9)
    all_pairs = [(t0, t1) for t0 in range(16) for t1 in range(16)]
    holdout_a = executor_examples([p for p in all_pairs if p not in set(pairs)])
    hardened = scaffold.harden(ra.selections)
    from tcn.search import evaluate as exact_error
    err = exact_error(hardened, {}, holdout_a, SIGNALS, registry)
    report['stage_a'] = {'space': space_size(scaffold), **ra.to_dict(),
                         'train_pairs': len(pairs), 'train_examples': len(train_a),
                         'holdout_examples': len(holdout_a), 'holdout_max_error': err}

    # every conforming executor, and whether they are the same function
    # are the conforming executors the same function?  Keep every one and check.
    from tcn.search import candidate_counts
    import itertools
    variants = []
    names = [n.name for n in scaffold.nodes]
    for combo in itertools.product(*(range(c) for c in candidate_counts(scaffold))):
        sel = dict(zip(names, combo))
        if exact_error(scaffold, sel, train_a, SIGNALS, registry) == 0.:
            variants.append(sel)
    agree = all(exact_error(scaffold, v, train_a + holdout_a, SIGNALS, registry) == 0.
                for v in variants)
    report['stage_a']['conforming_variants'] = len(variants)
    report['stage_a']['variants_extensionally_identical'] = agree

    module = registry.register_module(hardened)
    # which module port serves which cofactor -- read off stage A, not declared
    out_cand = hardened.nodes[[n.name for n in hardened.nodes].index('out')].candidates[0]
    sel, arm_true, arm_false = out_cand.sources
    port_of = {}
    for name in ('v0', 'v1'):
        node = hardened.nodes[[n.name for n in hardened.nodes].index(name)]
        port_of[name] = node.candidates[0].sources[0]
    report['stage_a']['selector'] = sel
    report['stage_a']['port_when_selector_false'] = port_of[arm_false]
    report['stage_a']['module'] = module

    # ---- stage B: the emitter --------------------------------------------
    joint = emitter_scaffold(registry, module)
    report['stage_b_unstaged'] = {'space': space_size(joint),
                                  'viability': viability(joint, examples=8 * 12)}

    splits = {
        'random12': sorted(rng.sample(range(256), 12)),
        'popcount2': [f for f in range(256) if bin(f).count('1') == 2],
        'x2_independent': [f for f in range(256) if (f & 0xF) == (f >> 4)],
    }
    report['stage_b'] = {}
    for name, train_ids in splits.items():
        train = [ALL_SPECS[f] for f in train_ids]
        test = [ALL_SPECS[f] for f in range(256) if f not in set(train_ids)]
        entry = {'train_behaviours': len(train), 'holdout_behaviours': len(test)}
        found = {}
        # head 'a' feeds module port c0, head 'b' port c1; which selector value
        # pins which head is read off stage A rather than declared.
        pin = {port_of[arm_false]: False, port_of[arm_true]: True}
        for tag, value in (('a', pin['c0']), ('b', pin['c1'])):
            # the head read when the selector is `value` is the one these examples pin
            other = 'b' if tag == 'a' else 'a'
            prog = emitter_scaffold(registry, module,
                                   **{f'fixed_{other}': (0, 0, 0, 0)})
            ex = emitter_examples(train, restrict=(sel, value))
            r = enumerate_prefix(prog, ex, SIGNALS, registry, tolerance=1e-9)
            entry[f'head_{tag}'] = {'space': space_size(prog), 'examples': len(ex), **r.to_dict()}
            found[tag] = addresses(r, tag) if r.solved else None
        entry['addresses'] = found
        if found['a'] and found['b']:
            entry['held_out'] = held_out(test, found['a'], found['b'], 'c0')
            entry['all_256'] = held_out(list(ALL_SPECS), found['a'], found['b'], 'c0')
            from common import artifact
            entry['example_artifacts'] = [
                {'spec': ''.join('1' if v else '0' for v in ALL_SPECS[f]),
                 'gate_list': artifact(*emit(ALL_SPECS[f], found['a'], found['b']))[0]}
                for f in (0b10010110, 0b11101000, 0b00000001)]
            entry['train_fit'] = held_out(train, found['a'], found['b'], 'c0')
            entry['baselines'] = baselines(train, test)
        report['stage_b'][name] = entry

    OUT.mkdir(exist_ok=True)
    (OUT / 'run.json').write_text(json.dumps(report, indent=2, sort_keys=True, default=str))
    print(json.dumps(report, indent=2, sort_keys=True, default=str))

if __name__ == '__main__':
    main()
