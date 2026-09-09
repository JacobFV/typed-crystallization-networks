"""Bound the target before searching: does the specification determine the program?

The specification the code generator sees is a *behaviour* -- the full truth table
of a function of three Boolean inputs, 256 of them.  The artifact is a gate list
for `generators/logic` at width 3: a sequence of `(wire_a, wire_b, table)` triples
executed by that generator's own `evaluate`.

Two questions, both answered exactly by exhaustion here rather than argued:

1. Over *free* gate lists of a given length, how many programs realise one
   behaviour?  This is the non-uniqueness of the target language itself.
2. Over the *declared target shape* -- the Shannon skeleton this track compiles
   to -- how many?

Wire values are carried as 8-bit masks over the eight input assignments, so one
gate is a table lookup on a pair of masks and the whole enumeration is exact.
The mask convention is checked against `generators.logic.generator.evaluate`.
"""
from __future__ import annotations
import json, pathlib, sys, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from generators.logic.generator import evaluate

WIDTH = 3
FULL = 0xFF
# mask of assignments m (m's bit i is input i) where input i is true
INPUT_MASKS = tuple(sum(1 << m for m in range(8) if (m >> i) & 1) for i in range(WIDTH))

def gate_mask(a, b, t):
    """Output mask of one gate, given the masks of its two source wires."""
    out = 0
    for u in (0, 1):
        for v in (0, 1):
            if (t >> (2 * u + v)) & 1:
                out |= (a if u else ~a) & (b if v else ~b)
    return out & FULL

def netlist_mask(gates):
    wires = list(INPUT_MASKS)
    for a, b, t in gates:
        wires.append(gate_mask(wires[a], wires[b], t))
    return wires[-1]

def check_mask_convention():
    """The mask algebra must agree with the generator's own evaluator, exactly."""
    import itertools, random
    rng = random.Random(0)
    for _ in range(2000):
        depth = rng.randrange(1, 4)
        gates = []
        for i in range(depth):
            n = WIDTH + i
            gates.append([rng.randrange(n), rng.randrange(n), rng.randrange(16)])
        m = netlist_mask(gates)
        for a in range(8):
            if bool((m >> a) & 1) != evaluate(WIDTH, gates, a):
                raise AssertionError("mask convention disagrees with generators/logic")
    return True

def counts_by_depth(max_depth=3):
    """counts[d][f] = number of width-3 gate lists of length d computing behaviour f."""
    out = {}
    states = {tuple(): 1}                      # partial netlist -> multiplicity, keyed by wire masks
    frontier = {tuple(INPUT_MASKS): 1}
    for d in range(1, max_depth + 1):
        nxt = {}
        started = time.perf_counter()
        for wires, mult in frontier.items():
            n = len(wires)
            for a in range(n):
                A = wires[a]
                for b in range(n):
                    B = wires[b]
                    for t in range(16):
                        key = wires + (gate_mask(A, B, t),)
                        nxt[key] = nxt.get(key, 0) + mult
        per_f = {}
        for wires, mult in nxt.items():
            per_f[wires[-1]] = per_f.get(wires[-1], 0) + mult
        out[d] = {'per_behaviour': per_f,
                  'programs': sum(per_f.values()),
                  'behaviours_reached': len(per_f),
                  'seconds': time.perf_counter() - started,
                  'states': len(nxt)}
        frontier = nxt
    return out

def canonical(t0, t1):
    """The declared target shape: Shannon expansion on input 2, five gates.

    wires 0,1,2 are the inputs; 3 = t0(x0,x1), 4 = t1(x0,x1),
    5 = w3 and not x2, 6 = w4 and x2, 7 = w5 or w6.
    """
    return [[0, 1, int(t0)], [0, 1, int(t1)], [3, 2, 4], [4, 2, 8], [5, 6, 14]]

def canonical_bound():
    """How many (t0, t1) pairs realise each behaviour under the declared shape."""
    per_f = {}
    for t0 in range(16):
        for t1 in range(16):
            f = netlist_mask(canonical(t0, t1))
            per_f.setdefault(f, []).append((t0, t1))
    return per_f

if __name__ == '__main__':
    check_mask_convention()
    cb = canonical_bound()
    depth = counts_by_depth(3)
    report = {'mask_convention_checked_against_generator': True,
              'canonical_shape': {'behaviours_covered': len(cb),
                                  'programs_per_behaviour': sorted({len(v) for v in cb.values()}),
                                  'total_pairs': sum(len(v) for v in cb.values())},
              'free_gate_lists': {}}
    for d, r in depth.items():
        pf = r['per_behaviour']
        vals = sorted(pf.values())
        report['free_gate_lists'][d] = {
            'programs': r['programs'], 'behaviours_reached': r['behaviours_reached'],
            'min': vals[0], 'median': vals[len(vals) // 2], 'max': vals[-1],
            'seconds': round(r['seconds'], 2)}
    path = pathlib.Path(__file__).resolve().parent / 'out' / 'bound.json'
    path.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))
