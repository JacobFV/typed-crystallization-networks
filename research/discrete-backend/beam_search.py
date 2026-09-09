"""Acceptance test for the prefix walk and the beam over it.

Two arms, because they answer different questions.

**A, a real task.** The rung-3 foreground module of
`research/discrete-perception/rung3_mask.py` at R=8: 32,000 programs over 384
supervised pixels, the arm FINDINGS section 14 reports 2,464-2,608 conforming for
and where a prefix-reusing prototype was measured 17.8x faster than the flat
sweep. Measured here: `enumerate_fit` against `enumerate_prefix` on the identical
scaffold and data (same conforming count, same returned program, certificate
intact), then the beam at a range of widths.

**B, a controlled arm.** A four-node chain of truth tables, 65,536 programs,
run twice over the *same* program and the *same* data with only the supervision
changed: probes on every node, or a probe on the output alone. This isolates the
one thing that decides whether a beam is a search or merely a budget -- whether
a partial prefix has a score at all.

Hand-initializations, declared. Arm A's scaffold, byte pool and example sampler
are the discrete-perception track's unchanged. Arm B's reference program (tables
6, 1, 9, 14) is used only to generate targets; the search is over all 16
candidates at each of the four nodes and is never told any of them.
"""
from __future__ import annotations
import itertools
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'research' / 'discrete-perception'))

from tcn.graph import Program, Node, Candidate, Signal
from tcn.operators import Registry
from tcn.search import enumerate_fit, enumerate_prefix, evaluate, space_size, viability
from tcn.types import BOOL, Value

OUT = Path(__file__).parent / 'out'
RESOLUTION = 8
TRAIN_IMAGES = 8
PER_IMAGE = 48
WIDTHS = (1, 4, 16, 64, 256, 1024)
CHAIN = (6, 1, 9, 14)


# ---------------------------------------------------------------------------
# arm A: the rung-3 foreground module
# ---------------------------------------------------------------------------
def arm_perception():
    from rung3_mask import POOL, module_scaffold, pixel_examples, signals
    registry = Registry()
    program = module_scaffold(registry, RESOLUTION, POOL)
    train = pixel_examples(tuple(range(TRAIN_IMAGES)), RESOLUTION, 'train', PER_IMAGE,
                           seed=1, objects=6)
    sig = signals()
    row = {'resolution': RESOLUTION, 'observation_width': 3 * RESOLUTION * RESOLUTION,
           'records': len(train), 'space_size': space_size(program),
           'probes': [s.source for s in sig],
           'viability': viability(program, examples=len(train))}

    flat = enumerate_fit(program, train, sig, registry, tolerance=1e-6)
    walk = enumerate_prefix(program, train, sig, registry, tolerance=1e-6)
    row['enumerate_fit'] = flat.to_dict()
    row['enumerate_prefix'] = walk.to_dict()
    row['identical'] = (flat.conforming == walk.conforming and flat.selections == walk.selections
                        and flat.unique == walk.unique and flat.certificate == walk.certificate)
    row['speedup'] = flat.seconds / max(1e-9, walk.seconds)
    row['beam'] = []
    for width in WIDTHS:
        found = enumerate_prefix(program, train, sig, registry, tolerance=1e-6, beam=width)
        d = found.to_dict()
        d['matches_exhaustive_pick'] = found.selections == flat.selections
        d['fraction_of_space_examined'] = found.evaluated / found.space_size
        row['beam'].append(d)
    return row


# ---------------------------------------------------------------------------
# arm B: a chain, supervised densely and then only at the output
# ---------------------------------------------------------------------------
def chain_program(registry):
    nodes = []
    sources = [('x0', 'x1'), ('h0', 'x0'), ('h1', 'x1'), ('h2', 'x0')]
    for i, src in enumerate(sources):
        nodes.append(Node(f'h{i}', BOOL,
                          tuple(Candidate(registry.resolve(f'truth_{t}', (BOOL, BOOL)), src)
                                for t in range(16)), 'core', i + 1))
    return Program((('x0', BOOL), ('x1', BOOL)), tuple(nodes), (('y', 'h3'),)).validate(registry)


def chain_examples(registry, program):
    """Every input assignment, with the reference program's own intermediates."""
    reference = {f'h{i}': t for i, t in enumerate(CHAIN)}
    out = []
    for x0, x1 in itertools.product((False, True), repeat=2):
        inputs = {'x0': Value.of(BOOL, x0), 'x1': Value.of(BOOL, x1)}
        _, _, trace = program.execute(inputs, registry=registry, selections=reference)
        out.append({'inputs': inputs, 'targets': {f'h{i}': trace[f'h{i}'] for i in range(4)}})
    return reference, out


def arm_chain():
    registry = Registry()
    program = chain_program(registry)
    reference, examples = chain_examples(registry, program)
    dense = tuple(Signal(f'h{i}', f'h{i}', ('core',), BOOL, 'bce') for i in range(4))
    sparse = (Signal('h3', 'h3', ('core',), BOOL, 'bce'),)
    row = {'space_size': space_size(program), 'reference': reference,
           'examples': len(examples), 'supervision': {}}
    for label, sig in (('dense', dense), ('output_only', sparse)):
        flat = enumerate_fit(program, examples, sig, registry, tolerance=1e-6)
        walk = enumerate_prefix(program, examples, sig, registry, tolerance=1e-6)
        entry = {'enumerate_fit': flat.to_dict(), 'enumerate_prefix': walk.to_dict(),
                 'identical': flat.conforming == walk.conforming and flat.selections == walk.selections,
                 'reference_conforms': evaluate(program, reference, examples, sig, registry) == 0.,
                 'beam': []}
        for width in (1, 4, 16, 64):
            found = enumerate_prefix(program, examples, sig, registry, tolerance=1e-6, beam=width)
            d = found.to_dict()
            d['recovers_reference'] = found.selections == reference
            d['fraction_of_space_examined'] = found.evaluated / found.space_size
            entry['beam'].append(d)
        row['supervision'][label] = entry
    return row


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.time()
    report = {'widths': list(WIDTHS)}
    report['chain'] = arm_chain()
    for label, entry in report['chain']['supervision'].items():
        f = entry['enumerate_fit']; w = entry['enumerate_prefix']
        print(f"chain/{label:11s} exhaustive conforming {f['conforming']:5d} "
              f"certificate {f['certificate']:8s} flat {f['seconds']:.2f}s "
              f"prefix {w['seconds']:.2f}s ({f['node_evaluations']:,} vs "
              f"{w['node_evaluations']:,} node evaluations), reference conforms "
              f"{entry['reference_conforms']}")
        for d in entry['beam']:
            print(f"    beam {d['beam']:5d}: solved {d['solved']} conforming {d['conforming']:4d} "
                  f"reference {d['recovers_reference']} discarded {d['discarded']:6d} "
                  f"examined {d['fraction_of_space_examined']:.4f} certificate {d['certificate']}")
    report['perception'] = arm_perception()
    p = report['perception']
    print(f"rung3 R={RESOLUTION} space {p['space_size']:,} records {p['records']}: "
          f"flat {p['enumerate_fit']['seconds']:.1f}s -> prefix "
          f"{p['enumerate_prefix']['seconds']:.1f}s ({p['speedup']:.1f}x), identical "
          f"{p['identical']}, conforming {p['enumerate_fit']['conforming']}, certificate "
          f"{p['enumerate_prefix']['certificate']}")
    for d in p['beam']:
        print(f"    beam {d['beam']:5d}: solved {d['solved']} conforming {d['conforming']:5d} "
              f"same pick {d['matches_exhaustive_pick']} discarded {d['discarded']:6d} "
              f"examined {d['fraction_of_space_examined']:.4f} certificate {d['certificate']} "
              f"{d['seconds']:.1f}s")
    report['seconds'] = time.time() - started
    (OUT / 'beam_search.json').write_text(json.dumps(report, indent=2))
    print(f"total {report['seconds']:.1f}s")


if __name__ == '__main__':
    main()
