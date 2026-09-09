"""R1: measure the row-deduplication gain and its worst case, and check identity.

The "before" is reproduced here rather than taken from the previous tree, so both
sides run in this process against the same registry and the same tensors.
"""
import itertools, json, math, statistics, sys, time
import torch

sys.path.insert(0, '../recursive-abstraction-retest')
import common
from tcn.types import BOOL, Value, integer, floating
from tcn.operators import Registry
from tcn.graph import Program, Node, Candidate
from tcn.learning import exact_tensor, SoftProgram, tensor


def shipped(registry, op, xs):
    """`exact_tensor` as it stood before R1: one exact execution per batch row."""
    shape = torch.broadcast_shapes(*(x.shape[:-1] for x in xs)) if xs else ()
    batch = math.prod(shape) if shape else 1
    flat = [x.detach().expand(*shape, x.shape[-1]).reshape(batch, -1).cpu().tolist() for x in xs]
    vals = []
    for i in range(batch):
        args = [Value.unflat(t, x[i]) for t, x in zip(op.inputs, flat)]
        vals.append(registry.exact(op, args).flat())
    device = xs[0].device if xs else None
    return torch.tensor(vals, dtype=torch.float32, device=device).reshape(*shape, op.output.width)


def timed(fn, reps):
    fn()
    best = []
    for _ in range(5):
        t0 = time.perf_counter()
        for _ in range(reps): fn()
        best.append((time.perf_counter() - t0) / reps * 1e6)
    return min(best)


def case(label, registry, op, xs, reps=20):
    a = shipped(registry, op, xs); b = exact_tensor(registry, op, xs)
    identical = torch.equal(a, b) and a.dtype == b.dtype and a.shape == b.shape
    batch = a.shape[0]
    flat = [x.expand(batch, x.shape[-1]).reshape(batch, -1).tolist() for x in xs]
    distinct = len({tuple(tuple(x[i]) for x in flat) for i in range(batch)})
    before = timed(lambda: shipped(registry, op, xs), reps)
    after = timed(lambda: exact_tensor(registry, op, xs), reps)
    return {'case': label, 'batch': batch, 'distinct_rows': distinct,
            'bit_identical': identical, 'shipped_us': round(before, 1), 'deduped_us': round(after, 1),
            'speedup': round(before / after, 2)}


out = {'cases': []}
r = Registry()
maj = common.minimal_module(r, 'maj')
name = r.register_module(maj)
mop = r.resolve(name, (BOOL, BOOL, BOOL))

rows = list(itertools.product((0., 1.), repeat=6))
cols = [torch.tensor([[row[i]] for row in rows]) for i in range(6)]
out['cases'].append(case('MAJ3 module over (a,b,c), 64-row truth table', r, mop, cols[:3]))
out['cases'].append(case('MAJ3 module over (a,a,b), 64-row truth table', r, mop, [cols[0], cols[0], cols[1]]))

# worst case: a carrier wide enough that rows do not repeat
I8 = integer(8, signed=False)
adder = Program((('a', I8), ('b', I8)),
                (Node('s', I8, (Candidate(r.resolve('add', (I8, I8)), ('a', 'b')),), 'core', 1, 0),),
                (('out', 's'),)).validate(r)
aname = r.register_module(adder)
aop = r.resolve(aname, (I8, I8))
torch.manual_seed(0)
for batch in (64, 512):
    xa = torch.randint(0, 128, (batch, 1)).float(); xb = torch.randint(0, 128, (batch, 1)).float()
    out['cases'].append(case(f'int[8] add module, {batch} random rows (worst case)', r, aop, [xa, xb], reps=5))

F = floating()
fop = r.resolve('atan2', (F, F))
xa = torch.randn(64, 1); xb = torch.randn(64, 1).abs() + .5
out['cases'].append(case('float atan2 primitive, 64 random rows (worst case)', r, fop, [xa, xb]))

# the figure that matters: one forward pass over arm B's scaffold
for label, scaffold in (('tight arm B', common.tight_scaffold(r, name)), ('wide arm B', common.wide_scaffold(r, name))):
    ex = common.composite_examples()
    model = SoftProgram(scaffold, r)
    inputs = {k: torch.stack([tensor(e['inputs'][k]) for e in ex]) for k, _ in scaffold.inputs}
    torch.set_num_threads(1)
    import tcn.learning as L
    def forward():
        model(inputs, return_trace=True)
    after = timed(forward, 1)
    original = L.exact_tensor
    L.exact_tensor = shipped
    before = timed(forward, 1)
    L.exact_tensor = original
    out['cases'].append({'case': f'{label} scaffold, one soft forward pass over 64 rows',
                         'batch': 64, 'distinct_rows': None, 'bit_identical': None,
                         'shipped_us': round(before, 1), 'deduped_us': round(after, 1),
                         'speedup': round(before / after, 2)})

out['all_bit_identical'] = all(c['bit_identical'] for c in out['cases'] if c['bit_identical'] is not None)
json.dump(out, open('dedup.json', 'w'), indent=1)
for c in out['cases']:
    print(f"{c['case']:<58} distinct={str(c['distinct_rows']):>5} identical={c['bit_identical']} "
          f"{c['shipped_us']:>9} -> {c['deduped_us']:>9} us  x{c['speedup']}")
print('all bit-identical:', out['all_bit_identical'])
