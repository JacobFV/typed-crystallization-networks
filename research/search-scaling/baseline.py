"""Random-program baseline: solution density of the discrete candidate space."""
import itertools, json, os, random, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import (N_IN, INPUT_MASKS, POOLS, apply_table, sample_target, build_program,
                     circuit_mask, FULL)

def density(n_nodes, tables, window=None, target_mask=0, draws=200000, seed=0):
    rng = random.Random(seed)
    names = list(range(N_IN))
    pools = []
    for j in range(n_nodes):
        avail = list(range(N_IN + j))
        if window is not None:
            avail = avail[max(0, len(avail) - window):]
        pools.append(avail)
    hits = 0; fns = set()
    for _ in range(draws):
        vals = list(INPUT_MASKS)
        for j in range(n_nodes):
            a = rng.choice(pools[j]); b = rng.choice(pools[j]); t = rng.choice(tables)
            vals.append(apply_table(vals[a], vals[b], t))
        fns.add(vals[-1])
        if vals[-1] == target_mask:
            hits += 1
    return hits / draws, len(fns)

if __name__ == '__main__':
    out = []
    for depth in (1, 2, 3, 4, 6, 8):
        for s in range(8):
            gates, mask, _, _ = sample_target(depth, 0, 1000 * s)
            d, nf = density(depth, list(range(16)), None, mask, draws=100000, seed=s)
            out.append({'depth': depth, 'seed': s, 'mask': mask, 'density': d,
                        'distinct_fns_sampled': nf})
    json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     'baseline.json'), 'w'))
    import statistics as st
    for depth in (1, 2, 3, 4, 6, 8):
        rs = [r for r in out if r['depth'] == depth]
        ds = [r['density'] for r in rs]
        print(f"depth {depth}: median solution density {st.median(ds):.2e}  "
              f"expected random draws {1/max(st.median(ds),1e-9):.0f}  "
              f"per-seed {[f'{d:.1e}' for d in ds]}")

def hard_report():
    import statistics as st, collections
    rows = []
    for depth in (3, 4, 6, 8):
        ds = []
        for s in range(8):
            gates, mask, _, _ = sample_target(depth, 0, 1000 * s, min_relevant=4)
            d, nf = density(depth, list(range(16)), None, mask, draws=100000, seed=s)
            ds.append(d)
        print(f"hard depth {depth}: median density {st.median(ds):.2e} "
              f"expected draws {1/max(st.median(ds),1e-9):.0f} per-seed {[f'{x:.1e}' for x in ds]}")
