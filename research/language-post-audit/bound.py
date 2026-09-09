"""Bound the stage-B family on the post-audit stream *before* spending the search.

The stage-B scaffold's only free choices are the length constant `c`, the two
step values `plus`/`minus`, and the rule `(op, v)`. With stage A exact, every
program in the family computes

    acc  = plus * (#open in the first K symbols) + minus * (K - #open)
    K    = min(positions, length_in_bytes - c)          (clipped at 0)
    ans  = op(acc, v)                                   op in {eq, ge, le}

so the family is exactly *thresholded affine functions of a bracket count over a
prefix*. This file enumerates all 121 x 5 x 5 x 15 = 45,375 members directly and
reports the best accuracy any of them attains on each split -- an upper bound on
what the search can return, independent of the search.

The simulator is validated against the real hardened program on random members
before any number is reported, because a bound computed from a wrong semantics
is worse than no bound.
"""
from __future__ import annotations
import sys, os, json, random, importlib.util, collections
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
ROOT = os.path.dirname(os.path.dirname(HERE)); sys.path.insert(0, ROOT)
import common, splits as splits_mod

_spec = importlib.util.spec_from_file_location(
    'lc_scaffolds', os.path.join(ROOT, 'research', 'language-capability', 'scaffolds.py'))
scaffolds = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(scaffolds)

SUB_RANGE = list(range(0, 121))
STEPS = (-2, -1, 0, 1, 2)
RULE_CONSTS = (-2, -1, 0, 1, 2)
OPS = ('eq', 'ge', 'le')
# rules are built c-major then op-minor, exactly as scaffolds.stage_b does
RULES = [(op, v) for v in RULE_CONSTS for op in OPS]


def features(e, positions):
    """(#open, #closed_or_absent) for every prefix length K = 0..positions."""
    s = e['string']
    opens = [0] * (positions + 1)
    for i in range(positions):
        opens[i + 1] = opens[i] + (1 if i < len(s) and s[i] == '(' else 0)
    return opens


def evaluate(e, positions, c, plus, minus, op, v, opens=None):
    """None where the real program would raise (POS underflow)."""
    length = e['prompt_bytes']
    if c > length:
        return None                      # unsigned sub underflows -> operator error
    symbols = length - c
    K = min(positions, symbols)
    if K < 0:
        K = 0
    if opens is None:
        opens = features(e, positions)
    o = opens[K]
    acc = plus * o + minus * (K - o)
    if op == 'eq':
        return acc == v
    if op == 'ge':
        return acc >= v
    return acc <= v


def _validate(eps, positions, module_builder, trials=150, seed=7):
    """Simulator vs. the real hardened program on random family members."""
    module, registry, frozen = module_builder()
    program, _ = scaffolds.stage_b(module, registry, positions=positions)
    names = [n.name for n in program.nodes]
    idx = {n.name: i for i, n in enumerate(program.nodes)}
    rnd = random.Random(seed)
    mism, checked, raised = 0, 0, 0
    for _ in range(trials):
        c = rnd.choice(SUB_RANGE); pi = rnd.randrange(5); mi = rnd.randrange(5)
        ri = rnd.randrange(len(RULES))
        e = rnd.choice(eps)
        sel = {n.name: 0 for n in program.nodes}
        sel['symbols'] = SUB_RANGE.index(c); sel['plus'] = pi; sel['minus'] = mi
        sel['answer'] = ri
        try:
            out, _ = program.run({'text': e['text']}, registry=registry, selections=sel)
            real = bool(out['answer'].decoded)
        except Exception:
            real = None
            raised += 1
        op, v = RULES[ri]
        sim = evaluate(e, positions, c, STEPS[pi], STEPS[mi], op, v)
        checked += 1
        if (real is None) != (sim is None) or (real is not None and real != sim):
            mism += 1
            if mism <= 3:
                print('  MISMATCH', dict(c=c, plus=STEPS[pi], minus=STEPS[mi], rule=(op, v),
                                         L=e['length'], real=real, sim=sim))
    return {'checked': checked, 'mismatches': mism, 'program_raised': raised}


def sweep(splits, positions):
    """Best-in-family accuracy per split, and the members that fit train exactly."""
    feats = {k: [features(e, positions) for e in eps] for k, eps in splits.items()}
    labels = {k: [e['label'] for e in eps] for k, eps in splits.items()}
    lengths = {k: [e['prompt_bytes'] for e in eps] for k, eps in splits.items()}
    best = {k: {'accuracy': -1.0, 'params': None} for k in splits}
    train_exact = []
    n_members = 0
    for c in SUB_RANGE:
        for pi, plus in enumerate(STEPS):
            for mi, minus in enumerate(STEPS):
                # acc per episode depends only on (c, plus, minus); rules are a readout
                accs = {}
                for k, eps in splits.items():
                    row = []
                    for j, e in enumerate(eps):
                        L = lengths[k][j]
                        if c > L:
                            row.append(None); continue
                        K = max(0, min(positions, L - c))
                        o = feats[k][j][K]
                        row.append(plus * o + minus * (K - o))
                    accs[k] = row
                for ri, (op, v) in enumerate(RULES):
                    n_members += 1
                    hits = {}
                    for k in splits:
                        ok = 0
                        for a, y in zip(accs[k], labels[k]):
                            if a is None:
                                continue     # program raises -> counted as a miss
                            p = (a == v) if op == 'eq' else (a >= v) if op == 'ge' else (a <= v)
                            ok += (p == y)
                        hits[k] = ok / max(1, len(labels[k]))
                    for k in splits:
                        if hits[k] > best[k]['accuracy']:
                            best[k] = {'accuracy': hits[k],
                                       'params': {'c': c, 'plus': plus, 'minus': minus,
                                                  'op': op, 'v': v},
                                       'other_splits': dict(hits)}
                    if hits.get('train', 0.0) == 1.0:
                        train_exact.append({'params': {'c': c, 'plus': plus, 'minus': minus,
                                                       'op': op, 'v': v}, **hits})
    return {'positions': positions, 'family_members': n_members,
            'best_in_family_per_split': best,
            'train_exact_fits': train_exact,
            'n_train_exact_fits': len(train_exact)}


def main():
    from run_stage_b import build_module
    s = splits_mod.build()
    out = {'stream': common.STREAM_POST_AUDIT,
           'split_sizes': {k: len(v) for k, v in s.items()},
           'majority_constant': {k: max(sum(e['label'] for e in v), len(v) - sum(e['label'] for e in v)) / len(v)
                                 for k, v in s.items()}}
    for positions in (16, 22):
        print(f'validating simulator at positions={positions} ...')
        out[f'validation_p{positions}'] = _validate(s['train'] + s['heldout_unseen_lengths'][:40],
                                                    positions, build_module)
        print(' ', out[f'validation_p{positions}'])
        print(f'sweeping family at positions={positions} ...')
        out[f'sweep_p{positions}'] = sweep(s, positions)
        b = out[f'sweep_p{positions}']['best_in_family_per_split']
        print('  best-in-family train ', round(b['train']['accuracy'], 4))
        print('  best-in-family unseen', round(b['heldout_unseen_lengths']['accuracy'], 4),
              b['heldout_unseen_lengths']['params'])
        print('  train-exact fits     ', out[f'sweep_p{positions}']['n_train_exact_fits'])
    json.dump(out, open(os.path.join(HERE, 'bound.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
