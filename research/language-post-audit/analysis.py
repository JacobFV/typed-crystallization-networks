"""Three follow-on measurements that the headline needs beside it.

1. `transfer`  -- what the pre-audit exported program (the selection recorded in
   `research/language-capability/stage_b.json`, i.e. section 19's program) scores
   on the honestly-posed post-audit split. Labelled as out-of-distribution
   transfer, not as a result about the method.
2. `best_train_members` -- since no member of the stage-B family fits the 24
   training episodes exactly, what the *best-fitting* members get on held-out.
   This is what a tolerance-relaxed search would have returned.
3. `full_string_ceiling` -- the closed-form ceiling for any stage-B member that
   reads the whole string: on the post-audit stream `#( == #)` always holds, so
   the accumulator is `(plus+minus) * L/2`, a function of length alone, and the
   best such program is the per-length majority.
"""
from __future__ import annotations
import sys, os, json, collections, importlib.util
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
ROOT = os.path.dirname(os.path.dirname(HERE)); sys.path.insert(0, ROOT)
import common, splits as splits_mod, bound
from run_stage_b import build_module, accuracy, baselines, scaffolds


def transfer(s):
    """Section 19's exported selection, run on the post-audit split."""
    sel = json.load(open(os.path.join(ROOT, 'research', 'language-capability',
                                      'stage_b.json')))['enumeration']['selections']
    module, registry, frozen = build_module()
    program, _ = scaffolds.stage_b(module, registry, positions=16)
    exported = program.harden(sel).pruned()
    out = {'note': 'pre-audit-trained frozen program (FINDINGS section 19) evaluated on the '
                   'post-audit stream; out-of-distribution transfer, not a method result',
           'selection_source': 'research/language-capability/stage_b.json'}
    for name, eps in s.items():
        acc, per_len = accuracy(exported, {n.name: 0 for n in exported.nodes}, registry, eps)
        out[name] = {'accuracy': acc, 'per_length': per_len, **baselines(eps)}
    return out


def best_train_members(s, positions):
    sw = bound.sweep(s, positions)
    # re-sweep keeping every member's train accuracy so the best-fitting ones can be listed
    feats = {k: [bound.features(e, positions) for e in eps] for k, eps in s.items()}
    labels = {k: [e['label'] for e in eps] for k, eps in s.items()}
    lengths = {k: [e['prompt_bytes'] for e in eps] for k, eps in s.items()}
    rows = []
    for c in bound.SUB_RANGE:
        for plus in bound.STEPS:
            for minus in bound.STEPS:
                accs = {}
                for k, eps in s.items():
                    row = []
                    for j, e in enumerate(eps):
                        L = lengths[k][j]
                        if c > L:
                            row.append(None); continue
                        K = max(0, min(positions, L - c))
                        o = feats[k][j][K]
                        row.append(plus * o + minus * (K - o))
                    accs[k] = row
                for (op, v) in bound.RULES:
                    hits = {}
                    for k in s:
                        ok = 0
                        for a, y in zip(accs[k], labels[k]):
                            if a is None: continue
                            p = (a == v) if op == 'eq' else (a >= v) if op == 'ge' else (a <= v)
                            ok += (p == y)
                        hits[k] = ok / max(1, len(labels[k]))
                    rows.append(({'c': c, 'plus': plus, 'minus': minus, 'op': op, 'v': v}, hits))
    top = max(r[1]['train'] for r in rows)
    best = [r for r in rows if r[1]['train'] == top]
    held = [r[1]['heldout_unseen_lengths'] for r in best]
    return {'positions': positions, 'best_train_accuracy': top, 'n_members_at_best': len(best),
            'heldout_unseen_of_those': {'min': min(held), 'max': max(held),
                                        'mean': sum(held) / len(held)},
            'examples': [{'params': p, **h} for p, h in best[:8]],
            'sweep': sw}


def full_string_ceiling(s):
    out = {'note': 'best accuracy of any function of string length alone, which is what a '
                   'stage-B member that reads the whole string reduces to on this stream'}
    for name, eps in s.items():
        tab = collections.defaultdict(collections.Counter)
        for e in eps:
            tab[e['length']][e['label']] += 1
        ok = sum(max(c.values()) for c in tab.values())
        out[name] = {'per_length_majority_ceiling': ok / len(eps),
                     'majority_constant': baselines(eps)['majority_constant'],
                     'per_length': {k: {'n': sum(c.values()),
                                        'positive_rate': c[True] / sum(c.values())}
                                    for k, c in sorted(tab.items())}}
    return out


def main():
    s = splits_mod.build()
    out = {'stream': common.STREAM_POST_AUDIT,
           'transfer_preaudit_program': transfer(s),
           'full_string_ceiling': full_string_ceiling(s),
           'best_train_members_p22': best_train_members(s, 22),
           'best_train_members_p16': best_train_members(s, 16)}
    json.dump(out, open(os.path.join(HERE, 'analysis.json'), 'w'), indent=1)
    for k in ('transfer_preaudit_program',):
        print(k, json.dumps({n: {kk: vv for kk, vv in v.items() if kk in ('accuracy', 'majority_constant')}
                             for n, v in out[k].items() if isinstance(v, dict)}, indent=1))
    print('full_string_ceiling unseen',
          out['full_string_ceiling']['heldout_unseen_lengths']['per_length_majority_ceiling'])
    for p in (16, 22):
        b = out[f'best_train_members_p{p}']
        print(f'p{p}: best train {b["best_train_accuracy"]} over {b["n_members_at_best"]} members; '
              f'their unseen {b["heldout_unseen_of_those"]}')


if __name__ == '__main__':
    main()
