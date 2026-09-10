"""Q3c: is "add a second accumulator" mechanically generable from core's operators?

The scaffold change section 45 authored is a template with one hole -- which
binary operator is folded along the prefix sums. `accum_scaffold.core_folds`
reads the candidates for that hole straight out of `tcn.operators.BINARY` by
asking the registry to resolve each name at `CNT x CNT -> CNT`; no list is
hand-curated and nothing is added to core.

For each candidate the whole family (121 x 5 x 5 x 15 x 15 = 680,625 members) is
decided on the 24 training episodes by the simulator in `family.py`, which is
validated against the real typed program first -- per candidate, since the
operator is what changes. Every headline is then confirmed on the **real**
program: the first-in-order conforming member for each fold is run through
`tcn.search.evaluate` and scored on all three splits by exact execution.

If sweeping the hole recovers an operator that solves the task, the scaffold
change was proposable without knowing the answer. If it does not, scaffold design
is a human input at the operator level too.

Post-audit stream, pinned by `prepare.py`.
"""
from __future__ import annotations
import sys, os, json, time, collections
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
ROOT = os.path.dirname(os.path.dirname(HERE)); sys.path.insert(0, ROOT)
import common, prepare, family, accum_scaffold
from run_stage_b import build_module, examples, accuracy, baselines
from tcn.search import evaluate, space_size

POSITIONS = 22
STEPS = family.STEP_VALUES
RULES = family.RULES
SUB = family.SUB


def mask(bits):
    m = 0
    for i, b in enumerate(bits):
        if b:
            m |= 1 << i
    return m


def sweep(fold, eps, labels, opens_cache):
    """Decide the whole family on `eps`. Returns conforming members and counts."""
    lab = mask(labels); n = len(eps)
    full = (1 << n) - 1
    conforming = []
    usable = 0
    for c in SUB:
        for plus in STEPS:
            for minus in STEPS:
                vs = [family.values(e, c, plus, minus, POSITIONS, fold, opens_cache[i])
                      for i, e in enumerate(eps)]
                if any(v is None for v in vs):
                    continue
                usable += 225
                accs = [v[0] for v in vs]; los = [v[1] for v in vs]
                tmask = [mask([family.rule(op, v, x) for x in accs]) for op, v in RULES]
                mmask = [mask([family.rule(op, v, x) for x in los]) for op, v in RULES]
                for ti, tm in enumerate(tmask):
                    for mi, mm in enumerate(mmask):
                        if (tm & mm & full) == lab:
                            conforming.append((c, plus, minus, RULES[ti], RULES[mi], ti, mi))
    return conforming, usable


def global_index(c, plus, minus, ti, mi):
    return ((((SUB.index(c) * 5 + STEPS.index(plus)) * 5 + STEPS.index(minus)) * 15 + ti) * 15 + mi)


def main():
    t0 = time.perf_counter()
    s = prepare.load()
    train, seen, unseen = s['train'], s['heldout_seen_lengths'], s['heldout_unseen_lengths']
    labels = [e['label'] for e in train]
    module, registry, _ = build_module()
    folds = accum_scaffold.core_folds(registry)
    print('core folds with a CNT x CNT -> CNT signature:', folds, flush=True)

    tr_opens = [family.opens(e, POSITIONS) for e in train]
    un_opens = [family.opens(e, POSITIONS) for e in unseen]

    rows = []
    for fold in folds:
        prog, signals = accum_scaffold.stage_b_accum(module, registry, positions=POSITIONS,
                                                     fold=fold)
        val = family.validate(n=150, fold=fold, program=prog, registry=registry, episodes=train)
        t1 = time.perf_counter()
        conf, usable = sweep(fold, train, labels, tr_opens)
        row = {'fold': fold, 'space_size': space_size(prog),
               'usable_members_on_train': usable, 'conforming_on_train': len(conf),
               'simulator_validation': val, 'sweep_seconds': time.perf_counter() - t1}
        if conf:
            conf_sorted = sorted(conf, key=lambda x: global_index(x[0], x[1], x[2], x[5], x[6]))
            first = conf_sorted[0]
            c, plus, minus, tot, mn, ti, mi = first
            row['first_conforming'] = {
                'c': c, 'plus': plus, 'minus': minus, 'total_ok': list(tot), 'min_ok': list(mn),
                'global_enumeration_index': global_index(c, plus, minus, ti, mi),
                'fraction_through_enumeration': global_index(c, plus, minus, ti, mi) / space_size(prog)}
            # simulator's held-out accuracy over every conforming member
            spread = []
            for (c2, p2, m2, t2, o2, _, _) in conf:
                pred = [family.predict(e, c2, p2, m2, t2, o2, POSITIONS, fold, un_opens[i])
                        for i, e in enumerate(unseen)]
                spread.append(sum(1 for p, e in zip(pred, unseen) if p is not None and p == e['label'])
                              / len(unseen))
            row['conforming_heldout_unseen'] = {
                'min': min(spread), 'max': max(spread), 'mean': sum(spread) / len(spread),
                'at_1.000': sum(1 for x in spread if x == 1.0)}
            # ---- confirm on the REAL typed program, not the simulator ----
            names = [n_.name for n_ in prog.nodes]
            sel = {k: 0 for k in names}
            sel.update({'symbols': SUB.index(c), 'plus': STEPS.index(plus),
                        'minus': STEPS.index(minus), 'total_ok': ti, 'min_ok': mi})
            err = evaluate(prog, sel, examples(train), signals, registry, tolerance=1e-6)
            real = {'train_max_error': err, 'conforms_on_train': err is not None and err <= 1e-6}
            for name, eps in (('train', train), ('heldout_seen_lengths', seen),
                              ('heldout_unseen_lengths', unseen)):
                acc, per_len = accuracy(prog, sel, registry, eps)
                real[name] = {'accuracy': acc, 'per_length': per_len, **baselines(eps)}
            row['real_program_confirmation'] = real
        rows.append(row)
        print(f"  {fold}: conforming {len(conf)} / usable {usable} "
              f"(space {space_size(prog)}), validation mismatches {val['mismatches']}, "
              f"{row['sweep_seconds']:.0f}s"
              + (f", real unseen {row['real_program_confirmation']['heldout_unseen_lengths']['accuracy']:.4f}"
                 if conf else ''), flush=True)

    rep = {'stream': common.STREAM_POST_AUDIT, 'positions': POSITIONS,
           'template': 'stage_b + a second accumulator folding the prefix sums with X',
           'hole_candidates_source': 'tcn.operators.BINARY resolved at CNT x CNT -> CNT',
           'hole_candidates': folds,
           'folds_with_a_solution': [r['fold'] for r in rows if r['conforming_on_train']],
           'baselines_heldout_unseen': baselines(unseen),
           'seconds': time.perf_counter() - t0, 'rows': rows}
    out = os.path.join(HERE, 'out', 'q3_operators.json')
    json.dump(rep, open(out, 'w'), indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k != 'rows'}, indent=1))
    print('wrote', out)


if __name__ == '__main__':
    main()
