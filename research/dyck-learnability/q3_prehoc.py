"""Q3a and Q3b: is there a signal, available before knowing the answer, that points
at "you need an accumulator that is not a sum"?

The evidence this script is allowed to use is exactly what a system would have
after the failed run of section 45: the failed counting scaffold, its exhaustion
certificate (space 45,375, exhausted, 0 conforming, certificate `complete`), and
the 24 training episodes. Labels are used only as supervision -- to fit and to
score -- never as a program input.

**3a-1. Is the failed accumulator's output label-independent?**
Mutual information between each node the failed scaffold already materialises
(`m0`, `acc1..acc21`) and the label, over the 24 training episodes. A terminal
accumulator with **zero** mutual information is a mechanically detectable
statement that the quantity being accumulated carries no label signal at all on
this distribution -- which is a stronger and cheaper signal than the exhaustion
certificate, and needs no search.

**3a-2. Does a core reduction over the nodes the scaffold already computes
separate the labels?** The prefix sums are already nodes. Sweep the reductions
core declares (`reduce_min`, `reduce_max`, `sum`, `mean`, `count`) over that
existing node set, threshold each with the same `{eq,ge,le} x {-2..2}` grid stage
B searches, **fit on the 24 training episodes only**, and report held-out accuracy
beside the 0.5262 majority. Selection is by training accuracy with ties broken by
enumeration order -- never by the held-out split.

**3b. What does the failed family's error structure reveal?** Section 45 records
24 members tying at train 0.75 with held-out spread 0.474-0.703. Re-derive the tie
set, then ask where its errors sit: on episodes with `min_prefix < 0`, or
elsewhere.

Every number comes from the post-audit stream (`hardening='context_free_language'`,
pinned in `prepare.py`). The simulator is validated against the real typed
program first and the validation record is written into the output.
"""
from __future__ import annotations
import sys, os, json, math, time, collections
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
ROOT = os.path.dirname(os.path.dirname(HERE)); sys.path.insert(0, ROOT)
import common, prepare, family, accum_scaffold
from run_stage_b import build_module, baselines

POSITIONS = 22
STEPS = family.STEP_VALUES
RULES = family.RULES
SUB = family.SUB
COUNTING_C, COUNTING_PLUS, COUNTING_MINUS = 101, 1, -1   # section 45's counting program


def mutual_information(xs, ys):
    """MI in bits between a discrete feature and a boolean label."""
    n = len(xs)
    joint = collections.Counter(zip(xs, ys))
    px = collections.Counter(xs); py = collections.Counter(ys)
    mi = 0.
    for (x, y), c in joint.items():
        mi += (c / n) * math.log2((c / n) / ((px[x] / n) * (py[y] / n)))
    return mi


def prefix_sums(e, c, plus, minus, positions=POSITIONS):
    """m0, acc1..acc_{positions-1} -- the nodes the failed scaffold already has."""
    length = e['prompt_bytes']
    if c > length:
        return None
    symbols = length - c
    if not 0 <= symbols <= family.POS_HI:
        return None
    b = family.opens(e, positions)
    out = []; a = 0
    for i in range(positions):
        m = (plus if b[i] else minus) if i < symbols else 0
        a = a + m
        out.append(a)
    return out


REDUCTIONS = {'reduce_min': min, 'reduce_max': max, 'sum': sum,
              'mean': lambda v: sum(v) / len(v), 'count': len}


def main():
    t0 = time.perf_counter()
    s = prepare.load()
    train, seen, unseen = s['train'], s['heldout_seen_lengths'], s['heldout_unseen_lengths']
    module, registry, _ = build_module()
    prog, _ = accum_scaffold.stage_b_accum(module, registry, positions=POSITIONS, fold='min')
    val = family.validate(n=150, fold='min', program=prog, registry=registry, episodes=train)
    print('simulator validation:', val['members_checked'], 'checked,', val['mismatches'],
          'mismatches', flush=True)

    rep = {'stream': common.STREAM_POST_AUDIT, 'positions': POSITIONS,
           'evidence_available': ['failed counting scaffold', 'its exhaustion certificate',
                                  'the 24 training episodes'],
           'simulator_validation': val}

    # ---------------- 3a-1: label information in the existing nodes ----------------
    labels = [e['label'] for e in train]
    ps = [prefix_sums(e, COUNTING_C, COUNTING_PLUS, COUNTING_MINUS) for e in train]
    node_names = ['m0'] + [f'acc{i}' for i in range(1, POSITIONS)]
    mi = {}
    for i, name in enumerate(node_names):
        mi[name] = mutual_information([p[i] for p in ps], labels)
    rep['a1_node_label_information_bits'] = mi
    rep['a1_counting_program'] = {'c': COUNTING_C, 'plus': COUNTING_PLUS,
                                  'minus': COUNTING_MINUS,
                                  'note': "section 45's counting program: symbols = length - 101"}
    rep['a1_terminal_accumulator'] = {
        'node': node_names[-1], 'information_bits': mi[node_names[-1]],
        'distinct_values': sorted({p[-1] for p in ps}),
        'label_entropy_bits': mutual_information(labels, labels)}
    # the same question asked of the whole family, not just one member: what is the
    # most label-information any member's terminal accumulator carries?
    best = None
    for c in SUB:
        for plus in STEPS:
            for minus in STEPS:
                vs = [family.values(e, c, plus, minus, POSITIONS, 'min') for e in train]
                if any(v is None for v in vs):
                    continue
                m = mutual_information([v[0] for v in vs], labels)
                if best is None or m > best[0]:
                    best = (m, c, plus, minus)
    rep['a1_best_terminal_accumulator_information'] = {
        'information_bits': best[0], 'c': best[1], 'plus': best[2], 'minus': best[3],
        'label_entropy_bits': mutual_information(labels, labels)}
    # and of the running minimum, which is the quantity the scaffold does NOT have
    bestlo = None
    for c in SUB:
        for plus in STEPS:
            for minus in STEPS:
                vs = [family.values(e, c, plus, minus, POSITIONS, 'min') for e in train]
                if any(v is None for v in vs):
                    continue
                m = mutual_information([v[1] for v in vs], labels)
                if bestlo is None or m > bestlo[0]:
                    bestlo = (m, c, plus, minus)
    rep['a1_best_running_minimum_information'] = {
        'information_bits': bestlo[0], 'c': bestlo[1], 'plus': bestlo[2], 'minus': bestlo[3]}
    print('a1: terminal accumulator MI', round(mi[node_names[-1]], 6),
          '| best in family', round(best[0], 4), '| running min', round(bestlo[0], 4), flush=True)

    # ---------------- 3a-2: core reductions over the existing nodes ----------------
    def reduction_feature(eps, red, c, plus, minus):
        out = []
        for e in eps:
            p = prefix_sums(e, c, plus, minus)
            out.append(None if p is None else REDUCTIONS[red](p))
        return out

    a2 = []
    for red in REDUCTIONS:
        # selection is on TRAINING accuracy only, ties broken by enumeration order
        cands = []
        for c in SUB:
            for plus in STEPS:
                for minus in STEPS:
                    f = reduction_feature(train, red, c, plus, minus)
                    if any(v is None for v in f):
                        continue
                    for op, v in RULES:
                        pred = [family.rule(op, v, x) for x in f]
                        acc = sum(p == l for p, l in zip(pred, labels)) / len(labels)
                        cands.append((acc, c, plus, minus, op, v))
        if not cands:
            a2.append({'reduction': red, 'usable_members': 0})
            continue
        bestacc = max(x[0] for x in cands)
        tied = [x for x in cands if x[0] == bestacc]
        acc_, c, plus, minus, op, v = tied[0]
        row = {'reduction': red, 'usable_members': len(cands),
               'best_train_accuracy': bestacc, 'members_tied_at_best_train': len(tied),
               'selected': {'c': c, 'plus': plus, 'minus': minus, 'rule': [op, v]}}
        for name, eps in (('train', train), ('heldout_seen_lengths', seen),
                          ('heldout_unseen_lengths', unseen)):
            f = reduction_feature(eps, red, c, plus, minus)
            pred = [None if x is None else family.rule(op, v, x) for x in f]
            hit = sum(1 for p, e in zip(pred, eps) if p is not None and p == e['label'])
            row[name] = {'accuracy': hit / len(eps), **baselines(eps)}
        # held-out spread across the members tied at the best training accuracy:
        # the thing that decides whether training can pick among them
        spread = []
        for _, c2, p2, m2, o2, v2 in tied:
            f = reduction_feature(unseen, red, c2, p2, m2)
            pred = [None if x is None else family.rule(o2, v2, x) for x in f]
            spread.append(sum(1 for p, e in zip(pred, unseen) if p is not None and p == e['label'])
                          / len(unseen))
        row['tied_heldout_unseen_min'] = min(spread)
        row['tied_heldout_unseen_max'] = max(spread)
        row['tied_heldout_unseen_mean'] = sum(spread) / len(spread)
        a2.append(row)
        print('a2', red, 'train', round(bestacc, 4), 'tied', len(tied),
              'unseen', round(row['heldout_unseen_lengths']['accuracy'], 4),
              'tied spread', round(min(spread), 4), round(max(spread), 4), flush=True)
    rep['a2_core_reductions_over_existing_nodes'] = a2

    # ---------------- 3b: the failed family's error structure ----------------
    # The counting family: answer = op(acc_last, v). Nothing else is available.
    cands = []
    for c in SUB:
        for plus in STEPS:
            for minus in STEPS:
                vs = [family.values(e, c, plus, minus, POSITIONS, 'min') for e in train]
                if any(v is None for v in vs):
                    continue
                accs = [v[0] for v in vs]
                for op, v in RULES:
                    pred = [family.rule(op, v, x) for x in accs]
                    acc = sum(p == l for p, l in zip(pred, labels)) / len(labels)
                    cands.append((acc, c, plus, minus, op, v, tuple(pred)))
    bestacc = max(x[0] for x in cands)
    tied = [x for x in cands if x[0] == bestacc]
    wrong = collections.Counter()
    for _, c, plus, minus, op, v, pred in tied:
        for j, (p, l) in enumerate(zip(pred, labels)):
            if p != l:
                wrong[j] += 1
    unseen_spread = []
    for _, c, plus, minus, op, v, _ in tied:
        vs = [family.values(e, c, plus, minus, POSITIONS, 'min') for e in unseen]
        pred = [None if x is None else family.rule(op, v, x[0]) for x in vs]
        unseen_spread.append(sum(1 for p, e in zip(pred, unseen) if p is not None and p == e['label'])
                             / len(unseen))
    mp = [common.min_prefix(e['string']) for e in train]
    rep['b_counting_family'] = {
        'usable_members': len(cands), 'members_conforming_on_train': sum(1 for x in cands if x[0] == 1.0),
        'best_train_accuracy': bestacc, 'members_tied_at_best_train': len(tied),
        'tied_heldout_unseen_min': min(unseen_spread), 'tied_heldout_unseen_max': max(unseen_spread),
        'tied_heldout_unseen_mean': sum(unseen_spread) / len(unseen_spread),
        'heldout_unseen_majority': baselines(unseen)['majority_constant'],
        'train_episodes_by_times_wrong_across_tied_members':
            {str(j): wrong.get(j, 0) for j in range(len(train))},
        'train_episode_min_prefix': {str(j): mp[j] for j in range(len(train))},
        'train_episode_label': {str(j): bool(labels[j]) for j in range(len(train))},
        'errors_on_min_prefix_negative_episodes':
            sum(wrong.get(j, 0) for j in range(len(train)) if mp[j] < 0),
        'errors_on_min_prefix_zero_episodes':
            sum(wrong.get(j, 0) for j in range(len(train)) if mp[j] >= 0),
        'label_equals_min_prefix_nonnegative_on_train':
            all((mp[j] >= 0) == labels[j] for j in range(len(train))),
        'counts_match_on_all_train': all(common.counts_match(e['string']) for e in train)}
    print('b: best train', bestacc, 'tied', len(tied), 'spread',
          round(min(unseen_spread), 4), round(max(unseen_spread), 4), flush=True)

    rep['seconds'] = time.perf_counter() - t0
    out = os.path.join(HERE, 'out', 'q3_prehoc.json')
    json.dump(rep, open(out, 'w'), indent=1)
    print('wrote', out)


if __name__ == '__main__':
    main()
