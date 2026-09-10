"""Exhaustive family decision for the language scaffolds, over training episodes.

Built on `sim2.py` (which generalises §47's validated `family.py` by one
parameter).  What this adds over §47's `q3_operators.py` is what the probe
needs and that file did not report: the **best training accuracy** of a family,
not only its conforming members, and the same for the one-accumulator counting
scaffold §45 measured.

Usability follows `tcn.search.evaluate`: a member that raises on **any** training
episode is not a member of the searched space -- §47's convention, and why
`idiv` and `mod` have zero usable members.

Episodes are bitmasked (n <= 64), so a whole 680,625-member family is decided in
about half a second.  No held-out episode is read by `sweep`.
"""
from __future__ import annotations
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import sim2

STEPS, RULES, SUB = sim2.STEPS, sim2.RULES, sim2.SUB


def mask(bits):
    m = 0
    for i, b in enumerate(bits):
        if b:
            m |= 1 << i
    return m


def opens_cache(eps, positions):
    return [sim2.opens(e, positions) for e in eps]


def sweep(eps, labels, positions, opens, acc_fold='add', second_fold=None):
    """Decide the whole family on `eps`, in the scaffold's own enumeration order."""
    n = len(eps)
    full = (1 << n) - 1
    lab = mask(labels)
    two = second_fold is not None
    per_member = 225 if two else 15
    best_acc, best_member, tied, conforming, usable = -1, None, 0, 0, 0
    for ci, c in enumerate(SUB):
        for pi, plus in enumerate(STEPS):
            for mi, minus in enumerate(STEPS):
                vs = []
                for k, e in enumerate(eps):
                    v = sim2.values(e, c, plus, minus, positions, acc_fold,
                                    second_fold, opens[k])
                    if v is None:
                        vs = None
                        break
                    vs.append(v)
                if vs is None:
                    continue
                usable += per_member
                tmask = [mask([sim2.rule(op, v, x) for x, _ in vs]) for op, v in RULES]
                if not two:
                    for ti, tm in enumerate(tmask):
                        a = ((~(tm ^ lab)) & full).bit_count()
                        if a > best_acc:
                            best_acc, best_member, tied = a, (ci, pi, mi, ti), 1
                        elif a == best_acc:
                            tied += 1
                        conforming += (a == n)
                else:
                    fmask = [mask([sim2.rule(op, v, y) for _, y in vs]) for op, v in RULES]
                    for ti, tm in enumerate(tmask):
                        for fi, fm in enumerate(fmask):
                            a = ((~((tm & fm) ^ lab)) & full).bit_count()
                            if a > best_acc:
                                best_acc, best_member, tied = a, (ci, pi, mi, ti, fi), 1
                            elif a == best_acc:
                                tied += 1
                            conforming += (a == n)
    space = len(SUB) * 5 * 5 * per_member
    return {'acc_fold': acc_fold, 'second_fold': second_fold,
            'space_size': space, 'evaluated': space, 'exhausted': True,
            'certificate': 'complete', 'usable_members_on_train': usable,
            'best_train_accuracy': (best_acc / n) if best_acc >= 0 else None,
            'best_train_hits': best_acc if best_acc >= 0 else None,
            'members_tied_at_best_train': tied,
            'conforming_on_train': conforming,
            'best_member': list(best_member) if best_member else None,
            'best_member_readable': _readable(best_member, two)}


def _readable(m, two):
    if m is None:
        return None
    if not two:
        ci, pi, mi, ti = m
        return {'c': SUB[ci], 'plus': STEPS[pi], 'minus': STEPS[mi], 'rule': list(RULES[ti])}
    ci, pi, mi, ti, fi = m
    return {'c': SUB[ci], 'plus': STEPS[pi], 'minus': STEPS[mi],
            'total_ok': list(RULES[ti]), 'fold_ok': list(RULES[fi])}


def selections(program, member):
    sel = {nd.name: 0 for nd in program.nodes}
    if len(member) == 4:
        ci, pi, mi, ti = member
        sel.update({'symbols': ci, 'plus': pi, 'minus': mi, 'answer': ti})
    else:
        ci, pi, mi, ti, fi = member
        sel.update({'symbols': ci, 'plus': pi, 'minus': mi, 'total_ok': ti, 'min_ok': fi})
    return sel


def accuracy_on(eps, labels, positions, member, acc_fold='add', second_fold=None):
    """Score one member on any split.  Held-out use only, never for selection."""
    op = opens_cache(eps, positions)
    two = second_fold is not None
    ok = 0
    for i, e in enumerate(eps):
        ci, pi, mi = member[0], member[1], member[2]
        v = sim2.values(e, SUB[ci], STEPS[pi], STEPS[mi], positions, acc_fold,
                        second_fold, op[i])
        if v is None:
            continue
        p = sim2.rule(*RULES[member[3]], v[0])
        if two:
            p = p and sim2.rule(*RULES[member[4]], v[1])
        ok += (p == labels[i])
    return ok / len(eps)


def global_index(member, two):
    if not two:
        ci, pi, mi, ti = member
        return ((ci * 5 + pi) * 5 + mi) * 15 + ti
    ci, pi, mi, ti, fi = member
    return ((((ci * 5 + pi) * 5 + mi) * 15 + ti) * 15 + fi)
