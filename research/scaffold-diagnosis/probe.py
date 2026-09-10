"""The probe under test: per-node information diagnosis, then an operator sweep.

Two **separable** stages, exactly as pre-registered, so that "the sweep is doing
the work, not the diagnosis" can be measured rather than argued:

  * `stage_d` -- select the failed scaffold's best member on the training
    episodes, execute the **real typed program** under it, and compute per node
    the number of distinct values, its entropy, and its mutual information with
    the label.  Fire iff some node is constant, or carries zero label
    information while the label carries at least half a bit.
  * `stage_s` -- read the flagged site's operator signature off the registry,
    enumerate every core operator that resolves at that signature, substitute
    each, refit the scaffold's free parameters on the **training episodes
    only**, and rank by best training accuracy.

Nothing here reads a held-out episode, and nothing here reads `answers.py`.
Probes are supervision: the label is read to compute information and to fit, and
is never packed into a program input.
"""
from __future__ import annotations
import math, sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
ROOT = os.path.dirname(os.path.dirname(HERE)); sys.path.insert(0, ROOT)
import langfam, boolfam, cases as case_mod
from tcn.graph import Node, Program, Candidate

MI_ZERO = 0.0            # exact zero, as §47 reported it
H_LABEL_MIN = 0.5        # bits

# A decided family depends only on (task, stream, positions, acc_fold, second_fold),
# so the same sweep is shared across cases instead of being recomputed.  Purely a
# cache: it changes no number.
_SWEEP_CACHE = {}


def _lang_sweep(case, acc_fold, second_fold):
    key = (case['task'], case['stream'], case['positions'], acc_fold, second_fold)
    if key not in _SWEEP_CACHE:
        opens = langfam.opens_cache(case['train'], case['positions'])
        _SWEEP_CACHE[key] = langfam.sweep(case['train'], case['labels'],
                                          case['positions'], opens,
                                          acc_fold, second_fold)
    return _SWEEP_CACHE[key]


# ------------------------------------------------------------- information

def _entropy(counts, n):
    return -sum((c / n) * math.log2(c / n) for c in counts if c)


def information(values, labels):
    """(distinct, H(node), I(node;label)) in bits, over the training batch."""
    n = len(values)
    vx, vy, vxy = {}, {}, {}
    for v, y in zip(values, labels):
        vx[v] = vx.get(v, 0) + 1
        vy[y] = vy.get(y, 0) + 1
        vxy[(v, y)] = vxy.get((v, y), 0) + 1
    hx = _entropy(vx.values(), n)
    hy = _entropy(vy.values(), n)
    hxy = _entropy(vxy.values(), n)
    return len(vx), hx, hy, max(0.0, hx + hy - hxy)


# ---------------------------------------------------------------- stage D

def best_member(case):
    """The scaffold's best member on the training episodes.  Training only."""
    if case['adapter'] == 'lang':
        res = _lang_sweep(case, case['acc_fold'], case['second_fold'])
        sel = (None if res['best_member'] is None
               else langfam.selections(case['program'], res['best_member']))
        return res, sel
    res = boolfam.sweep(case['program'], case['registry'], case['train'])
    return res, boolfam.selections(case['program'], res['best_member'])


def input_dependent(program):
    """Which nodes can vary with the program's inputs at all.

    Static and label-free: a node is input-dependent iff its transitive source
    closure reaches a program input.  A node that reaches only constants is a
    *chosen constant* -- the scaffold's `plus`, `minus` and base-address nodes --
    and its constancy across a batch says nothing about the data.
    """
    inputs = {k for k, _ in program.inputs}
    dep = {}
    for nd in program.nodes:
        srcs = {s for c in nd.candidates for s in c.sources}
        dep[nd.name] = any(s in inputs or dep.get(s, False) for s in srcs)
    return dep


def node_table(case, selections):
    """Execute the real typed program under `selections`; one row per node."""
    prog, registry = case['program'], case['registry']
    dep = input_dependent(prog)
    per_node = {nd.name: [] for nd in prog.nodes}
    raised = 0
    for e in case['train']:
        inputs = ({'text': e['text']} if case['adapter'] == 'lang' else e['inputs'])
        try:
            _, _, trace = prog.execute(inputs, registry=registry, selections=selections)
        except Exception:
            raised += 1
            continue
        for nd in prog.nodes:
            v = trace.get(nd.name)
            per_node[nd.name].append(None if v is None else tuple(v.flat()))
    rows = []
    for nd in prog.nodes:
        vals = per_node[nd.name]
        if len(vals) != len(case['labels']):
            rows.append({'node': nd.name, 'depth': nd.depth, 'distinct': None,
                         'entropy_bits': None, 'information_bits': None,
                         'input_dependent': dep[nd.name],
                         'operator': nd.candidates[selections[nd.name]].operator.name,
                         'incomplete_trace': True})
            continue
        d, hx, hy, mi = information(vals, case['labels'])
        rows.append({'node': nd.name, 'depth': nd.depth, 'distinct': d,
                     'entropy_bits': hx, 'information_bits': mi,
                     'label_entropy_bits': hy, 'input_dependent': dep[nd.name],
                     'operator': nd.candidates[selections[nd.name]].operator.name,
                     'distinct_values': (sorted({v for v in vals})[:4] if d <= 4 else None)})
    return rows, raised


def stage_d(case, member_selections=None, member_note='best on training'):
    """Diagnosis.  Returns fired / site / the full per-node table.

    `member_selections` overrides the pre-registered "best member on the training
    episodes" rule; it exists only so the addendum arm can evaluate the same
    diagnosis at §47's own member and check that its recorded numbers reproduce.
    """
    if member_selections is not None:
        rows, raised = node_table(case, member_selections)
        return _decide(case, rows, raised, {'member_note': member_note},
                       member_selections)
    res, sel = best_member(case)
    if sel is None:
        # Rule D0: no member of the scaffold runs on the training batch.
        site = _raising_node(case)
        return {'fired': True, 'rule': 'D0 no usable member',
                'site': site, 'site_distinct': None,
                'best_member': res, 'nodes': []}
    rows, raised = node_table(case, sel)
    return _decide(case, rows, raised, {'best_member': res}, sel)


def _decide(case, rows, raised, extra, sel):
    hy = next((r['label_entropy_bits'] for r in rows if r.get('label_entropy_bits')), 0.0)

    def fires(r):
        return r['distinct'] == 1 or (r['information_bits'] == MI_ZERO
                                      and hy >= H_LABEL_MIN)

    fired = [r for r in rows if fires(r)]
    # the exploratory refinement: ignore nodes that cannot vary with the input
    fired_id = [r for r in fired if r.get('input_dependent')]
    order = {r['node']: i for i, r in enumerate(rows)}

    def pick(fs):
        if not fs:
            return None, None, None
        s = max(fs, key=lambda r: (r['depth'], order[r['node']]))
        return (s['node'], s['distinct'],
                'constant node' if s['distinct'] == 1 else 'zero label information')

    site, dist, rule = pick(fired)
    site_id, dist_id, rule_id = pick(fired_id)
    out = {'fired': bool(fired), 'rule': rule, 'site': site, 'site_distinct': dist,
           'selections': sel, 'label_entropy_bits': hy,
           'nodes': rows, 'episodes_that_raised': raised,
           'n_nodes': len(rows), 'n_fired_nodes': len(fired),
           'fired_nodes': [r['node'] for r in fired],
           'input_dependent_variant': {
               'fired': bool(fired_id), 'site': site_id, 'site_distinct': dist_id,
               'rule': rule_id, 'n_fired_nodes': len(fired_id),
               'fired_nodes': [r['node'] for r in fired_id]}}
    out.update(extra)
    return out


def _raising_node(case):
    """The earliest node that raises on the training batch (rule D0).

    "Does the prefix ending at node k run?" is monotone in k, so the first
    raising node is found by bisection over prefixes -- about nine executions
    for a 500-node scaffold rather than five hundred.
    """
    prog, registry = case['program'], case['registry']
    e = case['train'][0]
    inputs = ({'text': e['text']} if case['adapter'] == 'lang' else e['inputs'])

    def raises(k):
        sub = Program(prog.inputs, tuple(prog.nodes[:k]),
                      ((prog.nodes[k - 1].name, prog.nodes[k - 1].name),),
                      prog.constants)
        try:
            sub.validate(registry).execute(
                inputs, registry=registry,
                selections={x.name: 0 for x in sub.nodes})
            return False
        except Exception:
            return True

    lo, hi = 1, len(prog.nodes)
    if not raises(hi):
        return None
    while lo < hi:
        mid = (lo + hi) // 2
        if raises(mid):
            hi = mid
        else:
            lo = mid + 1
    return prog.nodes[lo - 1].name


# ---------------------------------------------------------------- stage S

def site_signature(case, site):
    nd = next(x for x in case['program'].nodes if x.name == site)
    op = nd.candidates[0].operator
    return op.name, op.inputs, op.output


def core_candidates(case, site):
    """Every core operator name that resolves at the site's signature.

    Read off `tcn.operators` by asking the registry to resolve each name; no
    list is curated and nothing is added to core.
    """
    from tcn.operators import BINARY, UNARY, LOGIC, COMPARE
    _, ins, out = site_signature(case, site)
    r = case['registry']
    names = []
    for name in sorted(set(BINARY) | set(UNARY) | set(LOGIC) | set(COMPARE) |
                       {'not', 'identity'}):
        try:
            r.resolve(name, tuple(ins), out, None)
        except Exception:
            continue
        names.append(name)
    return names


def stage_s(case, site, repair_family):
    """Sweep the site's operator over the core candidates.  Training only."""
    names = core_candidates(case, site)
    rows = []
    for name in names:
        rows.append(_fit_one(case, site, repair_family, name))
    usable = [r for r in rows if r['best_train_accuracy'] is not None]
    best = max((r['best_train_accuracy'] for r in usable), default=None)
    tie = sorted(r['operator'] for r in usable if r['best_train_accuracy'] == best)
    return {'site': site, 'repair_family': repair_family,
            'candidates': names, 'n_candidates': len(names),
            'rows': rows, 'best_train_accuracy': best,
            'prediction': tie[0] if tie else None, 'tie_set': tie,
            'solves_on_train': [r['operator'] for r in usable
                                if r['conforming_on_train']]}


def chain_hole(case, site):
    """Which of the two folds the site sits in: `acc`, `second`, or None."""
    if case['adapter'] != 'lang':
        return 'bool'
    if site.startswith('acc'):
        return 'acc'
    if site.startswith('lo'):
        return 'second'
    return None


def _fit_one(case, site, repair_family, op_name):
    if case['adapter'] == 'lang':
        acc_fold, second_fold = case['acc_fold'], case['second_fold']
        if repair_family == 'swap':
            hole = chain_hole(case, site)
            if hole == 'acc':
                acc_fold = op_name
            elif hole == 'second':
                second_fold = op_name
            else:
                return {'operator': op_name, 'best_train_accuracy': None,
                        'note': 'the site is not an accumulator: no swap defined'}
        elif repair_family == 'fold':
            if second_fold is not None:
                return {'operator': op_name, 'best_train_accuracy': None,
                        'note': 'the scaffold already carries a second accumulator'}
            second_fold = op_name
        res = _lang_sweep(case, acc_fold, second_fold)
        return {'operator': op_name, 'acc_fold': acc_fold, 'second_fold': second_fold,
                'best_train_accuracy': res['best_train_accuracy'],
                'members_tied_at_best_train': res['members_tied_at_best_train'],
                'conforming_on_train': res['conforming_on_train'],
                'usable_members_on_train': res['usable_members_on_train'],
                'space_size': res['space_size'],
                'best_member': res['best_member'],
                'best_member_readable': res['best_member_readable']}
    ext = boolfam.extend_site(case['program'], case['registry'], site, op_name)
    if not ext:
        return {'operator': op_name, 'best_train_accuracy': None,
                'note': 'no wiring resolves at this signature'}
    res = boolfam.sweep(case['program'], case['registry'], case['train'],
                        override={site: ext})
    return {'operator': op_name,
            'best_train_accuracy': res['best_train_accuracy'],
            'members_tied_at_best_train': res['members_tied_at_best_train'],
            'conforming_on_train': res['conforming_on_train'],
            'space_size': res['space_size'],
            'best_member': res['best_member']}


# ---------------------------------------------------------------- controls

def b_random(case, site, repair_family, sweep_result):
    """Uniform draw from the signature-matching candidate set."""
    names = sweep_result['candidates']
    solving = set(sweep_result['solves_on_train'])
    return {'n_candidates': len(names),
            'n_that_solve_on_train': len(solving),
            'p_correct_uniform_draw': (len(solving) / len(names)) if names else None,
            'candidates': names, 'solving': sorted(solving)}
