"""Q2: the differentiable path on the min-prefix scaffold, against its own control.

Two scaffolds, the same protocol, the same episodes, the same split:

  * `--scaffold dyck`     -- `dyck_scaffold.stage_b_dyck`, where section 45 proves
                             a solution EXISTS (an exhibited witness at 1.000 on
                             859 held-out episodes, and 110 conforming members
                             recovered by exhaustive enumeration inside a window);
  * `--scaffold counting` -- `research/language-capability/scaffolds.stage_b`, where
                             section 45 proves NO solution exists (45,375
                             evaluated, exhausted, 0 conforming, certificate
                             `complete`).

The second is the control. A gradient path that conforms at the same rate on both
is not selecting the running minimum, it is fitting noise, and that is the
comparison this arm exists to make.

`SoftProgram` is `tcn.learning.SoftProgram`, unmodified. The protocol follows the
language track's own `run_stage_b_grad.py`: Adam, lr 0.05, logits perturbed by
`0.01 * randn` because SoftProgram zero-initialises them, and `--tau-lt` widening
the `lt` surrogate exactly as that track's `stage_b_grad_tau` arm does (128.0),
which matters here because section 19 measured `lt`'s surrogate gradient as
exactly 0.0 at delta >= 17 and this scaffold operates at delta up to 22. The
`in{i}` nodes have a single candidate, so raising `temperatures[in{i}]` widens the
surrogate without flattening any choice distribution.

Conformance is the exact-execution test, not the relaxed loss: an argmax export
conforms iff `tcn.search.evaluate` gives max error <= 1e-6 on all 24 training
episodes. It is checked every `--check-every` steps so "steps to first conforming
export" is a measured number rather than an end-of-run snapshot.

`hardening` is pinned: the episodes come from `prepare.py`, which pins
`context_free_language` (post-audit).
"""
from __future__ import annotations
import sys, os, json, time, argparse, importlib.util, collections
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
ROOT = os.path.dirname(os.path.dirname(HERE)); sys.path.insert(0, ROOT)
import torch
import common, prepare
from run_stage_b import build_module, examples, accuracy, baselines
from dyck_scaffold import stage_b_dyck
from tcn.learning import SoftProgram, tensor
from tcn.search import space_size, evaluate

_spec = importlib.util.spec_from_file_location(
    'lc_scaffolds', os.path.join(ROOT, 'research', 'language-capability', 'scaffolds.py'))
lc = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(lc)

STEPS = (-2, -1, 0, 1, 2)
RULES = [(op, v) for v in (-2, -1, 0, 1, 2) for op in ('eq', 'ge', 'le')]
SUB = list(range(0, 121))


def node_bools(program, sel, registry, eps, names):
    """Exact boolean value of the named nodes over `eps`, by reading the trace."""
    out = {n: [] for n in names}
    for e in eps:
        try:
            _, _, trace = program.execute({'text': e['text']}, registry=registry, selections=sel)
            for n in names:
                out[n].append(bool(round(trace[n].flat()[0])))
        except Exception:
            for n in names:
                out[n].append(None)
    return out


def classify_readout(vals, truth):
    """informative / correct / vacuous, per PREREGISTRATION.md."""
    clean = [v for v in vals if v is not None]
    if not clean:
        return {'status': 'unrunnable', 'true_rate': None, 'agreement_with_truth': None}
    rate = sum(clean) / len(clean)
    agree = sum(1 for v, t in zip(vals, truth) if v is not None and v == t) / len(clean)
    constant = rate in (0.0, 1.0)
    status = 'vacuous' if constant else ('correct' if agree == 1.0 else 'informative')
    return {'status': status, 'true_rate': rate, 'agreement_with_truth': agree,
            'constant': constant}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scaffold', choices=('dyck', 'counting'), required=True)
    ap.add_argument('--positions', type=int, default=22)
    ap.add_argument('--seeds', type=int, default=8)
    ap.add_argument('--steps', type=int, default=600)
    ap.add_argument('--lr', type=float, default=0.05)
    ap.add_argument('--tau-lt', type=float, default=0.)
    ap.add_argument('--tau-eq', type=float, default=0.)
    ap.add_argument('--check-every', type=int, default=10)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    tag = f"{a.scaffold}_tau{int(a.tau_lt)}"
    out = a.out or os.path.join(HERE, 'out', f'q2_{tag}.json')

    module, registry, frozen = build_module()
    if a.scaffold == 'dyck':
        program, signals = stage_b_dyck(module, registry, positions=a.positions)
        multi = ('symbols', 'plus', 'minus', 'total_ok', 'min_ok')
    else:
        program, signals = lc.stage_b(module, registry, positions=a.positions)
        multi = ('symbols', 'plus', 'minus', 'answer')

    s = prepare.load()
    train_eps, seen_held, unseen = s['train'], s['heldout_seen_lengths'], s['heldout_unseen_lengths']
    ex = examples(train_eps)
    inputs = {'text': torch.stack([tensor(e['inputs']['text']) for e in ex])}
    targets = {'answer': torch.stack([tensor(e['targets']['answer']) for e in ex])}
    truth_min = [common.min_prefix(e['string']) >= 0 for e in unseen]

    print(f'{tag}: nodes {len(program.nodes)} space {space_size(program)} '
          f'seeds {a.seeds} steps {a.steps}', flush=True)
    runs = []
    for sd in range(a.seeds):
        torch.manual_seed(sd); torch.set_num_threads(1)
        model = SoftProgram(program, registry)
        with torch.no_grad():
            for p in model.choices:
                p.add_(torch.randn_like(p) * .01)
        if a.tau_lt:
            model.temperatures.update({f'in{i}': a.tau_lt for i in range(a.positions)})
        if a.tau_eq:
            for n in ('total_ok', 'min_ok', 'answer'):
                if n in model.temperatures:
                    model.surrogate_scale[n] = a.tau_eq
        opt = torch.optim.Adam(model.parameters(), lr=a.lr)
        t0 = time.perf_counter(); first_grads = None; first_conforming = None
        losses = []
        for step in range(a.steps):
            opt.zero_grad()
            _, _, trace = model(inputs, return_trace=True)
            loss = model.probe_loss(trace, targets, signals)
            loss.backward()
            if step == 0:
                first_grads = {k: float(model.choices[i].grad.abs().max())
                               for i, k in enumerate(n.name for n in program.nodes)
                               if k in multi}
            opt.step()
            if step % a.check_every == 0 or step == a.steps - 1:
                sel = model.selections()
                err = evaluate(program, sel, ex, signals, registry, tolerance=1e-6)
                if first_conforming is None and err is not None and err <= 1e-6:
                    first_conforming = step
                losses.append([step, float(loss.detach()),
                               None if err is None else float(err)])
        sel = model.selections()
        err = evaluate(program, sel, ex, signals, registry, tolerance=1e-6)
        rec = {'seed': sd, 'loss': float(loss.detach()), 'seconds': time.perf_counter() - t0,
               'final_train_max_error': None if err is None else float(err),
               'conforming': bool(err is not None and err <= 1e-6),
               'first_conforming_step': first_conforming,
               'first_step_choice_gradients': first_grads,
               'trajectory': losses}
        if a.scaffold == 'dyck':
            rec['chosen'] = {'c': SUB[sel['symbols']], 'plus': STEPS[sel['plus']],
                             'minus': STEPS[sel['minus']],
                             'total_ok': list(RULES[sel['total_ok']]),
                             'min_ok': list(RULES[sel['min_ok']])}
        else:
            rec['chosen'] = {'c': SUB[sel['symbols']], 'plus': STEPS[sel['plus']],
                             'minus': STEPS[sel['minus']],
                             'answer': list(RULES[sel['answer']])}
        for name, eps in (('train', train_eps), ('heldout_seen_lengths', seen_held),
                          ('heldout_unseen_lengths', unseen)):
            acc, per_len = accuracy(program, sel, registry, eps)
            rec[name] = {'accuracy': acc, 'per_length': per_len, **baselines(eps)}
        if a.scaffold == 'dyck':
            nb = node_bools(program, sel, registry, unseen, ('total_ok', 'min_ok'))
            rec['min_ok_readout'] = classify_readout(nb['min_ok'], truth_min)
            rec['total_ok_readout'] = classify_readout(nb['total_ok'],
                                                       [True] * len(unseen))
        runs.append(rec)
        print(f"  seed {sd} conforming={rec['conforming']} first={first_conforming} "
              f"chosen={rec['chosen']} unseen={rec['heldout_unseen_lengths']['accuracy']:.4f} "
              f"min_ok={rec.get('min_ok_readout', {}).get('status')} "
              f"grads={first_grads} {rec['seconds']:.0f}s", flush=True)

    accs = [r['heldout_unseen_lengths']['accuracy'] for r in runs]
    accs_sorted = sorted(accs)
    rep = {'stream': common.STREAM_POST_AUDIT, 'scaffold': a.scaffold,
           'scaffold_module': 'dyck_scaffold.stage_b_dyck' if a.scaffold == 'dyck'
                              else 'language-capability/scaffolds.stage_b',
           'solution_exists_in_family': a.scaffold == 'dyck',
           'positions': a.positions, 'space_size': space_size(program),
           'nodes': len(program.nodes), 'seeds': a.seeds, 'steps': a.steps, 'lr': a.lr,
           'tau_lt': a.tau_lt, 'tau_eq': a.tau_eq, 'check_every': a.check_every,
           'train_episodes': len(train_eps),
           'conforming_seeds': sum(r['conforming'] for r in runs),
           'conforming_rate': sum(r['conforming'] for r in runs) / a.seeds,
           'first_conforming_steps': [r['first_conforming_step'] for r in runs],
           'mean_heldout_unseen': sum(accs) / len(accs),
           'median_heldout_unseen': (accs_sorted[len(accs) // 2] if len(accs) % 2 else
                                     (accs_sorted[len(accs) // 2 - 1] +
                                      accs_sorted[len(accs) // 2]) / 2),
           'max_heldout_unseen': max(accs), 'min_heldout_unseen': min(accs),
           'baselines_heldout_unseen': baselines(unseen),
           'min_ok_status_counts': dict(collections.Counter(
               r['min_ok_readout']['status'] for r in runs if 'min_ok_readout' in r)),
           'runs': runs}
    json.dump(rep, open(out, 'w'), indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k != 'runs'}, indent=1), flush=True)
    print('wrote', out)


if __name__ == '__main__':
    main()
