"""Does the actor gradient point at a reward-optimal candidate at all?

Mirrors the perception-ladder's steepest-descent measurement: accumulate the
gradient of each loss term with respect to the two 16-way choice logit vectors
and ask where the reward-optimal candidates rank in the descent direction.
Chance rank for one of {6,9} being top-1 is 2/16 = 0.125.
"""
import json, statistics, torch
from pl import Cfg, Counter, Runner, joint_program

OPT = {6, 9}

def measure(policy_init, n_episodes, seeds=8, w_probe=0.):
    rows = []
    for seed in range(seeds):
        c = Counter()
        prog, reg = joint_program(policy_init=policy_init)
        cfg = Cfg(episodes=1, seed=seed, w_probe=w_probe, w_actor=1., w_value=.5, w_entropy=.01)
        r = Runner(prog, reg, cfg, c)
        names = [n.name for n in r.model.program.nodes]
        idx = {nm: names.index(nm) for nm in ('relation', 'goal_relation')}
        acc = {t: {k: torch.zeros(16) for k in idx} for t in ('actor', 'probe')}
        for k in range(n_episodes):
            rw, rt, _ = r.rollout(k)
            actor, value, entropy, probe = r.loss_terms(rw, rt)
            for tname, term in (('actor', actor), ('probe', probe)):
                g = torch.autograd.grad(term, [r.model.choices[idx[nm]] for nm in idx],
                                        retain_graph=True, allow_unused=True)
                for nm, gi in zip(idx, g):
                    if gi is not None: acc[tname][nm] += gi.detach()
        row = {'seed': seed}
        for tname in acc:
            for nm in idx:
                d = -acc[tname][nm] / n_episodes           # descent direction
                order = torch.argsort(d, descending=True).tolist()
                row[f'{tname}.{nm}.top1'] = order[0]
                row[f'{tname}.{nm}.optimal_is_top1'] = int(order[0] in OPT)
                row[f'{tname}.{nm}.best_optimal_rank'] = min(order.index(6), order.index(9))
                row[f'{tname}.{nm}.norm'] = float(d.norm())
        rows.append(row)
    agg = {'policy_init': policy_init, 'episodes_averaged': n_episodes, 'seeds': seeds}
    for k in rows[0]:
        if k == 'seed': continue
        m = statistics.fmean(r[k] for r in rows)
        agg[k] = float(f'{m:.4g}')
    return agg

if __name__ == '__main__':
    out = []
    for init in ('zero', 'oracle'):
        for n in (1, 8, 64, 256):
            a = measure(init, n)
            out.append(a)
            print(json.dumps(a), flush=True)
    json.dump(out, open('research/policy-learning/out/e2_direction.json', 'w'), indent=1)
