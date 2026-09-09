"""E6 (primary, arm B): model-based control against a crystallized exact program.

No policy is learned. Supervision fits the world/perception model; the only thing
reward has to supply is the sign of the reward model, which is one bit.
"""
import json, statistics, time
from concurrent.futures import ProcessPoolExecutor
import torch
from tcn.types import Value, product
from tcn.scaffold import F
from pl import (Cfg, Counter, Runner, joint_program, reference_returns, freeze_choices,
                SETTINGS, ACTIONS, mpc_rollout)

EVAL = 64

def model_exactness(counter, program, registry, n=EVAL, start=10000, horizon=4):
    ok_target = ok_gate = 0
    for k in range(n):
        h = counter.create(seed=0, index=start + k, split='test',
                           configuration=SETTINGS | {'horizon': horizon},
                           objective={'invert': bool((start + k) % 2)})
        v = h.view()
        inputs = {'bits': v.observations['bits'], 'goal': v.observations['goal'],
                  'action': Value.of(product(F, F), (1., 0.)), 'dt': Value.of(F, 1.)}
        out, _ = program.run(inputs, None, registry)
        z, world = out['probe'].decoded
        ok_target += int(int(z > .5) == int(h.records[0].probes['target'].decoded))
        ok_gate += int(int(world > .5) == int(h.records[0].probes['gate'].decoded))
    return ok_target / n, ok_gate / n

def identify_sign(counter, program, registry, k, horizon=4, seed=0):
    """Cost 2k environment episodes: greedy under each sign, keep the better."""
    scores = []
    for sign in (0, 1):
        tot = [mpc_rollout(counter, program, registry, sign, 700000 + j, horizon,
                           split='train', invert=bool(j % 2), seed=seed, plan_horizon=1)
               for j in range(k)]
        scores.append(statistics.fmean(tot))
    return (0 if scores[0] >= scores[1] else 1), scores

def one(spec):
    probe_episodes, seed = spec
    torch.set_num_threads(1)
    t0 = time.time(); c = Counter()
    prog, reg = joint_program(policy_init='zero', value_head='constant')
    cfg = Cfg(episodes=probe_episodes, horizon=4, seed=seed, lr=.04,
              w_probe=1., w_actor=0., w_value=0., w_entropy=.01)
    r = Runner(prog, reg, cfg, c)
    r.train(log_every=25)
    sel = freeze_choices(r)
    exported = r.model.export()
    supervised_episodes = c.episodes
    acc_t, acc_g = model_exactness(c, exported, r.model.registry)
    res = {'probe_episodes': probe_episodes, 'seed': seed, 'selections': sel,
           'supervised_env_episodes': supervised_episodes,
           'model_target_accuracy': acc_t, 'model_gate_accuracy': acc_g}
    for k in (1, 2, 4, 8):
        c2 = Counter()
        sign, scores = identify_sign(c2, exported, reg, k, seed=seed)
        cc = Counter()
        ret = statistics.fmean(mpc_rollout(cc, exported, reg, sign, 10000 + j, 4,
                                           split='test', invert=bool(j % 2), seed=0)
                               for j in range(EVAL))
        res[f'sign_k{k}'] = {'sign': sign, 'scores': scores,
                             'reward_env_episodes': c2.episodes, 'eval_return': ret}
    # horizon sweep with full-sequence enumeration where feasible
    sign = res['sign_k4']['sign']; res['horizon'] = {}
    for h, plan in ((4, 4), (8, 8), (16, 16), (32, 12)):
        cc = Counter(); t = time.time()
        ret = statistics.fmean(mpc_rollout(cc, exported, reg, sign, 10000 + j, h,
                                           split='test', invert=bool(j % 2), seed=0,
                                           plan_horizon=plan)
                               for j in range(16))
        res['horizon'][h] = {'plan_horizon': plan, 'sequences_enumerated_per_step': 2 ** plan,
                             'mean_return': ret, 'normalized': ret / h,
                             'wall_s': round(time.time() - t, 2)}
    res['total_env_episodes_including_eval'] = c.episodes
    res['wall_s'] = round(time.time() - t0, 1)
    return res

if __name__ == '__main__':
    jobs = [(n, s) for n in (25, 50, 100, 200) for s in range(8)]
    out = []
    with ProcessPoolExecutor(max_workers=10) as ex:
        for r in ex.map(one, jobs):
            out.append(r)
            print(f"probe{r['probe_episodes']:4d} seed{r['seed']} model_acc {r['model_target_accuracy']:.3f} "
                  f"mpc_eval_k4 {r['sign_k4']['eval_return']:.3f} {r['wall_s']}s", flush=True)
    json.dump(out, open('research/policy-learning/out/e6.json', 'w'), indent=1)
