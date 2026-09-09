"""Run a named set of arms across seeds and write raw JSON. Nothing is aggregated here."""
import json, os, sys, time, statistics
from concurrent.futures import ProcessPoolExecutor
import torch
from pl import Cfg, Counter, Runner, joint_program, reference_returns, grad_snr, logit_trace, freeze_choices

EVAL_N = 64

def build(arm, seed=0):
    init = arm.get('policy_init', 'zero')
    if init == 'noise': init = f'noise:{seed}'
    prog, reg = joint_program(policy_init=init,
                              value_head=arm.get('value_head', 'constant'),
                              pin_choices=arm.get('pin_choices'),
                              readout='external' if arm.get('cfg', {}).get('external_readout') else 'program')
    return prog, reg

def one(spec):
    name, arm, seed = spec
    torch.set_num_threads(1)
    t0 = time.time()
    c = Counter()
    cfgkw = dict(arm.get('cfg', {})); cfgkw['seed'] = seed
    prog, reg = build(arm, seed)
    cfg = Cfg(**cfgkw)
    r = Runner(prog, reg, cfg, c)
    result = {'arm': name, 'seed': seed}
    if arm.get('snr_at_init'):
        result['snr_init'] = grad_snr(r, 64, start=900000)
        result['logits_init'] = logit_trace(r, 32, start=900000)
    stages = arm.get('stages')
    if stages:
        curves = []
        for si, stage in enumerate(stages):
            for k, v in stage.get('cfg', {}).items(): setattr(r.cfg, k, v)
            if stage.get('freeze'):
                result['frozen_selections'] = freeze_choices(r)
                result[f'probe_eval_before_stage{si}'] = r.evaluate(EVAL_N)
            if stage.get('reset_optimizer'):
                r.optimizer = torch.optim.Adam(r.params, lr=r.cfg.lr)
            r.history = []
            r.train(log_every=stage['cfg'].get('log_every', 50))
            curves.append(r.history)
            result[f'stage{si}_env_episodes'] = c.episodes
            result[f'stage{si}_eval'] = r.evaluate(EVAL_N)
            if stage.get('snr_after'):
                result[f'stage{si}_snr'] = grad_snr(r, 64, start=900000)
        result['curves'] = curves
    else:
        r.train(log_every=arm.get('log_every', 100))
        result['curve'] = r.history
    if arm.get('snr_after'):
        result['snr_trained'] = grad_snr(r, 64, start=900000)
        result['logits_trained'] = logit_trace(r, 32, start=900000)
    result['train_env_episodes'] = c.episodes
    result['eval'] = r.evaluate(EVAL_N)
    result['choices'] = r.choice_report()
    result['total_env_episodes'] = c.episodes
    result['env_steps'] = c.steps
    result['wall_s'] = round(time.time() - t0, 1)
    return result

def main(path, arms, seeds):
    jobs = [(n, a, s) for n, a in arms.items() for s in range(seeds)]
    out = []
    with ProcessPoolExecutor(max_workers=int(os.environ.get('WORKERS', 10))) as ex:
        for res in ex.map(one, jobs):
            out.append(res)
            print(f"{res['arm']:38s} seed {res['seed']} eval {res['eval']:.3f} "
                  f"env_ep {res['total_env_episodes']:6d} {res['wall_s']}s", flush=True)
    json.dump(out, open(path, 'w'), indent=1)
    # references on the same evaluation episodes
    c = Counter()
    refs = reference_returns(c, range(10000, 10000 + EVAL_N))
    json.dump(refs, open(path.replace('.json', '_refs.json'), 'w'), indent=1)
    print('references', refs)
    return out
