"""Aggregate raw arm JSON into the tables used in RESULTS.md."""
import json, statistics, sys

def load(p): return json.load(open(p))

def table(path, extra=()):
    rows = {}
    for r in load(path):
        rows.setdefault(r['arm'], []).append(r)
    out = []
    for name, rs in rows.items():
        ev = [r['eval'] for r in rs]
        rec = {'arm': name, 'seeds': len(rs),
               'eval_mean': round(statistics.fmean(ev), 3),
               'eval_sd': round(statistics.pstdev(ev), 3),
               'seeds_at_4': sum(1 for x in ev if x >= 3.99),
               'seeds_ge_3.8': sum(1 for x in ev if x >= 3.8),
               'env_episodes': int(statistics.fmean(r['total_env_episodes'] for r in rs)),
               'env_steps': int(statistics.fmean(r['env_steps'] for r in rs)),
               'wall_s': round(statistics.fmean(r['wall_s'] for r in rs), 1)}
        sel = [tuple(sorted(r['choices']['selections'].items())) for r in rs]
        rec['selections'] = {str(s): sel.count(s) for s in set(sel)}
        for k in extra:
            vals = [r[k] for r in rs if k in r]
            if vals and isinstance(vals[0], (int, float)):
                rec[k] = round(statistics.fmean(vals), 4)
        if 'snr_init' in rs[0]:
            rec['snr_init_actor'] = round(statistics.fmean(r['snr_init']['actor']['snr'] for r in rs), 4)
            rec['snr_init_actor_eps'] = round(statistics.fmean(min(r['snr_init']['actor']['episodes_to_average'],1e9) for r in rs), 1)
            rec['gnorm_init_actor'] = round(statistics.fmean(r['snr_init']['actor']['mean_of_norms'] for r in rs), 5)
        if 'snr_trained' in rs[0]:
            rec['snr_trained_actor'] = round(statistics.fmean(r['snr_trained']['actor']['snr'] for r in rs), 4)
            rec['gnorm_trained_actor'] = round(statistics.fmean(r['snr_trained']['actor']['mean_of_norms'] for r in rs), 5)
        if 'logits_init' in rs[0]:
            rec['logit_maxabs_init'] = round(statistics.fmean(r['logits_init']['mean_maxabs'] for r in rs), 3)
        if 'logits_trained' in rs[0]:
            rec['logit_maxabs_trained'] = round(statistics.fmean(r['logits_trained']['mean_maxabs'] for r in rs), 3)
            rec['pmax_trained'] = round(statistics.fmean(r['logits_trained']['mean_pmax'] for r in rs), 3)
        for k in ('stage0_eval','stage1_eval','stage0_env_episodes','stage1_env_episodes',
                  'probe_eval_before_stage1'):
            vals = [r[k] for r in rs if k in r]
            if vals: rec[k] = round(statistics.fmean(vals), 3)
        out.append(rec)
    return sorted(out, key=lambda r: r['arm'])

def show(path, extra=()):
    t = table(path, extra)
    print('###', path)
    for r in t: print(json.dumps(r))
    return t

if __name__ == '__main__':
    for p in sys.argv[1:]: show(p)
