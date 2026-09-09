"""Every table in RESULTS.md, regenerated from `out/*.json`."""
import glob,json,statistics,sys
ROOT='/home/brandonin/Documents/typed-crystallization-networks'
OUT=ROOT+'/research/credit-assignment/out'

def load(pattern):
    return [json.load(open(p)) for p in sorted(glob.glob(f'{OUT}/{pattern}'))]

def arms():
    rows={}
    for d in load('arm_*.json'):
        if d['seed']>=90:continue
        rows.setdefault(d['arm'],[]).append(d)
    table=[]
    for arm,runs in sorted(rows.items()):
        det=[r['final']['mean'] for r in runs]
        sto=[r['final_stochastic']['mean'] for r in runs if 'final_stochastic' in r]
        table.append({'arm':arm,'seeds':len(runs),'episodes':runs[0]['episodes'],
                      'eval_deterministic':round(statistics.fmean(det),4),
                      'sd':round(statistics.pstdev(det),4) if len(det)>1 else 0.,
                      'eval_stochastic':round(statistics.fmean(sto),4) if sto else None,
                      'seeds_at_1.00':sum(1 for x in det if x>=0.999),
                      'seeds_above_0.25':sum(1 for x in det if x>0.25),
                      'slot_selected':[r['program']['candidate'].get('next_slot') for r in runs],
                      'logits':[{k:max(range(4),key=lambda i:v[i]) for k,v in r['program']['logits'].items()} for r in runs],
                      'logstd_slot':[r['program'].get('logstd_slot') for r in runs],
                      'reward_rate':reward_rate(runs),
                      'env_episodes':int(statistics.fmean(r['environment_episodes'] for r in runs)),
                      'minutes':round(statistics.fmean(r['seconds'] for r in runs)/60,1)})
    return table

def reward_rate(runs):
    rows={}
    for run in runs:
        for h in run['history']:
            if 'reward_rate' in h:rows.setdefault(h['episode'],[]).append(h['reward_rate'])
    return {e:round(statistics.fmean(v),3) for e,v in sorted(rows.items())}

def curves():
    rows={}
    for d in load('arm_*.json'):
        if d['seed']>=90:continue
        for point in d.get('curve',()):
            rows.setdefault((d['arm'],point['episode']),[]).append(point['eval']['mean'])
    out={}
    for (arm,ep),values in sorted(rows.items()):
        out.setdefault(arm,{})[ep]=round(statistics.fmean(values),3)
    return out

def macros():
    rows={}
    for d in load('macro_*.json'):
        if d['seed']>=90:continue
        rows.setdefault('+'.join(d['library']),[]).append(d)
    table=[]
    for library,runs in sorted(rows.items()):
        det=[r['final']['mean'] for r in runs]
        table.append({'library':library,'seeds':len(runs),'episodes':runs[0]['episodes'],
                      'eval':round(statistics.fmean(det),4),
                      'sd':round(statistics.pstdev(det),4) if len(det)>1 else 0.,
                      'selected':[r['selected'] for r in runs],
                      'chose_sweep':sum(1 for r in runs if r['selected']=='sweep'),
                      'env_episodes':int(statistics.fmean(r['environment_episodes'] for r in runs)),
                      'costs':next((r['costs'] for r in runs if 'costs' in r),None)})
    return table

VERBS=('wait','look','dial','commit')

def markdown(report):
    out=[]
    refs=report.get('refs')
    if refs:
        out.append('### baselines\n')
        out.append('| policy | mean return | sd | solved / n |')
        out.append('|---|---|---|---|')
        for r in refs['policies']:
            out.append(f"| `{r['policy']}` | {r['mean']:.4f} | {r['sd']:.3f} | {r['solved']}/{r['episodes']} |")
        out.append('')
        out.append('### horizon sweep\n')
        out.append('| H | oracle | commit_now (myopic) | uniform_random | neutral_typed |')
        out.append('|---|---|---|---|---|')
        for r in refs['horizon_sweep']:
            out.append(f"| {r['horizon']} | {r['oracle']:.4f} | {r['commit_now']:.4f} | {r['uniform_random']:.4f} | {r['neutral_typed']:.4f} |")
        out.append('')
    out.append('### learning arms\n')
    out.append('| arm | seeds | episodes | eval (deterministic) | sd | eval (stochastic) | seeds at 1.00 | env episodes |')
    out.append('|---|---|---|---|---|---|---|---|')
    for r in report['arms']:
        st='--' if r['eval_stochastic'] is None else f"{r['eval_stochastic']:.4f}"
        out.append(f"| `{r['arm']}` | {r['seeds']} | {r['episodes']} | **{r['eval_deterministic']:.4f}** | {r['sd']:.3f} | {st} | {r['seeds_at_1.00']}/{r['seeds']} | {r['env_episodes']} |")
    out.append('')
    out.append('### training reward rate, mean over seeds (fraction of training episodes rewarded)\n')
    eps=sorted({e for r in report['arms'] for e in r['reward_rate']})
    out.append('| arm | '+' | '.join(str(e) for e in eps)+' |')
    out.append('|---'*(len(eps)+1)+'|')
    for r in report['arms']:
        out.append(f"| `{r['arm']}` | "+' | '.join(f"{r['reward_rate'][e]:.2f}" if e in r['reward_rate'] else '--' for e in eps)+' |')
    out.append('')
    out.append('### held-out evaluation curve, deterministic (mean over seeds)\n')
    episodes=sorted({e for c in report['curves'].values() for e in c})
    out.append('| arm | '+' | '.join(str(e) for e in episodes)+' |')
    out.append('|---'*(len(episodes)+1)+'|')
    for arm,c in sorted(report['curves'].items()):
        out.append(f'| `{arm}` | '+' | '.join(f'{c[e]:.3f}' if e in c else '--' for e in episodes)+' |')
    out.append('')
    if report.get('macros'):
        out.append('### composite actions\n')
        out.append('| module library | seeds | eval | sd | chose `sweep` | env episodes |')
        out.append('|---|---|---|---|---|---|')
        for r in report['macros']:
            out.append(f"| `{r['library']}` | {r['seeds']} | **{r['eval']:.4f}** | {r['sd']:.3f} | {r['chose_sweep']}/{r['seeds']} | {r['env_episodes']} |")
        out.append('')
    out.append('### what each arm selected\n')
    for r in report['arms']:
        out.append(f"* `{r['arm']}`: next_slot {r['slot_selected']}, log sigma(slot) {r['logstd_slot']}, argmax verb per situation {r['logits']}")
    return '\n'.join(out)

def main():
    report={'arms':arms(),'curves':curves(),'macros':macros()}
    for name in ('analysis','refs','mpc','perception','equivalence','session_equivalence'):
        try:report[name]=json.load(open(f'{OUT}/{name}.json'))
        except FileNotFoundError:report[name]=None
    for row in report['arms']:
        row['logits']=[{k:VERBS[v] for k,v in d.items()} for d in row['logits']]
    open(f'{OUT}/aggregate.json','w').write(json.dumps(report,indent=2,default=str))
    text=markdown(report)
    open(f'{OUT}/tables.md','w').write(text)
    print(text)

if __name__=='__main__':main()
