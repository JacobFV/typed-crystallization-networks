"""Condense the two large raw instrumentation dumps into reviewable summaries."""
import json
import os

d = os.path.dirname(os.path.abspath(__file__))

src = os.path.join(d, 'instrument_96.json')
if os.path.exists(src):
    rows = json.load(open(src))
    out = {}
    for name, rs in rows.items():
        n = len(rs)
        out[name] = {
            'episodes': n,
            'mean_grad_norm': {k: sum(r['grad_norm'][k] for r in rs) / n for k in rs[0]['grad_norm']},
            'mean_cosine': {k: sum(r['cosine'][k] for r in rs) / n for k in rs[0]['cosine']},
            'mean_return': sum(r['return'] for r in rs) / n,
            'mean_logit_absmax': sum(r['logit_absmax'] for r in rs) / n,
            'mean_preclip_norm': sum(r['clipped_from'] for r in rs) / n,
            'fraction_clipped': sum(r['clipped_from'] > 5 for r in rs) / n,
        }
    json.dump(out, open(os.path.join(d, 'instrument_96_summary.json'), 'w'), indent=2)
    os.remove(src)

src = os.path.join(d, 'diagnose.json')
if os.path.exists(src):
    log = json.load(open(src))
    keys = ['return', 'value_mean', 'return_mean', 'logit_absmax', 'value_bias', 'value_w_norm',
            'hidden_w_norm', 'hidden_w_absmax', 'policy_w_norm', 'prediction_w_norm']
    small = [{**{k: r[k] for k in keys}, 'grad_norm': r['grad_norm'], 'cosine': r['cosine'],
              'clipped_from': r['clipped_from'], 'loss': r['loss']} for r in log]
    json.dump(small, open(os.path.join(d, 'diagnose_summary.json'), 'w'), indent=1)
    os.remove(src)

print('done')

# Downsample the long per-episode training histories to keep the record reviewable.
import glob

for path in sorted(glob.glob(os.path.join(d, 'fixed_ep*.json')) + glob.glob(os.path.join(d, 'repro_seed*.json'))):
    blob = json.load(open(path))
    h = blob.get('history') or []
    if len(h) <= 64:
        continue
    keep = ['episode', 'return', 'prediction_loss', 'policy_loss', 'value_loss', 'loss',
            'prediction_policy_gradient_cosine', 'gradient_norm']
    k = max(1, len(h) // 64)
    blocks = []
    for a in range(0, len(h), k):
        w = h[a:a + k]
        blocks.append({'from': a, 'n': len(w),
                       **{f: sum(x.get(f, 0.) for x in w) / len(w) for f in keep if f in w[0]}})
    blob['history'] = blocks
    blob['history_note'] = f'per-episode history averaged into blocks of {k} episodes'
    json.dump(blob, open(path, 'w'), indent=1, default=str)
print('downsampled')
