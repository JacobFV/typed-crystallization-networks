"""Does FINDINGS section 19's language number reproduce, and on which stream?

Runs the track's own build/eval path twice: once on the post-audit default
stream (what `final_eval.py` draws today) and once with `hardening='none'`,
which HANDOFF states preserves the pre-audit stream bit-identically.
"""
import sys, os, json, collections
HERE = '/home/brandonin/Documents/typed-crystallization-networks/research/language-capability'
sys.path.insert(0, HERE)
sys.path.insert(0, '/home/brandonin/Documents/typed-crystallization-networks')
import common, scaffolds
from tcn.generation import Host

_orig = Host.create
def patched(generator, **kw):
    cfg = dict(kw.get('configuration') or {})
    cfg['hardening'] = 'none'
    kw['configuration'] = cfg
    return _orig(generator, **kw)

from run_stage_b import build_module, accuracy, baselines

def evaluate(tag):
    sel = json.load(open(os.path.join(HERE, 'stage_b.json')))['enumeration']['selections']
    module, registry, frozen = build_module()
    program, _ = scaffolds.stage_b(module, registry)
    exported = program.harden(sel).pruned()
    pool = common.dataset(900, seed0=0, split='train')
    train = [e for e in pool if e['length'] in (2, 4, 6)][:24]
    test_pool = common.dataset(1500, seed0=100000, split='test')
    unseen = [e for e in test_pool if e['length'] not in (2, 4, 6)]
    r = {'stream': tag, 'n_train_lengths_246': len(train), 'n_unseen': len(unseen),
         'train_pool_lengths': sorted({e['length'] for e in pool})}
    if unseen:
        acc, per_len = accuracy(exported, {n.name: 0 for n in exported.nodes}, registry, unseen)
        b = baselines(unseen)
        r['unseen_accuracy'] = acc
        r['majority_baseline'] = max(b['constant_yes'], b['constant_no'])
        r['unseen_lengths'] = sorted({e['length'] for e in unseen})
    return r

print(json.dumps(evaluate('post-audit default (what final_eval.py draws today)'), indent=1))
Host.create = staticmethod(patched) if isinstance(Host.__dict__.get('create'), staticmethod) else classmethod(lambda cls, g, **kw: patched(g, **kw))
import importlib
importlib.reload(common)
print(json.dumps(evaluate("hardening='none' (pre-audit stream)"), indent=1))
