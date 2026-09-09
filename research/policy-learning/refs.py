"""Trivial references at every horizon used, on the same held-out episodes."""
import json
from pl import Counter, reference_returns
out = {}
for h in (4, 8, 16, 32):
    c = Counter()
    r = reference_returns(c, range(10000, 10064), horizon=h)
    out[h] = {'mean_return': r, 'normalized': {k: v / h for k, v in r.items()},
              'env_episodes_used': c.episodes}
json.dump(out, open('research/policy-learning/out/refs.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
