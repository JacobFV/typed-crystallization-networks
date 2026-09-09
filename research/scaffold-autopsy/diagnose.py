"""Track the internal state of the arithmetic scaffold across joint training:
value head output vs. actual returns, advantage sign, policy logit saturation,
hidden-layer weight scale, and per-term gradient norms.
"""
import json
import sys

import torch

from arith_joint import trainer
from instrument import InstrumentedTrainer


def snapshot(t):
    c = t.model.constants
    return {
        'value_bias': float(c['value_bias0']),
        'value_w_norm': float(torch.stack([c[f'value_w0_{i}'] for i in range(8)]).norm()),
        'hidden_w_norm': float(torch.stack([c[f'hidden_w{j}_{i}'] for j in range(8) for i in range(8)]).norm()),
        'hidden_w_absmax': float(max(abs(float(c[f'hidden_w{j}_{i}'])) for j in range(8) for i in range(8))),
        'policy_w_norm': float(torch.stack([c[f'policy_w{j}_{i}'] for j in range(2) for i in range(8)]).norm()),
        'prediction_w_norm': float(torch.stack([c[f'prediction_w{j}_{i}'] for j in range(2) for i in range(8)]).norm()),
    }


def main():
    episodes = int(sys.argv[1]) if len(sys.argv) > 1 else 320
    t = trainer(episodes=episodes, seed=0)
    t.__class__ = InstrumentedTrainer
    torch.manual_seed(0)
    log = []
    for i in range(episodes):
        s = t.instrumented_step(i)
        s.update(snapshot(t))
        log.append(s)
    for a in range(0, episodes, max(1, episodes // 10)):
        w = log[a:a + max(1, episodes // 10)]
        m = lambda k: sum(x[k] for x in w) / len(w)
        print(f"ep{a:4d} ret={m('return'):.2f} V={m('value_mean'):+.3f} G={m('return_mean'):.3f} "
              f"advantage={m('return_mean') - m('value_mean'):+.3f} "
              f"|logit|={m('logit_absmax'):.2f} Vbias={m('value_bias'):+.3f} "
              f"|Wh|={m('hidden_w_norm'):.2f} maxWh={m('hidden_w_absmax'):.2f} "
              f"|Wpol|={m('policy_w_norm'):.2f} |Wpred|={m('prediction_w_norm'):.2f} "
              f"cos(pred,actor)={sum(x['cosine']['prediction|actor'] for x in w) / len(w):+.3f} "
              f"cos(pred,value)={sum(x['cosine']['prediction|value'] for x in w) / len(w):+.3f} "
              f"gn={{p:{sum(x['grad_norm']['prediction'] for x in w) / len(w):.2f},"
              f"a:{sum(x['grad_norm']['actor'] for x in w) / len(w):.2f},"
              f"v:{sum(x['grad_norm']['value'] for x in w) / len(w):.2f}}} "
              f"clip={sum(x['clipped_from'] for x in w) / len(w):.2f}")
    with open('diagnose.json', 'w') as f:
        json.dump(log, f, default=str)


if __name__ == '__main__':
    main()
