"""Policy-learning diagnosis harness.

Self-contained: nothing under `tcn/` or `generators/` is modified. The training
loop below is a re-implementation of `tcn.training.JointTrainer.episode` with the
loss weights, the baseline, the batch size, the entropy schedule and the reward
shaping all made configurable, plus per-term gradient instrumentation. It was
checked against the stock trainer (see `parity.py`): with the shipped settings it
reproduces `examples/joint.py`'s recorded numbers.

Every environment episode is counted explicitly in `Counter`.
"""
from __future__ import annotations
import json, math, random, statistics
from dataclasses import dataclass, field
import torch

from tcn.types import BOOL, Value, product, floating
from tcn.generation import Host, Action
from tcn.graph import Program, Node, Candidate
from tcn.operators import Registry
from tcn.learning import SoftProgram, tensor
from tcn.scaffold import F

SETTINGS = {'depth': 1, 'table': 6, 'fixed_inputs': True}
ACTIONS = (Action('answer', arguments=(('value', Value.of(BOOL, False)),)),
           Action('answer', arguments=(('value', Value.of(BOOL, True)),)))


class Counter:
    """Environment episodes and environment steps actually consumed."""
    def __init__(self): self.episodes = 0; self.steps = 0
    def create(self, **kw):
        self.episodes += 1
        return Host.create('logic', **kw)
    def step(self, host, action, dt=1.):
        self.steps += 1
        return host.step((action,), dt)


# --------------------------------------------------------------------------
# programs
# --------------------------------------------------------------------------

def joint_program(policy_init='zero', value_head='constant', pin_choices=None,
                  readout='program'):
    """The `examples/joint.py` scaffold with the hand-supplied answer removed.

    policy_init: 'oracle' = the shipped constants (w0=-2,w1=2,b0=1,b1=-1);
                 'zero'   = all four at 0.0;  'noise' = N(0, 0.1) (seeded by caller).
    value_head:  'constant' = the shipped trainable scalar (state-independent);
                 'state'    = a*z + b*world + c*goal_encoded + d (state-dependent).
    pin_choices: {'relation': k, 'goal_relation': j} to remove the discrete search.
    readout:     'program' = logits from the typed relaxed nodes;
                 'external' = logits from a torch Linear on (z, world), built by
                              the trainer; the program then exposes them as a
                              `features` output instead of `policy`.
    """
    r = Registry()
    host = Host.create('logic', configuration=SETTINGS)
    names = ('bits', 'goal')
    inputs = tuple((k, host.view().observations[k].type) for k in names) + \
             (('action', product(F, F)), ('dt', F))
    nodes = []
    for i in range(2):
        op = r.resolve('project', (dict(inputs)['bits'],), parameters={'index': i})
        nodes.append(Node(f'bit_{i}', BOOL, (Candidate(op, ('bits',)),), 'observation', 1))
    pin = pin_choices or {}
    for name, sources, depth in [('relation', ('bit_0', 'bit_1'), 2),
                                 ('goal_relation', ('relation', 'goal'), 3)]:
        cands = tuple(Candidate(r.resolve(f'truth_{i}', (BOOL, BOOL)), sources) for i in range(16))
        nodes.append(Node(name, BOOL, cands, 'latent', depth, pin.get(name)))
    for name, source in [('z', 'goal_relation'), ('world', 'relation')]:
        nodes.append(Node(name, F, (Candidate(r.resolve('encode', (BOOL,), F), (source,)),), 'encoding', 4))
    nodes.append(Node('goal_f', F, (Candidate(r.resolve('encode', (BOOL,), F), ('goal',)),), 'encoding', 1))

    init = {'oracle': (-2., 2., 1., -1.), 'zero': (0., 0., 0., 0.)}.get(policy_init)
    if init is None:
        g = random.Random(policy_init)                     # 'noise:<seed>'
        init = tuple(g.gauss(0, .1) for _ in range(4))
    constants = [('w0', Value.of(F, init[0])), ('w1', Value.of(F, init[1])),
                 ('bias0', Value.of(F, init[2])), ('bias1', Value.of(F, init[3]))]
    outputs = [('prediction', 'prediction')]

    if readout == 'program':
        for i in range(2):
            nodes.append(Node(f'mul{i}', F, (Candidate(r.resolve('mul', (F, F)), ('z', f'w{i}')),), 'policy', 5))
            nodes.append(Node(f'logit{i}', F, (Candidate(r.resolve('add', (F, F)), (f'mul{i}', f'bias{i}')),), 'policy', 6))
        nodes.append(Node('policy', product(F, F), (Candidate(r.resolve('tuple', (F, F)), ('logit0', 'logit1')),), 'policy', 7))
        outputs.append(('policy', 'policy'))
    nodes.append(Node('features', product(F, F), (Candidate(r.resolve('tuple', (F, F)), ('z', 'world')),), 'policy', 5))
    outputs.append(('features', 'features'))
    nodes.append(Node('prediction', product(F, F), (Candidate(r.resolve('tuple', (F, F)), ('z', 'world')),), 'prediction', 5))
    outputs.append(('probe', 'prediction'))

    if value_head == 'constant':
        constants.append(('baseline', Value.of(F, 1.)))
        nodes.append(Node('value', product(F), (Candidate(r.resolve('tuple', (F,)), ('baseline',)),), 'value', 1))
    elif value_head == 'state':
        for k, v in [('vw0', .0), ('vw1', .0), ('vw2', .0), ('vb', 1.)]:
            constants.append((k, Value.of(F, v)))
        nodes.append(Node('vm0', F, (Candidate(r.resolve('mul', (F, F)), ('z', 'vw0')),), 'value', 5))
        nodes.append(Node('vm1', F, (Candidate(r.resolve('mul', (F, F)), ('world', 'vw1')),), 'value', 5))
        nodes.append(Node('vm2', F, (Candidate(r.resolve('mul', (F, F)), ('goal_f', 'vw2')),), 'value', 5))
        nodes.append(Node('va', F, (Candidate(r.resolve('add', (F, F)), ('vm0', 'vm1')),), 'value', 6))
        nodes.append(Node('vb2', F, (Candidate(r.resolve('add', (F, F)), ('va', 'vm2')),), 'value', 7))
        nodes.append(Node('vsum', F, (Candidate(r.resolve('add', (F, F)), ('vb2', 'vb')),), 'value', 8))
        nodes.append(Node('value', product(F), (Candidate(r.resolve('tuple', (F,)), ('vsum',)),), 'value', 9))
    else:
        constants.append(('baseline', Value.of(F, 0.)))
        nodes.append(Node('value', product(F), (Candidate(r.resolve('tuple', (F,)), ('baseline',)),), 'value', 1))
    outputs.append(('value', 'value'))

    trainable = tuple(k for k, _ in constants) if value_head != 'none' else \
                tuple(k for k, _ in constants if k != 'baseline')
    return Program(inputs, tuple(nodes), tuple(outputs), tuple(constants),
                   trainable_constants=trainable), r


# --------------------------------------------------------------------------
# reference policies
# --------------------------------------------------------------------------

def oracle_action(view):
    b = view.observations['bits'].decoded; g = view.observations['goal'].decoded
    return int((b[0] ^ b[1]) ^ g)

def reference_returns(counter, indices, horizon=4, split='test', seed=0, objectives=(False, True)):
    out = {k: [] for k in ('always_false', 'always_true', 'uniform', 'oracle')}
    rng = random.Random(12345)
    for n, i in enumerate(indices):
        for name in out:
            h = counter.create(seed=seed, index=i, split=split,
                               configuration=SETTINGS | {'horizon': horizon},
                               objective={'invert': objectives[n % len(objectives)]})
            total = 0.
            for t in range(horizon):
                v = h.view()
                a = {'always_false': 0, 'always_true': 1,
                     'uniform': rng.randrange(2), 'oracle': oracle_action(v)}[name]
                rec = counter.step(h, ACTIONS[a])
                total += sum(x.decoded for x in rec.reward_components.values())
                if rec.done: break
            out[name].append(total)
    return {k: statistics.fmean(v) for k, v in out.items()}


# --------------------------------------------------------------------------
# trainer
# --------------------------------------------------------------------------

@dataclass
class Cfg:
    episodes: int = 800
    horizon: int = 4
    seed: int = 0
    lr: float = .04
    discount: float = .95
    w_pred: float = 0.
    w_probe: float = 0.
    w_actor: float = 1.
    w_value: float = .5
    w_entropy: float = .01
    entropy_final: float | None = None     # linear schedule to this value
    batch: int = 1
    normalize_advantage: bool = False
    terminal_reward_only: bool = False
    exact_policy_gradient: bool = False    # privileged all-actions estimator
    train_choices: bool = True
    train_constants: bool = True
    choice_noise: float = 0.               # sigma on choice logits at init
    choice_bias12: bool = False            # the shipped `p[12]=2.` cancellation break
    external_readout: bool = False
    external_inputs: str = 'features'      # 'features' (z,world) | 'raw' (obs vector)
    external_hidden: int = 0               # >0 makes the external readout a tanh MLP
    index_offset: int = 0
    eval_every: int = 0
    objectives: tuple = (False, True)


class Runner:
    def __init__(self, program, registry, cfg: Cfg, counter: Counter):
        self.cfg = cfg; self.counter = counter
        torch.manual_seed(cfg.seed)
        self.model = SoftProgram(program, registry)
        with torch.no_grad():
            for node, p in zip(program.nodes, self.model.choices):
                if not p.requires_grad: continue
                if cfg.choice_bias12 and len(node.candidates) == 16: p[12] = 2.
                if cfg.choice_noise: p.add_(torch.randn_like(p) * cfg.choice_noise)
        self.head = None
        if cfg.external_readout:
            width = 2 if cfg.external_inputs == 'features' else 8
            if cfg.external_hidden:
                h = cfg.external_hidden
                self.head = torch.nn.Sequential(torch.nn.Linear(width, h), torch.nn.Tanh(),
                                                torch.nn.Linear(h, h), torch.nn.Tanh(),
                                                torch.nn.Linear(h, 2))
            else:
                self.head = torch.nn.Linear(width, 2)
                torch.nn.init.zeros_(self.head.weight); torch.nn.init.zeros_(self.head.bias)
        params = []
        if cfg.train_choices: params += [p for p in self.model.choices if p.requires_grad]
        if cfg.train_constants: params += list(self.model.constants.parameters())
        if self.head is not None: params += list(self.head.parameters())
        for p in self.model.parameters():
            p.requires_grad_(any(p is q for q in params))
        self.params = params
        self.optimizer = torch.optim.Adam(params, lr=cfg.lr)
        self.history = []

    # -- one episode ------------------------------------------------------
    def rollout(self, index, train=True, split='train', collect_grad=False):
        c = self.cfg
        invert = c.objectives[index % len(c.objectives)]
        host = self.counter.create(seed=c.seed, index=index, split=split,
                                   configuration=SETTINGS | {'horizon': c.horizon},
                                   objective={'invert': invert})
        rows = []; previous = 0
        types = dict(self.model.program.inputs)
        for t in range(c.horizon):
            view = host.view()
            values = {k: tensor(view.observations[k]) for k in ('bits', 'goal')}
            values['action'] = torch.nn.functional.one_hot(torch.tensor(previous), 2).float()
            values['dt'] = torch.tensor([1.])
            out, _ = self.model(values)
            if self.head is not None:
                feats = out['features'] if c.external_inputs == 'features' else \
                        torch.cat([values['bits'], values['goal'], values['action'], values['dt']])
                logits = self.head(feats)
            else:
                logits = out['policy'].flatten()
            dist = torch.distributions.Categorical(logits=logits)
            choice = dist.sample() if train else logits.argmax()
            i = int(choice)
            rec = self.counter.step(host, ACTIONS[i])
            reward = sum(x.decoded for x in rec.reward_components.values())
            # counterfactual reward for the other action, for the exact estimator
            other = 1. - reward
            rows.append({'logits': logits, 'logp': dist.log_prob(choice), 'action': i,
                         'entropy': dist.entropy(), 'value': out['value'].reshape(()),
                         'probe': out['probe'].flatten(), 'reward': float(reward),
                         'counterfactual': (reward, other) if i == 0 else (other, reward),
                         'target': float(host.records[0].probes['target'].decoded),
                         'gate': float(host.records[0].probes['gate'].decoded)})
            previous = i
            if rec.done: break
        if c.terminal_reward_only:
            for row in rows[:-1]: row['reward'] = 0.
        g = 0.; returns = []
        for row in reversed(rows):
            g = row['reward'] + c.discount * g; returns.insert(0, g)
        return rows, returns, host

    def loss_terms(self, rows, returns, advantage_stats=None):
        c = self.cfg
        adv = torch.stack([torch.tensor(float(r)) - row['value'].detach() for row, r in zip(rows, returns)])
        if c.normalize_advantage and advantage_stats is not None:
            mu, sd = advantage_stats
            adv = (adv - mu) / (sd + 1e-6)
        if c.exact_policy_gradient:
            # E_a[R(a)] under the current distribution, differentiated exactly.
            actor = -torch.stack([(torch.softmax(row['logits'], 0) *
                                   torch.tensor(row['counterfactual'])).sum() for row in rows]).mean()
        else:
            actor = torch.stack([-row['logp'] * a for row, a in zip(rows, adv)]).mean()
        value = torch.stack([(row['value'] - float(r)).square() for row, r in zip(rows, returns)]).mean()
        entropy = torch.stack([row['entropy'] for row in rows]).mean()
        probe = torch.stack([(row['probe'] -
                              torch.tensor([row['target'], row['gate']])).square().mean()
                             for row in rows]).mean()
        return actor, value, entropy, probe

    def total(self, rows, returns, w_entropy, advantage_stats=None):
        c = self.cfg
        actor, value, entropy, probe = self.loss_terms(rows, returns, advantage_stats)
        loss = c.w_actor * actor + c.w_value * value - w_entropy * entropy
        if c.w_probe: loss = loss + c.w_probe * probe
        if c.w_pred: loss = loss + c.w_pred * probe
        return loss, {'actor': actor, 'value': value, 'entropy': entropy, 'probe': probe}

    def train(self, log_every=50):
        c = self.cfg
        i = 0
        while i < c.episodes:
            n = min(c.batch, c.episodes - i)
            frac = i / max(1, c.episodes)
            w_ent = c.w_entropy if c.entropy_final is None else \
                    c.w_entropy + (c.entropy_final - c.w_entropy) * frac
            self.optimizer.zero_grad()
            batch_rows = [self.rollout(c.index_offset + i + k) for k in range(n)]
            stats = None
            if c.normalize_advantage:
                allad = [float(r) - float(row['value']) for rows, rets, _ in batch_rows
                         for row, r in zip(rows, rets)]
                stats = (statistics.fmean(allad),
                         statistics.pstdev(allad) if len(allad) > 1 else 1.)
            terms_log = []
            for rows, rets, _ in batch_rows:
                loss, terms = self.total(rows, rets, w_ent, stats)
                (loss / n).backward()
                terms_log.append({k: float(v.detach()) for k, v in terms.items()} |
                                 {'return': sum(r['reward'] for r in rows),
                                  'maxlogit': float(rows[0]['logits'].detach().abs().max())})
            gn = float(torch.nn.utils.clip_grad_norm_([p for p in self.params], 5.))
            self.optimizer.step()
            i += n
            if log_every and (i % log_every < n):
                self.history.append({'episode': i, 'grad_norm': gn, 'w_entropy': w_ent,
                                     **{k: statistics.fmean([t[k] for t in terms_log])
                                        for k in terms_log[0]}})
        return self.history

    # -- evaluation --------------------------------------------------------
    def evaluate(self, n=64, start=10000, split='test'):
        totals = []
        with torch.no_grad():
            for k in range(n):
                rows, _, _ = self.rollout(start + k, train=False, split=split)
                totals.append(sum(r['reward'] for r in rows))
        return statistics.fmean(totals)

    def choice_report(self):
        sel = self.model.selections()
        ent = {}
        for node, p in zip(self.model.program.nodes, self.model.choices):
            if len(node.candidates) > 1:
                q = torch.softmax(p, 0)
                ent[node.name] = float(-(q * q.clamp_min(1e-12).log()).sum())
        return {'selections': {k: v for k, v in sel.items() if k in ('relation', 'goal_relation')},
                'entropy': ent,
                'constants': {k: float(v) for k, v in self.model.constants.items()}}


# --------------------------------------------------------------------------
# instrumentation
# --------------------------------------------------------------------------

def grad_snr(runner, n=64, start=0):
    """SNR = ||mean g|| / mean ||g|| per weighted term, over n single episodes."""
    c = runner.cfg
    acc = {k: [] for k in ('actor', 'value', 'probe')}
    for k in range(n):
        rows, rets, _ = runner.rollout(start + k)
        actor, value, entropy, probe = runner.loss_terms(rows, rets)
        for name, term in (('actor', actor), ('value', value), ('probe', probe)):
            if not term.requires_grad:
                acc[name].append(torch.cat([torch.zeros_like(p).flatten() for p in runner.params]))
                continue
            g = torch.autograd.grad(term, runner.params, retain_graph=True, allow_unused=True)
            acc[name].append(torch.cat([(x if x is not None else torch.zeros_like(p)).flatten()
                                        for x, p in zip(g, runner.params)]))
    out = {}
    for name, gs in acc.items():
        G = torch.stack(gs)
        mean = G.mean(0).norm().item(); mag = G.norm(dim=1).mean().item()
        out[name] = {'mean_norm': mean, 'mean_of_norms': mag,
                     'snr': mean / mag if mag > 0 else 0.,
                     'episodes_to_average': (mag / mean) ** 2 if mean > 0 else float('inf')}
    return out


def logit_trace(runner, n=32, start=0):
    """Policy logit magnitude and action probability, without training."""
    out = []
    with torch.no_grad():
        for k in range(n):
            rows, _, _ = runner.rollout(start + k)
            l = rows[0]['logits']
            out.append({'maxabs': float(l.abs().max()),
                        'pmax': float(torch.softmax(l, 0).max())})
    return {'mean_maxabs': statistics.fmean(x['maxabs'] for x in out),
            'mean_pmax': statistics.fmean(x['pmax'] for x in out)}


# --------------------------------------------------------------------------
# staged training: supervision first, then reward
# --------------------------------------------------------------------------

def freeze_choices(runner):
    """Crystallize the discrete choices at their argmax; no internal gradients."""
    for node in runner.model.program.nodes:
        if len(node.candidates) > 1 and node.name not in runner.model.frozen:
            runner.model.freeze(node.name)
    runner.params = [p for p in runner.model.constants.parameters()] + \
                    (list(runner.head.parameters()) if runner.head is not None else [])
    for p in runner.model.parameters(): p.requires_grad_(False)
    for p in runner.params: p.requires_grad_(True)
    runner.optimizer = torch.optim.Adam(runner.params, lr=runner.cfg.lr)
    return runner.model.selections()


# --------------------------------------------------------------------------
# model-predictive control against a crystallized program
# --------------------------------------------------------------------------

def mpc_rollout(counter, program, registry, reward_sign, index, horizon, split='test',
                invert=False, seed=0, plan_horizon=None):
    """Enumerate action sequences against the exact frozen program.

    `program` must be frozen. It supplies `z` exactly; the reward model is
    `r(a, z) = [a == (z xor reward_sign)]`, one bit, and that bit is the only
    content reward has to supply once the model is exact.
    """
    h = counter.create(seed=seed, index=index, split=split,
                       configuration=SETTINGS | {'horizon': horizon},
                       objective={'invert': invert})
    plan_horizon = plan_horizon or horizon
    total = 0.; previous = 0
    for t in range(horizon):
        view = h.view()
        inputs = {k: view.observations[k] for k in ('bits', 'goal')}
        inputs['action'] = Value.of(product(F, F), (float(previous == 0), float(previous == 1)))
        inputs['dt'] = Value.of(F, 1.)
        out, _ = program.run(inputs, None, registry)
        z = out['probe'].decoded[0]
        # enumerate every action sequence of length min(plan_horizon, remaining)
        remaining = min(plan_horizon, horizon - t)
        best = None
        for code in range(2 ** remaining):
            seq = [(code >> k) & 1 for k in range(remaining)]
            # the model says the latent does not change with the action, so the
            # predicted return of a sequence is the sum of predicted step rewards
            value = sum(1. if a == (int(z > .5) ^ reward_sign) else 0. for a in seq)
            if best is None or value > best[0]: best = (value, seq)
        a = best[1][0]
        rec = counter.step(h, ACTIONS[a])
        total += sum(x.decoded for x in rec.reward_components.values())
        previous = a
        if rec.done: break
    return total
