"""Per-term gradient norms and pairwise conflict for the joint objective.

Re-implements JointTrainer.episode's loss assembly (tcn/training.py) so each
weighted term is kept as a separate tensor, then measures per-term gradient
norms on the shared trainable parameters and the cosine between every pair.
"""
import torch

from tcn.training import JointTrainer, target_loss
from tcn.generation import Host
from tcn.policy import bind_action
from tcn.learning import tensor


class InstrumentedTrainer(JointTrainer):
    def terms(self, index, split='train'):
        c = self.config
        goal = c.objectives[index % len(c.objectives)]
        host = Host.create(c.generator, seed=c.seed, index=index, split=split,
                           configuration=c.generator_config | {'horizon': c.horizon}, objective=goal)
        memory = None; previous = 0; executed = None; rows = []
        for t in range(c.horizon):
            view = host.view(); inputs = self.inputs(view, previous, executed)
            out, _, _ = self.model(inputs, memory, return_trace=True)
            logits = out['policy'].flatten()
            allowed = torch.tensor([a.verb in view.available_actions for a in c.action_templates],
                                   dtype=torch.bool, device=logits.device)
            distribution = torch.distributions.Categorical(logits=logits.masked_fill(~allowed, float('-inf')))
            choice = distribution.sample()
            i = int(choice)
            action, alogp, aent = bind_action(c.action_templates[i], i, c.action_bindings, out)
            pred_inputs = self.inputs(view, i, action)
            predicted, newmemory, _ = self.model(pred_inputs, memory, return_trace=True)
            record = host.step((action,), c.dt)
            reward = sum(v.decoded for v in record.reward_components.values())
            rows.append({'logp': distribution.log_prob(choice) + alogp,
                         'entropy': distribution.entropy() + aent,
                         'value': out['value'].reshape(()),
                         'prediction': predicted['prediction'].flatten(),
                         'probe': predicted.get('probe'),
                         'logits': logits,
                         'reward': float(reward)})
            memory = newmemory; previous = i; executed = action
            if record.done:
                break
        returns = []; g = 0.
        for row in reversed(rows):
            g = row['reward'] + c.discount * g; returns.insert(0, g)
        pred_losses = []; probe_losses = []
        for t, row in enumerate(rows):
            offset = 0
            for target in c.targets:
                at = t + target.horizon; width = target.value(host.records[0]).type.width
                size = width * (2 if target.loss == 'gaussian' else 1)
                pred = row['prediction'][offset:offset + size]; offset += size
                if at < len(host.records):
                    truth = tensor(target.value(host.records[at]), pred.device) / target.scale
                    pred_losses.append(target_loss(target, pred, truth))
            if row['probe'] is not None:
                truth = torch.cat([tensor(target.value(host.records[t]), row['probe'].device) / target.scale
                                   for target in c.targets])
                probe_losses.append((row['probe'].flatten() - truth).square().mean())
        prediction = torch.stack(pred_losses).mean()
        probe = torch.stack(probe_losses).mean() if probe_losses else prediction * 0
        actor = torch.stack([-r['logp'] * (ret - r['value'].detach()) for r, ret in zip(rows, returns)]).mean()
        value = torch.stack([(r['value'] - ret).square() for r, ret in zip(rows, returns)]).mean()
        entropy = torch.stack([r['entropy'] for r in rows]).mean()
        terms = {
            'prediction': c.prediction_weight * prediction,
            'probe': c.probe_weight * probe,
            'actor': c.policy_weight * actor,
            'value': c.value_weight * value,
            'entropy': -c.entropy_weight * entropy,
            'mdl': c.mdl_weight * self.model.complexity(),
            'crystal': c.crystal_weight * self.model.entropy(),
        }
        stats = {'return': sum(r['reward'] for r in rows),
                 'value_mean': float(torch.stack([r['value'] for r in rows]).mean().detach()),
                 'return_mean': sum(returns) / len(returns),
                 'logit_absmax': float(max(l['logits'].abs().max() for l in rows).detach())}
        return terms, stats, rows, host

    def instrumented_step(self, index):
        terms, stats, rows, host = self.terms(index)
        active = [p for p in self.model.parameters() if p.requires_grad]
        names = [n for n, p in self.model.named_parameters() if p.requires_grad]
        grads = {}
        for k, v in terms.items():
            if isinstance(v, torch.Tensor) and v.requires_grad:
                g = torch.autograd.grad(v, active, retain_graph=True, allow_unused=True)
                grads[k] = [torch.zeros_like(p) if x is None else x for x, p in zip(g, active)]
            else:
                grads[k] = [torch.zeros_like(p) for p in active]
        flat = {k: torch.cat([x.flatten() for x in v]) for k, v in grads.items()}
        stats['grad_norm'] = {k: float(v.norm()) for k, v in flat.items()}
        keys = list(flat)
        stats['cosine'] = {}
        for a in range(len(keys)):
            for b in range(a + 1, len(keys)):
                x, y = flat[keys[a]], flat[keys[b]]
                den = x.norm() * y.norm()
                stats['cosine'][f'{keys[a]}|{keys[b]}'] = float(x @ y / den) if den > 0 else 0.
        # per-parameter-group breakdown for the two dominant terms
        groups = {}
        for name, g_pred, g_val, g_act in zip(names, grads['prediction'], grads['value'], grads['actor']):
            head = name.split('.')[-1]
            groups[head] = {'prediction': float(g_pred.norm()), 'value': float(g_val.norm()),
                            'actor': float(g_act.norm())}
        stats['by_parameter'] = groups
        total = sum(terms.values())
        self.optimizer.zero_grad(); total.backward()
        stats['clipped_from'] = float(torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.))
        self.optimizer.step()
        stats['loss'] = {k: float(v.detach()) for k, v in terms.items()}
        return stats
