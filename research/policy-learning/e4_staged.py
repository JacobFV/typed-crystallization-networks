"""E4 (primary): staged world model -> policy, against simultaneous training."""
from run_arms import main

H4 = dict(horizon=4, lr=.04, w_value=.5, w_entropy=.01)
def probe_stage(n): return {'cfg': dict(H4, episodes=n, w_probe=1., w_actor=0., w_value=0., log_every=25)}
def reward_stage(n, **kw): return {'freeze': True, 'reset_optimizer': True, 'snr_after': True,
                                   'cfg': dict(H4, episodes=n, w_probe=0., w_actor=1., w_value=.5,
                                               index_offset=500000, log_every=25, **kw)}
def reward_stage_nofreeze(n, **kw):
    d = reward_stage(n, **kw); d['freeze'] = False; return d

ARMS = {}
for n in (25, 50, 100, 200, 400):
    ARMS[f'S1_staged_probe{n}_then_reward800'] = {'policy_init': 'zero',
        'stages': [probe_stage(n), reward_stage(800)]}
ARMS['S2_staged_probe200_nofreeze'] = {'policy_init': 'zero',
    'stages': [probe_stage(200), reward_stage_nofreeze(800)]}
ARMS['S3_staged_probe200_reward_exactPG'] = {'policy_init': 'zero',
    'stages': [probe_stage(200), reward_stage(800, exact_policy_gradient=True)]}
ARMS['S4_staged_probe200_reward_statevalue'] = {'policy_init': 'zero', 'value_head': 'state',
    'stages': [probe_stage(200), reward_stage(800)]}
ARMS['S5_simultaneous_probe_reward'] = {'policy_init': 'zero',
    'cfg': dict(H4, episodes=1000, w_probe=1., w_actor=1.)}
ARMS['S6_staged_probe200_then_reward_externallinear'] = {'policy_init': 'zero',
    'cfg': dict(external_readout=True, external_inputs='features'),
    'stages': [probe_stage(200), reward_stage(800)]}

if __name__ == '__main__':
    main('research/policy-learning/out/e4.json', ARMS, seeds=8)
