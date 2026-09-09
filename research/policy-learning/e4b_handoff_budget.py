"""E4b: how little reward the staged route needs once the model is frozen."""
from run_arms import main
H4 = dict(horizon=4, lr=.04, w_value=.5, w_entropy=.01)
def probe_stage(n): return {'cfg': dict(H4, episodes=n, w_probe=1., w_actor=0., w_value=0., log_every=25)}
def reward_stage(n): return {'freeze': True, 'reset_optimizer': True,
    'cfg': dict(H4, episodes=max(n,1), w_probe=0., w_actor=1., w_value=.5,
                index_offset=500000, log_every=10)}
ARMS = {}
for p in (50, 200):
    for rr in (1, 5, 10, 25, 50, 100, 200, 400):
        ARMS[f'T_probe{p}_reward{rr}'] = {'policy_init':'zero','stages':[probe_stage(p), reward_stage(rr)]}
if __name__ == '__main__':
    main('research/policy-learning/out/e4b.json', ARMS, seeds=8)
