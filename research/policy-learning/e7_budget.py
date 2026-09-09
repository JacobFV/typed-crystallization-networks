"""E7: return against environment-episode budget, for the arms that matter."""
from run_arms import main
BUDGETS = (100, 200, 400, 800, 1200, 1600, 2000, 4000)
def cfg(**kw): return dict(dict(horizon=4, lr=.04, w_actor=1., w_value=.5, w_entropy=.01), **kw)
ARMS = {}
for e in BUDGETS:
    ARMS[f'P1_rewardonly_{e}']      = {'policy_init':'zero','cfg':cfg(episodes=e)}
    ARMS[f'P2_probe_reward_{e}']    = {'policy_init':'zero','cfg':cfg(episodes=e,w_probe=1.)}
    ARMS[f'P3_pinned_readout_{e}']  = {'policy_init':'zero','pin_choices':{'relation':6,'goal_relation':6},
                                       'cfg':cfg(episodes=e)}
    ARMS[f'P4_mlp_reinforce_{e}']   = {'policy_init':'zero','cfg':cfg(episodes=e,external_readout=True,
                                       external_inputs='raw',external_hidden=32,train_choices=False,
                                       train_constants=False)}
if __name__ == '__main__':
    main('research/policy-learning/out/e7.json', ARMS, seeds=8)
