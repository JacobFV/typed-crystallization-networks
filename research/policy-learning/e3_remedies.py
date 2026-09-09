"""E3: standard remedies, cheapest first, all reward-only at horizon 4."""
from run_arms import main
BASE = dict(episodes=2000, horizon=4, lr=.04, w_actor=1., w_value=.5, w_entropy=.01)
def cfg(**kw): return dict(BASE, **kw)
ARMS = {
 'R0_baseline_rewardonly':      {'policy_init':'zero','cfg':cfg()},
 'R1_batch8':                   {'policy_init':'zero','cfg':cfg(batch=8)},
 'R2_batch32':                  {'policy_init':'zero','cfg':cfg(batch=32)},
 'R3_batch32_lr01':             {'policy_init':'zero','cfg':cfg(batch=32,lr=.1)},
 'R4_entropy_schedule':         {'policy_init':'zero','cfg':cfg(w_entropy=.2,entropy_final=.001)},
 'R5_entropy_high_flat':        {'policy_init':'zero','cfg':cfg(w_entropy=.2)},
 'R6_advnorm_batch32':          {'policy_init':'zero','cfg':cfg(batch=32,normalize_advantage=True)},
 'R7_statevalue_advnorm_b32':   {'policy_init':'zero','value_head':'state','cfg':cfg(batch=32,normalize_advantage=True)},
 'R8_lr005':                    {'policy_init':'zero','cfg':cfg(lr=.005)},
 'R9_lr20':                     {'policy_init':'zero','cfg':cfg(lr=.2)},
 'R10_8000_episodes':           {'policy_init':'zero','cfg':cfg(episodes=8000)},
 'R11_everything':              {'policy_init':'zero','value_head':'state',
                                 'cfg':cfg(episodes=8000,batch=32,normalize_advantage=True,
                                           w_entropy=.2,entropy_final=.001)},
 'R12_exactPG_upperbound':      {'policy_init':'zero','cfg':cfg(exact_policy_gradient=True)},
}
if __name__ == '__main__':
    main('research/policy-learning/out/e3.json', ARMS, seeds=8)
