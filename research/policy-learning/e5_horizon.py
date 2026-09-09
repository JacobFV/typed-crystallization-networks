"""E5: horizon scaling, dense (the generator's own per-step reward) and terminal-only."""
from run_arms import main
def base(h, **kw): return dict(episodes=1000, horizon=h, lr=.04, w_actor=1., w_value=.5, w_entropy=.01, **kw)
def probe_stage(h): return {'cfg': base(h, episodes=200, w_probe=1., w_actor=0., w_value=0., log_every=25)}
def reward_stage(h, terminal): return {'freeze': True, 'reset_optimizer': True,
    'cfg': base(h, episodes=800, w_probe=0., terminal_reward_only=terminal,
                index_offset=500000, log_every=25)}
ARMS = {}
for h in (4, 8, 16, 32):
    ARMS[f'H{h}_rewardonly_dense']  = {'policy_init':'zero','cfg':base(h)}
    ARMS[f'H{h}_rewardonly_terminal']={'policy_init':'zero','cfg':base(h,terminal_reward_only=True)}
    ARMS[f'H{h}_staged_dense']      = {'policy_init':'zero','stages':[probe_stage(h),reward_stage(h,False)]}
    ARMS[f'H{h}_staged_terminal']   = {'policy_init':'zero','stages':[probe_stage(h),reward_stage(h,True)]}
    ARMS[f'H{h}_staged_terminal_b32']={'policy_init':'zero','stages':[probe_stage(h),
        {'freeze':True,'reset_optimizer':True,'cfg':base(h,episodes=800,w_probe=0.,
         terminal_reward_only=True,batch=32,normalize_advantage=True,index_offset=500000,log_every=25)}]}
if __name__ == '__main__':
    main('research/policy-learning/out/e5.json', ARMS, seeds=4)
