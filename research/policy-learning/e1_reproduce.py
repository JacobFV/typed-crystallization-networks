"""E1/E2: clean reproduction of the reward-only failure, and the per-cause ablations."""
import sys
from run_arms import main

E = 2000
BASE = dict(episodes=E, horizon=4, lr=.04, w_actor=1., w_value=.5, w_entropy=.01)
def cfg(**kw): return dict(BASE, **kw)

ARMS = {
 # ---- reproduction -------------------------------------------------------
 'A0_shipped_oracleinit_probe':   {'policy_init':'oracle','cfg':cfg(w_probe=1.,choice_bias12=True,episodes=160)},
 'A0b_shipped_2000ep':            {'policy_init':'oracle','cfg':cfg(w_probe=1.,choice_bias12=True)},
 'A1_rewardonly_zeroinit':        {'policy_init':'zero','cfg':cfg(),'snr_at_init':True,'snr_after':True},
 'A1c_rewardonly_zeroinit_bias12':{'policy_init':'zero','cfg':cfg(choice_bias12=True),'snr_at_init':True,'snr_after':True},
 'A1d_rewardonly_zeroinit_choicenoise':{'policy_init':'zero','cfg':cfg(choice_noise=.5),'snr_at_init':True,'snr_after':True},
 'A2_rewardonly_noiseinit':       {'policy_init':'noise','cfg':cfg(choice_noise=.1),'snr_at_init':True,'snr_after':True},
 'A3_rewardonly_oracleconstinit': {'policy_init':'oracle','cfg':cfg(),'snr_at_init':True,'snr_after':True},
 'A4_rewardonly_choiceonly':      {'policy_init':'oracle','cfg':cfg(train_constants=False),'snr_at_init':True},
 'A5_probeonly_zeroinit':         {'policy_init':'zero','cfg':cfg(w_probe=1.,w_actor=0.,w_value=0.)},
 'A6_probe_plus_reward_zeroinit': {'policy_init':'zero','cfg':cfg(w_probe=1.)},
 'A7_mlp_reinforce_raw':          {'policy_init':'zero','cfg':cfg(external_readout=True,external_inputs='raw',
                                    external_hidden=32,train_choices=False,train_constants=False),'snr_at_init':True},
 # ---- cause 1: score-function variance -----------------------------------
 'B1_exactPG_zeroinit':           {'policy_init':'zero','cfg':cfg(exact_policy_gradient=True),'snr_at_init':True},
 'B1b_exactPG_pinned':            {'policy_init':'zero','pin_choices':{'relation':6,'goal_relation':6},
                                   'cfg':cfg(exact_policy_gradient=True)},
 # ---- cause 2: the value baseline ----------------------------------------
 'B2_value_none':                 {'policy_init':'zero','value_head':'none','cfg':cfg()},
 'B3_value_state':                {'policy_init':'zero','value_head':'state','cfg':cfg()},
 'B4_advnorm_constant_baseline':  {'policy_init':'zero','cfg':cfg(normalize_advantage=True,batch=16)},
 'B5_value_state_advnorm':        {'policy_init':'zero','value_head':'state','cfg':cfg(normalize_advantage=True,batch=16)},
 # ---- cause 3: the discrete-choice bottleneck ----------------------------
 'B6_pinned_readout_only':        {'policy_init':'zero','pin_choices':{'relation':6,'goal_relation':6},'cfg':cfg(),
                                   'snr_at_init':True,'snr_after':True},
 'B7_pinned_readout_only_short':  {'policy_init':'zero','pin_choices':{'relation':6,'goal_relation':6},
                                   'cfg':cfg(episodes=200)},
 # ---- cause 4: the relaxed readout as a conditioner ----------------------
 'B8_external_linear_pinned':     {'policy_init':'zero','pin_choices':{'relation':6,'goal_relation':6},
                                   'cfg':cfg(external_readout=True,external_inputs='features',train_constants=False),
                                   'snr_at_init':True},
 'B9_external_linear_free':       {'policy_init':'zero','cfg':cfg(external_readout=True,external_inputs='features',
                                    train_constants=False),'snr_at_init':True,'snr_after':True},
}

if __name__ == '__main__':
    main('research/policy-learning/out/e1.json', ARMS, seeds=8)
