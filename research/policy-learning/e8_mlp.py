"""E8: is the MLP baseline being handicapped? A tuning grid for the neural arm.

The A7/P4 arms give the MLP the program's fixed constant baseline (1.0), which is
weaker than the value head research/baselines gave its MLPs. This grid gives it a
batch baseline (advantage normalisation over the batch) as well, across width,
learning rate and batch size, so the "never learns it" claim is not an artifact of
under-tuning.
"""
from run_arms import main
def cfg(**kw): return dict(dict(episodes=2000, horizon=4, w_actor=1., w_value=.5, w_entropy=.01,
                                external_readout=True, external_inputs='raw',
                                train_choices=False, train_constants=False), **kw)
ARMS = {}
for h in (8, 32):
    for lr in (.01, .04, .1):
        ARMS[f'M_h{h}_lr{lr}_b1']   = {'policy_init':'zero','cfg':cfg(external_hidden=h,lr=lr)}
        ARMS[f'M_h{h}_lr{lr}_b32n'] = {'policy_init':'zero','cfg':cfg(external_hidden=h,lr=lr,
                                        batch=32,normalize_advantage=True)}
if __name__ == '__main__':
    main('research/policy-learning/out/e8.json', ARMS, seeds=4)
