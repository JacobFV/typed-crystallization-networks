"""How often does the shipped policy sampler ever emit a rewardable write?

`tcn/policy.py:sample_typed` draws each field of a typed action argument
independently. The generator's `write.text` argument is a 512-byte text, so the
policy emits 1026 parameters: a mean/log-sigma pair for the length field and one
for every byte. `read_text` only reads the first `length` bytes, so the effective
target is two draws -- but they are two *continuous* draws squashed onto 0..512 and
0..255, and the reward is exact string equality.

This is the number that decides whether the scalar reward can start learning at
all, so it is measured rather than argued.
"""
import json,sys
sys.path.insert(0,'/home/brandonin/Documents/typed-crystallization-networks')
sys.path.insert(0,'/home/brandonin/Documents/typed-crystallization-networks/research/computer-capability')
import torch
from tcn.generation import read_text,text_value
from tcn.policy import sample_exact,parameter_width
import program as P

BASE='/home/brandonin/Documents/typed-crystallization-networks/research/computer-capability/out/'
TEXT=P.TEXT

def draw(parameters,draws,seed=0):
    # `sample_exact` is the frozen-execution sampler and implements the same latent
    # distribution as `sample_typed`; it is used here because it is standard-library
    # arithmetic rather than 1026 torch distribution objects per draw.
    import random
    rng=random.Random(seed)
    lengths={};hits=0;single=0;texts={}
    for _ in range(draws):
        value=sample_exact(TEXT,list(parameters),rng,deterministic=False)
        text=read_text(value)
        lengths[len(text)]=lengths.get(len(text),0)+1
        if len(text)==1:
            single+=1;texts[text]=texts.get(text,0)+1
            if text=='8': hits+=1
    return {'draws':draws,'length_one':single,'exactly_8':hits,
            'distinct_single_byte_texts':len(texts),
            'length_histogram_top':sorted(lengths.items(),key=lambda x:-x[1])[:5]}

report={'parameter_width':parameter_width(TEXT),'text_capacity':512}
report['neutral_policy']=draw(torch.zeros(parameter_width(TEXT)),20000)
# The same sampler once a program supplies the two means the encoder computes.
import math
tuned=[0.]*parameter_width(TEXT)
tuned[0]=P.MEAN_LENGTH;tuned[1]=P.LOG_STD
tuned[2]=.5*(math.log(56.)-math.log(255.-56.));tuned[3]=P.LOG_STD
report['program_supplied_parameters']=draw(tuned,20000)
def band(target,hi,lo=0.):
    """Exact probability that one neutral Normal(0,1) draw decodes to `target`.

    `sample_typed`/`sample_exact` map u -> lo + (tanh(u)+1)/2*(hi-lo) and round, so
    the event is an interval in u and its probability is a difference of normal CDFs.
    """
    import math
    def inverse(x):
        z=2*(x-lo)/(hi-lo)-1
        z=min(1-1e-15,max(-1+1e-15,z))
        return .5*math.log((1+z)/(1-z))
    a=inverse(target-.5);b=inverse(target+.5)
    phi=lambda u:.5*(1+math.erf(u/math.sqrt(2)))
    return phi(b)-phi(a)
length_probability=band(1,512.)
byte_probability=band(56,255.)
joint=length_probability*byte_probability
report['analytic']={
 'probability_length_field_decodes_to_1':length_probability,
 'probability_byte_field_decodes_to_the_successor':byte_probability,
 'probability_one_neutral_write_is_rewardable':joint,
 'expected_write_actions_to_first_reward':1/joint,
 'expected_episodes_at_horizon_3_uniform_verb':1/joint/(3*(1/3.)),
 'measured_episode_seconds':13.3,
 'expected_days_of_wall_clock_to_first_reward':1/joint/(3*(1/3.))*13.3/86400,

 'neutral_mean_decodes_to':'length 256, every byte 128 -- a 256-byte text of 0x80',
 'why_reinforce_cannot_start':'the reward is exact content equality, so a rollout scores 0 '
   'until the length draw lands on 1 and the byte draw lands on the successor digit at the same time',
}
open(BASE+'reinforce_feasibility.json','w').write(json.dumps(report,indent=2,sort_keys=True))
print(json.dumps(report,indent=2,sort_keys=True))
