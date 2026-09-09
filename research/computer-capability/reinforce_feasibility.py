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
from tcn.policy import sample_typed,parameter_width
import program as P

BASE='/home/brandonin/Documents/typed-crystallization-networks/research/computer-capability/out/'
TEXT=P.TEXT

def draw(parameters,draws,seed=0):
    torch.manual_seed(seed)
    lengths={};hits=0;single=0;texts={}
    for _ in range(draws):
        value,_,_=sample_typed(TEXT,parameters,deterministic=False)
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
tuned=torch.zeros(parameter_width(TEXT))
tuned[0]=P.MEAN_LENGTH;tuned[1]=P.LOG_STD
tuned[2]=.5*(torch.tensor(56.).log()-torch.tensor(255.-56.).log());tuned[3]=P.LOG_STD
report['program_supplied_parameters']=draw(tuned,20000)
report['analytic']={
 'neutral_mean_decodes_to':'length 256, every byte 128 -- a 256-byte text of 0x80',
 'why_reinforce_cannot_start':'the reward is exact content equality, so a rollout scores 0 '
   'until the length draw lands on 1 and the byte draw lands on the successor digit at the same time',
}
open(BASE+'reinforce_feasibility.json','w').write(json.dumps(report,indent=2,sort_keys=True))
print(json.dumps(report,indent=2,sort_keys=True))
