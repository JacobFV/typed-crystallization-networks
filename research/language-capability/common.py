"""Episodes, decoding, and splits for the language capability track.

Everything the agent may see comes from `record.observations`; `construction`
(latent) and `answer` (probe) are supervision only and are never packed into a
program input. The split helpers hold out *string length*, which is the
generating structure of a Dyck word, not merely the seed.
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from tcn.generation import Host, read_text
from tcn.types import Value

LESSON = 'context_free_language'
CAPACITY = 128
PREFIX = 'The string is '          # measured: one template, string always at byte 14
SUFFIX_LEN = len('.\nIs string balanced?\n\nAnswer with exactly one of: yes | no\nReply with the answer only.')

def episode(seed: int, split: str = 'train', capacity: int = CAPACITY):
    h = Host.create('language', seed=seed, split=split,
                    configuration={'lesson': LESSON, 'capacity': capacity})
    r = h.records[-1]
    text = r.observations['text'].value
    con = r.latent_states['construction']
    ans = read_text(r.probes['answer'])
    string = read_text(Value(con.type.items[1].items[1], con.raw[1][1]))
    prompt = read_text(text)
    return {'seed': seed, 'split': split, 'text': text, 'prompt': prompt,
            'string': string, 'answer': ans, 'label': ans == 'yes',
            'length': len(string), 'offset': prompt.index(string),
            'prompt_bytes': len(prompt.encode('utf8'))}

def dataset(n, seed0=0, split='train', capacity=CAPACITY):
    return [episode(seed0 + i, split, capacity) for i in range(n)]

def balanced(s: str) -> bool:
    d = 0
    for c in s:
        d += 1 if c == '(' else -1
        if d < 0: return False
    return d == 0

def counts_match(s: str) -> bool:
    return s.count('(') == s.count(')')
