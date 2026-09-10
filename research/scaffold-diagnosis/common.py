"""Episodes, decoding and splits for the *post-audit* language track.

Identical to `research/language-capability/common.py` except that `hardening`
is **pinned explicitly in every call** rather than inherited from the generator
default. FINDINGS section 39 exists because the original file inherited it; this
file states which stream every number came from.

`STREAM_POST_AUDIT` is the hardened `context_free_language` draw introduced by
section 24; `STREAM_PRE_AUDIT` is the bit-identical pre-audit stream, reachable
as `hardening='none'`, which is the stream section 19 was measured on.

Everything the agent may see comes from `record.observations`; `construction`
(latent) and `answer` (probe) are supervision only and are never packed into a
program input.
"""
from __future__ import annotations
import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from tcn.generation import Host, read_text
from tcn.types import Value

LESSON = 'context_free_language'
CAPACITY = 128
PREFIX = 'The string is '
STREAM_POST_AUDIT = 'context_free_language'   # the named hardened draw, explicitly
STREAM_PRE_AUDIT = 'none'


def episode(seed: int, split: str = 'train', capacity: int = CAPACITY,
            hardening: str = STREAM_POST_AUDIT):
    h = Host.create('language', seed=seed, split=split,
                    configuration={'lesson': LESSON, 'capacity': capacity,
                                   'hardening': hardening})
    r = h.records[-1]
    text = r.observations['text'].value
    con = r.latent_states['construction']
    ans = read_text(r.probes['answer'])
    string = read_text(Value(con.type.items[1].items[1], con.raw[1][1]))
    depth = Value(con.type.items[0].items[1], con.raw[0][1]).decoded
    prompt = read_text(text)
    return {'seed': seed, 'split': split, 'text': text, 'prompt': prompt,
            'string': string, 'answer': ans, 'label': ans == 'yes',
            'length': len(string), 'depth': depth, 'offset': prompt.index(string),
            'prompt_bytes': len(prompt.encode('utf8')), 'stream': hardening}


def dataset(n, seed0=0, split='train', capacity=CAPACITY, hardening=STREAM_POST_AUDIT):
    """`hardening` is pinned by convention at every call site: never omit it."""
    return [episode(seed0 + i, split, capacity, hardening) for i in range(n)]


def balanced(s: str) -> bool:
    d = 0
    for c in s:
        d += 1 if c == '(' else -1
        if d < 0: return False
    return d == 0


def counts_match(s: str) -> bool:
    return s.count('(') == s.count(')')


def min_prefix(s: str) -> int:
    d = m = 0
    for c in s:
        d += 1 if c == '(' else -1
        m = min(m, d)
    return m
