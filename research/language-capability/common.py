"""Episodes, decoding, and splits for the language capability track.

Everything the agent may see comes from `record.observations`; `construction`
(latent) and `answer` (probe) are supervision only and are never packed into a
program input. The split helpers hold out *string length*, which is the
generating structure of a Dyck word, not merely the seed.

**`hardening` is a named argument and never inherited.** FINDINGS section 39
exists because this file used to take the generator default silently, so the
numbers it produced did not say which stream they came from -- and when section
24 re-drew `context_free_language` the whole track moved underneath its own
record without saying so. `STREAM_POST_AUDIT` is the shipped hardened draw and
is the default here; `STREAM_PRE_AUDIT` (`hardening='none'`) is the bit-identical
pre-audit stream, which is the one section 19 was measured on. Pinning the name
is bit-identical to inheriting the default for this lesson -- verified over 240
episodes across both splits -- so no number this track recorded moves.
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from tcn.generation import Host, read_text
from tcn.types import Value

LESSON = 'context_free_language'
CAPACITY = 128
STREAM_POST_AUDIT = 'context_free_language'   # the shipped hardened draw, named
STREAM_PRE_AUDIT = 'none'                     # the pre-audit stream of section 19
PREFIX = 'The string is '          # measured: one template, string always at byte 14
SUFFIX_LEN = len('.\nIs string balanced?\n\nAnswer with exactly one of: yes | no\nReply with the answer only.')

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
    """`hardening` is pinned at every call site by convention: never omit it."""
    return [episode(seed0 + i, split, capacity, hardening) for i in range(n)]

def balanced(s: str) -> bool:
    d = 0
    for c in s:
        d += 1 if c == '(' else -1
        if d < 0: return False
    return d == 0

def counts_match(s: str) -> bool:
    return s.count('(') == s.count(')')


def synthetic_text(symbols: str, capacity: int = CAPACITY):
    """A prompt in the lesson's own single template, with an arbitrary symbol string.

    Off-distribution by construction: it is used only to interrogate an exported
    program on strings the generator cannot emit, never as training or as a
    reported accuracy on the lesson.
    """
    from tcn.generation import text_value
    prompt = (PREFIX + symbols +
              '.\nIs string balanced?\n\nAnswer with exactly one of: yes | no\nReply with the answer only.')
    return text_value(prompt, capacity), prompt
