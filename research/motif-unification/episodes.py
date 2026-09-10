"""Real episodes for the three domains, and the hole assignments they induce.

Nothing synthetic. The language stream is `research/language-capability`'s own
`episode()` (the `context_free_language` lesson at capacity 128); the visual
stream is `research/visual-ladder`'s FLAT screen configuration; the computer
stream is `research/computer-capability`'s own scripted host. No generator
setting is changed and no artifact is re-hardened, so no `hardening` choice is
exercised anywhere in this track (§39/§45): the language artifacts are static
programs, and the language *episodes* here come from the same lesson call the
track itself uses.

The certification episodes bind the motif's holes exactly as the artifact binds
them, with one deliberate exception, stated: the compared byte `x3` is drawn so
that the label is balanced, half the episodes True and half False. That is what
makes the best-constant baseline 0.5 rather than the artifact's own skew, and it
is the honest way to certify a *detector* rather than a detector-on-one-literal.
"""
from __future__ import annotations

import pathlib
import random
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

LANG = ROOT / "research" / "language-capability"
VISUAL = ROOT / "research" / "visual-ladder"
COMP = ROOT / "research" / "computer-capability"

# The artifacts' own bindings, read off in step 1.
LANGUAGE_BASE = 14        # `p14`, the PREFIX offset `L_stage_a` selects
VISUAL_OFFSETS = (1, 2)   # `one`, `two`: the green and blue channel offsets


def _load(name, path):
    """Load a track's `common.py` under a distinct module name.

    Three of the tracks ship a module called `common`; importing them by name
    would silently hand the second caller the first one's module. This binds each
    to its own key in `sys.modules`.
    """
    import importlib.util
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def language_buffers(n, seed0=0, split="train"):
    """`n` real prompts as 128-byte buffers."""
    sys.path.insert(0, str(LANG))
    LC = _load("_mu_lang_common", LANG / "common.py")
    out = []
    for i in range(n):
        e = LC.episode(seed0 + i, split)
        length, data = e["text"].decoded
        out.append(tuple(int(x) for x in data))
    return out


def visual_buffers(n, seed0=0, split="train"):
    """`n` real FLAT screens as 3,072-byte rasters."""
    sys.path.insert(0, str(VISUAL))
    VC = _load("_mu_visual_common", VISUAL / "common.py")
    out = []
    for i in range(n):
        e = VC.episode(seed0 + i, split, **VC.FLAT)
        out.append(tuple(int(x) for x in e["pixels"]))
    return out


def computer_records(documents, horizon=3):
    """The real scripted trajectories: terminal `Value`s, previous action, reference."""
    sys.path.insert(0, str(COMP))
    import task as CT
    rows = []
    for name, digit in documents:
        h, acts = CT.scripted_episode(name, digit, horizon=horizon)
        for r in CT.examples_from(h, acts):
            rows.append(r)
    return rows


# --------------------------------------------------------- hole assignments
def motif_episodes(buffers, base, offsets, width, seed=0, n_per_buffer=8):
    """(base, offset, buffer, other) with a balanced label.

    `base` is the artifact's own base for that domain (a fixed 14 in language, a
    drawn raster address in visual); `offset` is drawn from the artifact's own
    offset pool. `other` is the buffer's own byte at the address half the time
    and a different byte the other half, so the label is balanced by
    construction and the best constant is 0.5.
    """
    rng = random.Random(seed)
    rows = []
    for buf in buffers:
        for _ in range(n_per_buffer):
            off = rng.choice(offsets)
            b = base if base is not None else rng.randrange(0, width - max(offsets) - 1)
            a = b + off
            if a >= width:
                continue
            v = buf[a]
            if rng.random() < 0.5:
                other, label = v, True
            else:
                other, label = (v + 1 + rng.randrange(255)) % 256, False
            rows.append({"base": b, "offset": off, "buffer": buf,
                         "other": other, "label": label, "address": a})
    return rows
