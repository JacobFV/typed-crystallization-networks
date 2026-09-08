"""``identity_continuity`` — whether two descriptions denote the same entity under stated identity rules.

History, narrative, perspective, and identity.
"""

from __future__ import annotations

import random

from .._structure import Ident, Lst, Num, Pred, Rec
from ..lesson import Lesson
from ..generators.reflective import _nonces, _shuffled


def gen_identity_continuity(rng: random.Random, ctx):
    """Do these two descriptions pick out the same entity?

    Renaming preserves identity, replacing more than ``k`` parts does not,
    merging makes something new and splitting leaves identity with the larger
    fragment. The rules are stated in the episode and applied by a simulator
    that tracks identity tokens through the whole history, so the Ship of
    Theseus has an answer here rather than an opinion.
    """
    want = rng.random() < 0.5              # target answer, drawn once and balanced
    fallback = None
    for _ in range(400):
        names = _nonces(rng, ctx.at(6, 18, default=6), 3)
        start = names[:3]
        fresh = list(names[3:])
        k = rng.randint(1, 2)
        ident = {n: n for n in start}
        parts = {n: 0 for n in start}
        origin = dict(ident)
        ops: list[tuple] = []
        t = 1
        for _ in range(rng.randint(*ctx.span((3, 5), (9, 14)))):
            live = sorted(ident)
            kind = rng.choice(["rename", "replace", "replace", "merge", "split"])
            if kind == "rename" and fresh:
                x = rng.choice(live)
                new = fresh.pop()
                ops.append(("rename", t, x, new))
                ident[new], parts[new] = ident.pop(x), parts.pop(x)
            elif kind == "replace":
                x = rng.choice(live)
                ops.append(("replace_part", t, x))
                parts[x] += 1
                if parts[x] > k:
                    ident[x] = f"i{t}"
                    parts[x] = 0
            elif kind == "merge" and len(live) >= 2 and fresh:
                x, y = rng.sample(live, 2)
                new = fresh.pop()
                ops.append(("merge", t, x, y, new))
                del ident[x], ident[y]
                parts.pop(x), parts.pop(y)
                ident[new], parts[new] = f"i{t}", 0
            elif kind == "split" and len(fresh) >= 2:
                x = rng.choice(live)
                big, small = fresh.pop(), fresh.pop()
                ops.append(("split", t, x, big, small))
                ident[big], parts[big] = ident.pop(x), parts.pop(x)
                ident[small], parts[small] = f"i{t}s", 0
            else:
                continue
            t += 1
        if not ident:
            continue
        pairs = [(a, b) for a in start for b in sorted(ident)]
        yes = [p for p in pairs if origin[p[0]] == ident[p[1]]]
        no = [p for p in pairs if origin[p[0]] != ident[p[1]]]
        pool = (yes if want else no) or yes or no
        a, b = rng.choice(pool)
        truth = origin[a] == ident[b]
        cand = (start, ops, a, b, truth, k, t)
        if fallback is None:
            fallback = cand
        if truth == want:
            fallback = cand
            break
    start, ops, a, b, truth, k, t = fallback
    op_syms = []
    for op in ops:
        if op[0] == "rename":
            op_syms.append(Pred("rename", Num(op[1]), Ident(op[2]), Ident(op[3])))
        elif op[0] == "replace_part":
            op_syms.append(Pred("replace_part", Num(op[1]), Ident(op[2])))
        elif op[0] == "merge":
            op_syms.append(Pred("merge", Num(op[1]), Ident(op[2]), Ident(op[3]), Ident(op[4])))
        else:
            op_syms.append(Pred("split", Num(op[1]), Ident(op[2]), Ident(op[3]), Ident(op[4])))
    obs = Rec(entities=Lst([Ident(n) for n in start]),
              rules=Lst([Pred("rule", Ident("rename"), Pred("preserves_identity")),
                         Pred("rule", Pred("replaced_parts_allowed"), Num(k)),
                         Pred("rule", Ident("merge"), Pred("creates_new_identity")),
                         Pred("rule", Ident("split"), Pred("larger_fragment_keeps_identity"))]),
              history=Lst(op_syms),
              query=Pred("same_entity", Pred("at_start", Ident(a)), Pred("at_end", Ident(b))))
    return (obs, _shuffled(rng, ["yes", "no"]), "yes" if truth else "no",
            {"start_name": a, "end_name": b, "same": bool(truth), "n_ops": len(ops)})


class IdentityContinuity(Lesson):
    """Whether two descriptions denote the same entity under stated identity rules."""

    id = "identity_continuity"
    level = 134
    tags = ("history", "narrative", "perspective", "identity")
    teaches = "whether two descriptions denote the same entity under stated identity rules"
    capabilities = ('ontology_learning', 'temporal_reasoning', 'abstraction')
    axes = {'reasoning_depth': 5, 'discourse_horizon': 4, 'ambiguity': 4, 'world_complexity': 4}
    answers = ['yes', 'no']

    generate = staticmethod(gen_identity_continuity)
