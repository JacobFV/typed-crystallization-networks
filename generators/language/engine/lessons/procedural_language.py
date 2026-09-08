"""``procedural_language`` — loops, branches and state: executing a described procedure.

Analogy, causality, planning, and programs.
"""

from __future__ import annotations

import random

from .._structure import Ident, Lst, Num, Pred, Rec, Tok
from ..lesson import Lesson
from ..generators.causal import _PROC_MOD, _nonce_names, _proc_body, _proc_exec, _proc_simple, _proc_symbol


def gen_procedural_language(rng: random.Random, ctx):
    """A tiny imperative program with a bounded loop and a branch. All
    arithmetic is mod 10 and stated in the observation; the answer is what the
    interpreter leaves in the queried variable."""
    vs = _nonce_names(rng, 2)
    init = {v: rng.randrange(_PROC_MOD) for v in vs}

    prog: list[tuple] = [_proc_simple(rng, vs)]
    prog.append(("repeat", rng.randint(*ctx.span((2, 4), (7, 12))),
                 _proc_body(rng, vs, rng.randint(*ctx.span((1, 2), (3, 4))))))
    if rng.random() < 0.75:
        prog.append(("ifgt", rng.choice(vs), _proc_body(rng, vs, 1),
                     rng.randrange(_PROC_MOD), _proc_body(rng, vs, 1)))
    prog.append(_proc_simple(rng, vs))
    rng.shuffle(prog)

    final = _proc_exec(prog, init)
    target = rng.choice(vs)
    obs = Rec(modulus=Num(_PROC_MOD),
              # was Str("every assignment is taken mod m"), which is a literal
              # and went out in English whatever the language
              note=Pred("mod", Tok("value"), Tok("modulus")),
              init=Lst([Pred("init", Ident(v), Num(init[v])) for v in vs]),
              program=Lst([_proc_symbol(s) for s in prog]),
              query=Pred("final_value", Ident(target)))
    hidden = {"init": init, "final": final, "target": target}
    return obs, list(range(_PROC_MOD)), final[target], hidden


class ProceduralLanguage(Lesson):
    """Loops, branches and state: executing a described procedure."""

    id = "procedural_language"
    level = 45
    tags = ("analogy", "causality", "planning", "programs")
    teaches = "loops, branches and state: executing a described procedure"
    capabilities = ('program_synthesis', 'variable_binding', 'recursive_syntax')
    axes = {'reasoning_depth': 4, 'recursion_depth': 2, 'compositional_depth': 3, 'grammar_complexity': 3}
    answers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    generate = staticmethod(gen_procedural_language)
