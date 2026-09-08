"""``question_generation`` — identifying the one missing fact and the question that supplies it.

Language as action.
"""

from __future__ import annotations

import random

from .._structure import Ident, Lst, Pred, Rec
from ..lesson import Lesson
from ..generators.base import COLORS, NAMES
from ..generators.causal import _labels, _nonce_names, _options, _qg_colors


def gen_question_generation(rng: random.Random, ctx):
    """The agent must answer 'what colour is the box in the room where P is?'
    and is missing exactly one link of that chain.

    Necessity and sufficiency are *computed*: the goal has ≥2 possible colours
    under the visible facts, exactly 1 once the correct question is answered,
    and ≥2 once any distractor is."""
    n_rooms = ctx.at(4, 9, default=4)            # rooms, and one box more than rooms
    persons = rng.sample(NAMES, 3)
    rooms = _nonce_names(rng, n_rooms)
    boxes = _nonce_names(rng, n_rooms + 1)
    while set(boxes) & set(rooms):
        boxes = _nonce_names(rng, n_rooms + 1)
    palette = rng.sample(COLORS, 5) if len(COLORS) >= 5 else list(COLORS)

    for _ in range(200):
        at = dict(zip(persons, rng.sample(rooms, 3)))
        contains = dict(zip(rooms, rng.sample(boxes, n_rooms)))
        color = {b: rng.choice(palette) for b in boxes}
        target = rng.choice(persons)
        missing = rng.choice(["at", "contains", "color"])
        r_t, b_t = at[target], contains[at[target]]

        vis_at, vis_contains, vis_color = dict(at), dict(contains), dict(color)
        if missing == "at":
            del vis_at[target]
            need = ("where_is", target)
        elif missing == "contains":
            del vis_contains[r_t]
            need = ("what_is_in", r_t)
        else:
            del vis_color[b_t]
            need = ("what_colour_is", b_t)

        def possible(a=None, c=None, col=None) -> set[str]:
            return _qg_colors(a if a is not None else vis_at,
                              c if c is not None else vis_contains,
                              col if col is not None else vis_color,
                              target, rooms, boxes, palette)

        if len(possible()) < 2:              # already derivable: no fact is missing
            continue

        pool = ([("where_is", p) for p in persons] + [("what_is_in", r) for r in rooms]
                + [("what_colour_is", b) for b in boxes])
        pool = [q for q in pool if q != need]
        rng.shuffle(pool)
        distractors: list[tuple[str, str]] = []
        for q in pool:
            a, c, col = dict(vis_at), dict(vis_contains), dict(vis_color)
            if q[0] == "where_is":
                a[q[1]] = at[q[1]]
            elif q[0] == "what_is_in":
                c[q[1]] = contains[q[1]]
            else:
                col[q[1]] = color[q[1]]
            if len(possible(a, c, col)) >= 2:          # answering it still leaves the goal open
                distractors.append(q)
            if len(distractors) == 3:
                break
        if len(distractors) < 3:
            continue

        opts, correct = _options(rng, need, distractors)
        labels = _labels("q", len(opts))
        facts = ([Pred("at", Ident(p), Ident(r)) for p, r in sorted(vis_at.items())]
                 + [Pred("contains", Ident(r), Ident(b)) for r, b in sorted(vis_contains.items())]
                 + [Pred("color", Ident(b), Ident(c)) for b, c in sorted(vis_color.items())])
        rng.shuffle(facts)
        obs = Rec(
            facts=Lst(facts),
            rules=Lst([Pred("rule", Pred("each_person_is_in_exactly_one_room")),
                       Pred("rule", Pred("no_two_people_share_a_room")),
                       Pred("rule", Pred("each_room_contains_exactly_one_box")),
                       Pred("rule", Pred("no_box_is_in_two_rooms"))]),
            goal=Pred("colour_of_box_in_room_of", Ident(target)),
            questions=Lst([Pred("question", Ident(lab), Ident(q[0]), Ident(q[1]))
                           for lab, q in zip(labels, opts)]),
            query=Ident("which_question_do_you_need"))
        hidden = {"missing": missing, "needed_question": list(need),
                  "answer_label": labels[correct]}
        return obs, labels, labels[correct], hidden

    raise RuntimeError("question_generation: no admissible world")


class QuestionGeneration(Lesson):
    """Identifying the one missing fact and the question that supplies it."""

    id = "question_generation"
    level = 37
    tags = ("pragmatics", "language-as-action")
    teaches = "identifying the one missing fact and the question that supplies it"
    capabilities = ('metareasoning', 'quantification', 'planning')
    axes = {'reasoning_depth': 3, 'world_complexity': 2, 'ambiguity': 2, 'compositional_depth': 2}
    answers = ['q0', 'q1', 'q2', 'q3']

    generate = staticmethod(gen_question_generation)
