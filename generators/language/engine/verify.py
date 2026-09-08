"""The admission test a lesson has to pass before its numbers mean anything.

A lesson is only evidence if its floors are right. Two generator bugs in this
curriculum's history produced fake competence — object-id order correlated with
the answer, and referring expressions that did not uniquely refer — and both
would have shown up as an agent "solving" a lesson it had not. So every lesson
must demonstrate three things over a block of freshly generated episodes:

* **it generates** — no exceptions across the block;
* **it is deterministic** — the same seed gives the same episode, every time;
* **its floor is low** — a constant guesser (always answering the most common
  gold label) scores near chance, not near one.

That last check is what catches a lesson that has quietly become a coin whose
answer is "no" 90% of the time. The threshold is loose for binary lessons (0.60)
and tighter for lessons with a larger answer set (0.45), and a lesson that trips
it is re-measured over a larger block before being condemned, because a constant
guesser's score is itself a random variable and a fair generator will trip a
fixed threshold now and then.

Since replies went open-form, this matters *more* rather than less. Nothing
grades the output any more, so a lesson whose answer has silently stopped
matching its observation does not show up as a bad score — it shows up as a
corpus that teaches the wrong thing, forever. :func:`verify_all` is a gate on
export, not a report you may skip.

:func:`verify_surface` is the same question asked of a transcode. A rasterized
episode whose glyphs are missing, or a dictated one whose options sound alike,
is not answerable from that surface at all, and its floor has nothing to do with
it. See ``INTENT.md``.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Sequence

from .lesson import Lesson, as_text
from .registry import all_lessons, get, resolve
from .surfaces import NATIVE_SURFACES, render_native, renders_natively, transcode

__all__ = ["verify_lesson", "verify_all", "verify_surface", "failures"]


def verify_lesson(lesson: Lesson | str, *, episodes: int = 200, seed0: int = 0,
                  difficulty: float | None = None) -> dict[str, Any]:
    """Generate episodes and check that the lesson measures what it claims to."""
    if isinstance(lesson, str):
        lesson = get(lesson)
    if lesson.status != "implemented":
        return {"lesson": lesson.id, "status": lesson.status, "ok": None, "note": lesson.note}
    from .context import GenerationContext
    ctx = GenerationContext(difficulty=difficulty)
    answers: list[Any] = []
    vocabs: list[int] = []
    errors: list[str] = []
    determinism_ok = True
    for i in range(episodes):
        try:
            obs, vocab, ans, hidden = lesson.build(seed0 + i, ctx)
        except Exception as e:
            errors.append(f"{type(e).__name__}: {e}")
            continue
        if ans not in list(vocab):
            errors.append(f"answer {ans!r} not in the episode's answer set")
            continue
        answers.append(ans)
        vocabs.append(len(list(vocab)))
        if i < 5:                                    # same seed must give the same episode
            obs2, _, ans2, _ = lesson.build(seed0 + i, ctx)
            determinism_ok &= (str(obs) == str(obs2) and ans == ans2)
    if not answers:
        return {"lesson": lesson.id, "ok": False,
                "reason": errors[:2] or ["no episodes generated"]}

    counts = Counter(answers)
    n = len(answers)
    mean_vocab = sum(vocabs) / n
    uniform = sum(1.0 / v for v in vocabs) / n       # expected accuracy of a uniform guesser
    constant = counts.most_common(1)[0][1] / n       # accuracy of always guessing the mode
    limit = 0.60 if mean_vocab <= 2.2 else 0.45
    resampled = None
    if constant > limit and not errors:
        more: list[Any] = []
        for i in range(episodes * 4):
            try:
                _, _, a2, _ = lesson.build(seed0 + 100_000 + i, ctx)
                more.append(a2)
            except Exception:
                break
        if more:
            resampled = Counter(more).most_common(1)[0][1] / len(more)
            constant = resampled
    ok = (not errors) and determinism_ok and constant <= limit
    return {"lesson": lesson.id, "tags": list(lesson.tags), "level": lesson.level,
            "episodes": n, "answer_set": round(mean_vocab, 1),
            "uniform": round(uniform, 3), "constant": round(constant, 3), "limit": limit,
            "deterministic": determinism_ok,
            "constant_resampled": (round(resampled, 3) if resampled is not None else None),
            "errors": errors[:2], "ok": bool(ok)}


def verify_all(lessons: str | Sequence[str] | None = None, *,
               episodes: int = 200, difficulty: float | None = None) -> list[dict[str, Any]]:
    """Verify a selection of lessons."""
    chosen = resolve(lessons) if lessons is not None else list(all_lessons().values())
    return [verify_lesson(l, episodes=episodes, difficulty=difficulty) for l in chosen]


def failures(rows: Sequence[dict[str, Any]]) -> list[str]:
    """The lesson ids that failed, for a caller using this as a gate."""
    return [r["lesson"] for r in rows if r.get("ok") is False]


def verify_surface(lesson: Lesson | str, surface: str, *, episodes: int = 20,
                   seed0: int = 0, language: str = "english",
                   **options: Any) -> dict[str, Any]:
    """Check that a transcode keeps the episode answerable.

    A floor is a claim about an answer set. A transcode does not change the
    answer set, so the floor carries over untouched — which is the whole reason
    transcoding is cheap. What a transcode *can* do is destroy the evidence: draw
    a glyph the font lacks, or read two distinct options as the same sound. That
    is what this measures, per lesson and per surface, by rendering real episodes
    rather than by declaring compatibility in a table.
    """
    if isinstance(lesson, str):
        lesson = get(lesson)
    if lesson.status != "implemented":
        return {"lesson": lesson.id, "surface": surface, "ok": None, "note": lesson.note}
    native = surface in NATIVE_SURFACES
    if native and not renders_natively(lesson, seed0):
        return {"lesson": lesson.id, "surface": surface, "ok": None,
                "note": f"builds nothing {surface} can draw"}
    lossy = 0
    dropped: set[str] = set()
    notes: set[str] = set()
    errors: list[str] = []
    for i in range(episodes):
        try:
            if native:
                content = render_native(lesson, seed0 + i, language=language,
                                        surface=surface, **options)
            else:
                ex = lesson.example(seed0 + i, language=language)
                # the episode's language travels with it: dictation only has
                # English rules, and a check that did not say which language it
                # was reading would pass everything
                content = transcode(ex.prompt, surface, target=ex.target,
                                    choices=ex.choices, language=ex.language,
                                    **options)
        except Exception as e:
            errors.append(f"{type(e).__name__}: {e}")
            continue
        if not content.fidelity.lossless:
            lossy += 1
        dropped |= set(content.fidelity.dropped)
        notes |= set(content.fidelity.notes)
    return {"lesson": lesson.id, "surface": surface, "episodes": episodes,
            "lossy_episodes": lossy, "dropped": sorted(dropped),
            "notes": sorted(notes), "errors": errors[:2],
            "ok": bool(not errors and lossy == 0)}
