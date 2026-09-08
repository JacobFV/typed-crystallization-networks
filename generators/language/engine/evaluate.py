"""Evaluating a text agent, and the floor it has to beat.

An agent here is the simplest thing that can be one:

.. code-block:: python

    def agent(prompt: str) -> str:
        ...

Nothing else is required — no class, no state, no framework. Whatever answers
strings with strings can be measured.

**Why the floor is reported next to every number.** Accuracy alone is not a
result on this curriculum, because the size of the answer set varies per lesson
and per *episode*: a yes/no lesson floors at 0.5 and a "which of these nine
objects" lesson floors at 0.11. Two lessons scored 0.55 are not comparable, and
one of them may be worse than guessing. So every report carries, for the same
episodes:

``random``
    a uniform guesser over each episode's own choices — what knowing nothing
    scores;
``majority``
    always answering the most common gold answer across the sampled episodes —
    what exploiting a label imbalance scores, which is the honest floor whenever
    it is higher than random.

``lift`` is accuracy minus that floor, normalized by the headroom above it, so
0.0 means "no better than not knowing" and 1.0 means perfect. That is the number
to compare across lessons.
"""

from __future__ import annotations

import random
import re
import statistics
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from .languages import DEFAULT_LANGUAGE, get_language
from .lesson import Lesson
from .presentation import Presentation
from .registry import get, resolve
from .scoring import score

__all__ = ["evaluate", "evaluate_lesson", "Report", "LessonResult",
           "random_agent", "constant_agent", "TextAgent"]

#: what an agent is: text in, text out
TextAgent = Callable[[str], str]


@dataclass
class LessonResult:
    """The outcome of one lesson, with its floors beside it."""

    lesson_id: str
    tags: tuple[str, ...]
    level: int
    n: int
    correct: int
    accuracy: float
    random_baseline: float
    majority_baseline: float
    mean_choices: float
    errors: int = 0
    language: str = DEFAULT_LANGUAGE
    presentation: str = ""
    wrong: list[dict[str, str]] = field(default_factory=list)

    @property
    def floor(self) -> float:
        """The higher of the two baselines: what a knowing-nothing agent gets."""
        return max(self.random_baseline, self.majority_baseline)

    @property
    def lift(self) -> float:
        """Accuracy normalized into [0, 1] over the headroom above the floor."""
        f = self.floor
        if f >= 1.0:
            return 0.0
        return max(0.0, (self.accuracy - f) / (1.0 - f))

    @property
    def solved(self) -> bool:
        """A conventional threshold: 90% of the headroom above the floor."""
        return self.lift >= 0.9

    def to_dict(self) -> dict[str, Any]:
        d = {k: getattr(self, k) for k in
             ("lesson_id", "level", "n", "correct", "accuracy", "random_baseline",
              "majority_baseline", "mean_choices", "errors", "language", "presentation")}
        d["tags"] = list(self.tags)
        d.update(floor=self.floor, lift=self.lift, solved=self.solved)
        if self.wrong:
            d["wrong"] = self.wrong
        return d


@dataclass
class Report:
    """Results across lessons, plus the aggregate views worth quoting."""

    results: list[LessonResult]
    language: str = DEFAULT_LANGUAGE
    n: int = 0
    seed0: int = 0

    def __len__(self) -> int:
        return len(self.results)

    def __iter__(self):
        return iter(self.results)

    def __getitem__(self, lesson_id: str) -> LessonResult:
        for r in self.results:
            if r.lesson_id == lesson_id:
                return r
        raise KeyError(lesson_id)

    @property
    def accuracy(self) -> float:
        """Macro-average accuracy: every lesson counts once."""
        return statistics.fmean(r.accuracy for r in self.results) if self.results else 0.0

    @property
    def floor(self) -> float:
        return statistics.fmean(r.floor for r in self.results) if self.results else 0.0

    @property
    def lift(self) -> float:
        return statistics.fmean(r.lift for r in self.results) if self.results else 0.0

    @property
    def solved(self) -> list[str]:
        return [r.lesson_id for r in self.results if r.solved]

    def by_tag(self) -> dict[str, float]:
        """Mean lift per tag — the profile, rather than the single number."""
        acc: dict[str, list[float]] = {}
        for r in self.results:
            for t in r.tags:
                acc.setdefault(t, []).append(r.lift)
        return {k: statistics.fmean(v) for k, v in sorted(acc.items())}

    def by_curriculum(self, curriculum: str = "progressive") -> list[tuple[str, float]]:
        """Lift in a curriculum's own order, for reading a profile as a curve."""
        from .curricula import get as get_curriculum
        have = {r.lesson_id: r.lift for r in self.results}
        return [(n.lesson, have[n.lesson])
                for n in get_curriculum(curriculum).linearize() if n.lesson in have]

    def by_capability(self) -> dict[str, float]:
        """Mean lift per capability tag."""
        acc: dict[str, list[float]] = {}
        for r in self.results:
            for c in get(r.lesson_id).capabilities:
                acc.setdefault(c, []).append(r.lift)
        return {k: statistics.fmean(v) for k, v in sorted(acc.items())}

    def to_dict(self) -> dict[str, Any]:
        return {"language": self.language, "n": self.n, "seed0": self.seed0,
                "lessons": len(self.results), "accuracy": self.accuracy,
                "floor": self.floor, "lift": self.lift,
                "solved": self.solved, "by_tag": self.by_tag(),
                "results": [r.to_dict() for r in self.results]}

    def table(self, *, limit: int | None = None) -> str:
        """A fixed-width summary, for printing."""
        head = f"{'lesson':<34}{'n':>5}{'acc':>8}{'floor':>8}{'lift':>8}"
        rows = [head, "-" * len(head)]
        for r in sorted(self.results, key=lambda r: r.lesson_id)[:limit]:
            rows.append(f"{r.lesson_id:<34}{r.n:>5}{r.accuracy:>8.3f}"
                        f"{r.floor:>8.3f}{r.lift:>8.3f}")
        rows.append("-" * len(head))
        rows.append(f"{'macro-average':<34}{self.n:>5}{self.accuracy:>8.3f}"
                    f"{self.floor:>8.3f}{self.lift:>8.3f}")
        return "\n".join(rows)

    def __str__(self) -> str:
        return self.table()


# --------------------------------------------------------------------------
# reference agents
# --------------------------------------------------------------------------
def random_agent(seed: int = 0) -> TextAgent:
    """Picks uniformly among the options the prompt lists. The floor, embodied."""
    rng = random.Random(seed)

    def _agent(prompt: str) -> str:
        opts = _options_from_prompt(prompt)
        return rng.choice(opts) if opts else ""

    return _agent


def constant_agent(reply: str = "yes") -> TextAgent:
    """Always says the same thing, whatever it was asked."""
    return lambda _prompt: reply


def _options_from_prompt(prompt: str) -> list[str]:
    """Recover the answer set an :class:`~langcurriculum.lesson.Example` printed.

    Only the reference agents need this; a real agent reads the prompt.
    """
    tail = prompt.rsplit("Answer with exactly one of:", 1)
    if len(tail) == 2:
        return [o.strip() for o in tail[1].splitlines()[0].split("|") if o.strip()]
    if "Options:" in prompt:
        block = prompt.rsplit("Options:", 1)[1].splitlines()
        rows = [ln.strip()[2:].strip() for ln in block if ln.strip().startswith("- ")]
        # a lettered list asks for the letter, not the statement behind it
        lettered = [re.match(r"^([A-Z])\s*:\s*(.+)$", r) for r in rows]
        if rows and all(lettered):
            return [m.group(1) for m in lettered if m]
        return rows
    return []


# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------
def evaluate_lesson(agent: TextAgent, lesson: Lesson | str, *, n: int = 20, seed0: int = 0,
                    language: str = DEFAULT_LANGUAGE,
                    presentation: str | Presentation | None = None,
                    difficulty: float | None = None, strict: bool = False,
                    keep_wrong: int = 0) -> LessonResult:
    """Run one lesson's episodes past an agent and score them.

    ``seed0`` fixes which episodes are drawn: the same ``seed0`` gives the same
    questions to every agent, and a different one gives a fresh world. Nothing
    is held out, because nothing needs to be.
    """
    if isinstance(lesson, str):
        lesson = get(lesson)
    # resolve first, so a result records the canonical pack rather than the
    # alias the caller happened to type
    language = get_language(language).code
    pres = Presentation.parse(presentation).with_(language=language)
    fmt = pres.format
    lex = get_language(language).lexicon
    examples = list(lesson.examples(n, seed0=seed0, presentation=pres, difficulty=difficulty))
    correct = errors = 0
    wrong: list[dict[str, str]] = []
    for ex in examples:
        try:
            reply = agent(ex.prompt)
        except Exception as e:                     # an agent that throws scores zero
            errors += 1
            reply = f"<agent error: {type(e).__name__}: {e}>"
        vocab = fmt.reply_vocabulary(lex, ex.choices)
        s = score(reply, ex.target, vocab, strict=strict)
        correct += int(s >= 1.0)
        if s < 1.0 and len(wrong) < keep_wrong:
            wrong.append({"seed": str(ex.seed), "answer": ex.target,
                          "reply": str(reply)[:400]})
    rnd = statistics.fmean(1.0 / max(1, len(ex.choices)) for ex in examples)
    counts = Counter(ex.answer for ex in examples)
    majority = counts.most_common(1)[0][1] / len(examples) if examples else 0.0
    return LessonResult(
        lesson_id=lesson.id, tags=tuple(lesson.tags),
        level=lesson.level, n=len(examples), correct=correct,
        accuracy=correct / len(examples) if examples else 0.0,
        random_baseline=rnd, majority_baseline=majority,
        mean_choices=statistics.fmean(len(ex.choices) for ex in examples) if examples else 0.0,
        errors=errors, language=language, presentation=pres.key(), wrong=wrong)


def evaluate(agent: TextAgent, lessons: str | Sequence[str] | None = None, *,
             n: int = 20, seed0: int = 0, language: str = DEFAULT_LANGUAGE,
             presentation: str | Presentation | None = None,
             difficulty: float | None = None,
             strict: bool = False, keep_wrong: int = 0,
             progress: Callable[[str, int, int], None] | None = None) -> Report:
    """Evaluate a text agent across the curriculum.

    ``lessons`` may be ``None`` (every implemented lesson), a curriculum name
    such as ``"core170"``, a ``tag:`` or ``capability:`` selector, a
    comma-separated string of lesson ids, or a sequence of ids.
    """
    language = get_language(language).code
    chosen = [l for l in resolve(lessons) if l.status == "implemented"]
    results = []
    for i, lesson in enumerate(chosen):
        if progress:
            progress(lesson.id, i, len(chosen))
        results.append(evaluate_lesson(agent, lesson, n=n, seed0=seed0, language=language,
                                       presentation=presentation, difficulty=difficulty,
                                       strict=strict, keep_wrong=keep_wrong))
    return Report(results=results, language=language, n=n, seed0=seed0)
