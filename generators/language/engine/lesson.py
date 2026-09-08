"""What a lesson is, and what one episode of it looks like.

A lesson is a **procedural generator**, not a dataset. An episode is a pure
function of the seed it is handed: the same seed always yields the same
question, and two different seeds yield two different worlds. That is the
property the whole resource rests on, because it means an evaluation set is
never a held-out sample of something a model might have seen — it is a world
that did not exist until the evaluator asked for it.

It also means the ground truth is *computed*, never annotated. The generator
invents the grammar, the ontology, the causal graph or the proof calculus, and
then decides the answer by reading its own construction. There is no labeller
to disagree with.

A lesson does not know where it sits in any curriculum. It has no number and no
section; ordering and prerequisites are properties of a
:class:`~langcurriculum.curricula.Curriculum`, of which there may be several
disagreeing about the same lesson. What a lesson does declare is what it
teaches, what it exercises, and where it sits on the difficulty axes — facts
about the lesson itself rather than about anybody's ordering of it.

Replies are open-form text. The answer set is written into the prompt body by
the :class:`~langcurriculum.presentation.Presentation` rather than handed over as
a field, but it is still *retained*, because it is what makes a floor computable
and a floor is what makes an accuracy number mean anything. See
:mod:`langcurriculum.verify`.
"""

from __future__ import annotations

import hashlib
import inspect
import random
from dataclasses import dataclass, field
from typing import Any, Callable, ClassVar, Iterator, Mapping, Sequence

from ._structure import Term, sexpr, to_json
from .context import GenerationContext
from .generators.extra import ACTIVE_LANGUAGE
from .languages import DEFAULT_LANGUAGE, Language, get_language
from .presentation import DEFAULT_PRESENTATION, OPEN_FORMAT, Presentation

__all__ = ["Lesson", "Example", "AXES", "CORE_AXES", "EXTENDED_AXES",
           "LessonNotImplemented", "as_text", "instance_id"]

#: The eight core difficulty axes every lesson declares a position on, so that a
#: result is a profile over axes rather than a single number.
CORE_AXES = ("lexical_novelty", "grammar_complexity", "recursion_depth", "compositional_depth",
             "discourse_horizon", "world_complexity", "ambiguity", "reasoning_depth")

#: Axes introduced by the later lessons, where the core eight do not say what
#: makes a lesson hard. ``anytime_reasoning`` is not deep or ambiguous; it is
#: hard because the budget runs out, and that deserves its own dimension rather
#: than being folded into ``reasoning_depth``.
EXTENDED_AXES = ("uncertainty", "adversariality", "planning_horizon",
                 "representation_freedom", "computational_budget", "ontology_novelty")

#: Every axis a lesson may declare.
AXES = CORE_AXES + EXTENDED_AXES


class LessonNotImplemented(NotImplementedError):
    """Raised by a lesson that is specified but deliberately not generated.

    Exactly one lesson is in this state. It is kept in the registry, with its
    reasoning attached, rather than quietly dropped: a specification that cannot
    yet be graded honestly is still part of the curriculum's claim.
    """


def as_text(x: Any) -> str:
    """Render an answer or a choice as the plain text a caller will compare."""
    if isinstance(x, Term):
        return sexpr(x)
    if isinstance(x, bool):
        return "yes" if x else "no"
    return str(x)


def instance_id(lesson_id: str, seed: int, difficulty: float | None = None) -> str:
    """A stable id for the *problem*, independent of how it is presented.

    Every rendering of one episode — English, Turkish, rasterized, dictated —
    carries the same instance id, which is what makes agreement across surfaces
    measurable without a gold label or a judge. A system that internalized the
    structure answers a shared instance id the same way every time; one that
    learned a surface does not. See ``INTENT.md``.
    """
    d = "-" if difficulty is None else f"{difficulty:.6f}"
    return hashlib.blake2b(f"{lesson_id}|{seed}|{d}".encode(), digest_size=8).hexdigest()


@dataclass(frozen=True)
class Example:
    """One generated episode, entirely as plain data.

    ``observation`` is the question as rendered text. ``prompt`` is that plus the
    answer set, written in whatever way the presentation asks for. ``target`` is
    the expected open-form reply, which is not always the answer itself — under a
    lettered format it may be ``B``, or ``B: the red cube``.

    ``answer`` and ``choices`` are the canonical internal pair, retained so the
    floor stays computable and so a caller can rebuild any other format.
    """

    lesson_id: str
    seed: int
    language: str
    observation: str
    prompt: str
    answer: str
    choices: tuple[str, ...]
    target: str = ""
    instance_id: str = ""
    presentation: str = ""
    difficulty: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = {"lesson_id": self.lesson_id, "seed": self.seed, "language": self.language,
             "observation": self.observation, "prompt": self.prompt,
             "answer": self.answer, "choices": list(self.choices),
             "target": self.target, "instance_id": self.instance_id,
             "presentation": self.presentation, "metadata": self.metadata}
        if self.difficulty is not None:
            d["difficulty"] = self.difficulty
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Example":
        return cls(lesson_id=d["lesson_id"], seed=int(d["seed"]), language=d["language"],
                   observation=d["observation"], prompt=d["prompt"], answer=d["answer"],
                   choices=tuple(d["choices"]), target=d.get("target", d["answer"]),
                   instance_id=d.get("instance_id", ""),
                   presentation=d.get("presentation", ""),
                   difficulty=d.get("difficulty"),
                   metadata=dict(d.get("metadata") or {}))


class Lesson:
    """Base class. One subclass per lesson, one module per subclass.

    Subclasses set the class attributes below and provide ``generate``, a pure
    function of a :class:`random.Random` returning
    ``(observation, choices, answer, hidden)``. A generator that wants to know
    the episode's language or difficulty declares a second parameter and is
    handed a :class:`~langcurriculum.context.GenerationContext`; one that does
    not is called with the single argument, exactly as before.

    ``hidden`` is the part of the world the generator knows and the agent is not
    shown — the grammar it sampled, the boundary it drew, the lexicon it
    invented. It travels in ``Example.metadata["hidden"]`` for the evaluator's
    benefit and never appears in the prompt.
    """

    #: stable snake_case identifier, unique across the registry
    id: ClassVar[str] = ""
    #: difficulty level as declared by the lesson itself. A curriculum may
    #: override it; this is the lesson's own account of itself.
    level: ClassVar[int] = 0
    #: topical tags. Flat and many-per-lesson on purpose — the thing that used
    #: to be a section, minus the claim that lessons partition into one tree.
    tags: ClassVar[tuple[str, ...]] = ()
    #: one line saying what the lesson teaches
    teaches: ClassVar[str] = ""
    #: capability tags, for grouping results by ability rather than by lesson
    capabilities: ClassVar[tuple[str, ...]] = ()
    #: position on the difficulty axes; keys are drawn from :data:`AXES`
    axes: ClassVar[Mapping[str, int]] = {}
    #: a fixed answer vocabulary, when the lesson has one across all episodes
    answers: ClassVar[Sequence[Any] | None] = None
    #: ``implemented`` or ``spec``
    status: ClassVar[str] = "implemented"
    #: why, when ``status`` is not ``implemented``
    note: ClassVar[str] = ""
    #: whether the episode can be answered without being shown its answer set.
    #: False for almost every lesson, because most invent their vocabulary per
    #: episode; the open answer format is offered only where this is True.
    open_answerable: ClassVar[bool] = False

    generate: ClassVar[Callable[..., tuple[Term, Sequence[Any], Any, dict[str, Any]]]]

    #: set at class creation: whether ``generate`` accepts a context argument
    _wants_context: ClassVar[bool] = False

    def __init_subclass__(cls, **kw: Any) -> None:
        super().__init_subclass__(**kw)
        fn = cls.__dict__.get("generate")
        if fn is None:
            return
        fn = fn.__func__ if isinstance(fn, staticmethod) else fn
        try:
            params = inspect.signature(fn).parameters
        except (TypeError, ValueError):                  # pragma: no cover
            cls._wants_context = False
            return
        cls._wants_context = len(params) >= 2

    # ---- generation -------------------------------------------------
    def invoke(self, rng: random.Random, ctx: GenerationContext | None = None):
        """Run the generator against an existing random source.

        The single seam every caller goes through, so that the
        context-or-not calling convention is decided in one place. Reaching for
        ``type(lesson).generate`` directly works until some lesson grows a
        difficulty knob and then raises ``TypeError`` — which, at three call
        sites that wrapped it in ``except Exception``, meant four lessons
        silently vanishing from a vocabulary harvest rather than failing loudly.

        The composing lessons need this rather than :meth:`build` because they
        draw sub-episodes from a random source they already own.
        """
        if self.status != "implemented":
            raise LessonNotImplemented(f"{self.id}: {self.note}")
        fn = type(self).generate
        return fn(rng, ctx or GenerationContext()) if type(self)._wants_context else fn(rng)

    def build(self, seed: int, ctx: GenerationContext | None = None):
        """Run the generator once, from a seed."""
        return self.invoke(random.Random(seed), ctx)

    def supports_difficulty(self) -> bool:
        """Whether this lesson has a difficulty knob at all.

        A curriculum that wants a schedule rather than an ordering needs to know
        which of its nodes can actually provide one, and an honest ``False`` is
        far better than a knob that is accepted and ignored.
        """
        return type(self)._wants_context

    def example(self, seed: int = 0, *,
                language: str | Language | None = None,
                presentation: str | Presentation | None = None,
                difficulty: float | None = None) -> Example:
        """Generate one episode. The same seed and settings give the same episode.

        ``presentation`` carries the language, the answer format and the
        modality together; ``language`` is accepted on its own because that is
        what almost every caller has, and means "that language, everything else
        default".
        """
        pres = _resolve_presentation(presentation, language)
        lang = get_language(pres.language)
        fmt = pres.format
        if fmt.options == "hidden" and not self.open_answerable:
            raise ValueError(
                f"{self.id} does not declare open_answerable, so the {OPEN_FORMAT!r} "
                f"format would hide an answer set the episode cannot be solved without")
        ctx = GenerationContext(language=lang.code, difficulty=difficulty)
        # The morphology lessons draw their inflected material from the pack the
        # episode will be read in, so the generator has to know which that is. It
        # was read once at import from the default language, which is why an
        # agreement lesson asked for in Russian came out in English.
        token = ACTIVE_LANGUAGE.set(lang.code)
        try:
            obs, choices, answer, hidden = self.build(seed, ctx)
        finally:
            ACTIVE_LANGUAGE.reset(token)
        raw_opts = tuple(as_text(c) for c in choices)
        opts, answer_text, collapsed = _translate_options(lang, raw_opts, as_text(answer))
        observation = lang.render(obs)
        block = fmt.render_options(lang.lexicon, opts)
        prompt = f"{observation}\n\n{block}" if block else observation
        return Example(
            lesson_id=self.id, seed=seed, language=lang.code,
            observation=observation, prompt=prompt,
            answer=answer_text, choices=opts,
            target=fmt.render_target(lang.lexicon, opts, answer_text),
            instance_id=instance_id(self.id, seed, difficulty),
            presentation=pres.key(), difficulty=difficulty,
            metadata={"level": self.level, "tags": list(self.tags),
                      "teaches": self.teaches, "capabilities": list(self.capabilities),
                      "axes": dict(self.axes), "hidden": _plain(hidden),
                      **({"untranslated_options": True} if collapsed else {})},
        )

    def examples(self, n: int = 100, *, seed0: int = 0,
                 language: str | Language | None = None,
                 presentation: str | Presentation | None = None,
                 difficulty: float | None = None) -> Iterator[Example]:
        """Generate ``n`` consecutive episodes starting from ``seed0``."""
        for i in range(n):
            yield self.example(seed0 + i, language=language,
                               presentation=presentation, difficulty=difficulty)

    def structured(self, seed: int = 0, *, difficulty: float | None = None) -> dict[str, Any]:
        """The episode as plain JSON-able data rather than as text.

        This is the structural probe. A system can be right about the answer for
        surface reasons; it cannot produce the right *tree* for surface reasons,
        so comparing a recovered structure against this one is a much stronger
        signal than accuracy — and needs no judge, because the comparison is
        exact. See ``INTENT.md``.
        """
        obs, choices, answer, hidden = self.build(seed, GenerationContext(difficulty=difficulty))
        return {"lesson_id": self.id, "seed": seed,
                "instance_id": instance_id(self.id, seed, difficulty),
                "observation": to_json(obs),
                "choices": [as_text(c) for c in choices], "answer": as_text(answer),
                "hidden": _plain(hidden)}

    # ---- metadata ----------------------------------------------------
    def info(self) -> dict[str, Any]:
        """Everything the lesson declares about itself, as plain data."""
        return {"id": self.id, "level": self.level, "tags": list(self.tags),
                "teaches": self.teaches, "capabilities": list(self.capabilities),
                "axes": dict(self.axes), "status": self.status, "note": self.note,
                "supports_difficulty": self.supports_difficulty(),
                "open_answerable": self.open_answerable,
                "fixed_answers": [as_text(a) for a in self.answers] if self.answers else None,
                "description": (type(self).generate.__doc__ or "").strip()}

    @property
    def description(self) -> str:
        """The generator's own account of what it builds."""
        return (type(self).generate.__doc__ or "").strip()

    def __repr__(self) -> str:
        return f"<Lesson {self.id}: {self.teaches}>"


def _resolve_presentation(presentation: str | Presentation | None,
                          language: str | Language | None) -> Presentation:
    """Fold the two ways of naming a surface into one presentation."""
    pres = Presentation.parse(presentation) if presentation is not None \
        else DEFAULT_PRESENTATION
    if language is not None:
        code = language.code if isinstance(language, Language) else str(language)
        pres = pres.with_(language=code)
    return pres


def _translate_options(lang, options: Sequence[str], answer: str) -> tuple[tuple[str, ...], str, bool]:
    """Render the answer set in the prompt's language — unless that collapses it.

    The options have to be in the same language as the prompt, or the episode
    asks its question in one language and offers its answers in another. But a
    few lessons are *about* morphology, and their answer set is a set of
    inflected English forms: ``opens`` versus ``open``. A language without that
    inflection translates both to one word, and the episode stops being
    answerable at all.

    An answer set is a paradigm, so it is rendered whole or not at all. The
    translation is applied only when the pack knows **every** option and the
    result stays injective. A set of object ids or nonce forms is known to
    nobody and stays as it is, which is right — those are not words. A set of
    inflected English forms is known only in part, and stays as it is too, with
    ``metadata["untranslated_options"]`` recording that it did: a prompt whose
    options are visibly in another language is a much smaller problem than a
    prompt with two identical options and one correct answer.
    """
    # Ask the language, not one particular kind of lexicon. The old question
    # was put to `lang.lexicon.vocabulary`, which a derived language does not
    # have, so every one of them answered "no" to every option and offered
    # English answers against translated evidence.
    known = [lang.knows(o) for o in options]
    if not options or not all(known):
        return tuple(options), answer, any(known)
    translated = tuple(lang.token(o) for o in options)
    if len(set(translated)) != len(set(options)):
        return tuple(options), answer, True
    return translated, lang.token(answer), False


def _plain(d: Any) -> Any:
    """Coerce a hidden-state mapping to something ``json.dumps`` will accept."""
    if isinstance(d, Mapping):
        return {str(k): _plain(v) for k, v in d.items()}
    if isinstance(d, (list, tuple, set, frozenset)):
        return [_plain(x) for x in d]
    if isinstance(d, Term):
        return sexpr(d)
    if isinstance(d, (int, float, str, bool)) or d is None:
        return d
    return str(d)
