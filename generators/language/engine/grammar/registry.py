"""Which languages exist, and what each one is actually backed by.

Two kinds of language live here and the difference between them is the only
thing in this module worth understanding.

A **hand-written** grammar is a Python module somebody wrote and somebody can
check. It has a verified lexicon, a real phonology where the language needs one,
and overrides for the things its language does that no parameter set predicts.
There are a handful, and there will only ever be a handful, because each one is
a day of work and a speaker's afternoon.

A **derived** grammar is assembled at load time from
:mod:`~langcurriculum.grammar.typology` (word order and friends, from WALS),
Wiktionary (the lexicon) and UniMorph (the morphology). There are over a
thousand available and around a hundred and seventy with enough data to be worth
presenting. Nobody has read them.

Both produce sentences through the identical engine. The registry keeps them
distinguishable so that no caller has to guess, and the tiers are checked
against the database rather than asserted by hand:

==== =========================================================== ==========
tier what backs it                                               verified?
==== =========================================================== ==========
1    hand-written grammar, curated lexicon, real phonology       yes
2    derived grammar, imported lexicon **and** morphology        no
3    derived grammar, imported lexicon, no morphology            no
4    derived grammar only — typology but nothing lexical         no
==== =========================================================== ==========

Construction is **lazy**. Instantiating a derived grammar costs a handful of
SQLite queries, and instantiating seventeen hundred of them at import time to
serve a caller who wants one would be indefensible. They are built on first
request and cached.
"""

from __future__ import annotations

import threading
from typing import Iterator, Mapping

from .derived import DerivedGrammar
from .linearize import Grammar
from .store import DB_PATH, LanguageDB

__all__ = ["LanguageRegistry", "REGISTRY", "get_grammar", "grammar_codes",
           "register_grammar", "DEFAULT_MIN_TIER"]

#: languages below this tier are not registered by default: a grammar with no
#: lexicon at all renders every content word in English, which is not a
#: presentation of the episode in that language and should not be offered as one
DEFAULT_MIN_TIER = 3


class LanguageRegistry:
    """Hand-written grammars, plus every derived one the database supports."""

    def __init__(self, db: LanguageDB | None = None, *,
                 min_tier: int = DEFAULT_MIN_TIER):
        self.db = db if db is not None else LanguageDB()
        self.min_tier = min_tier
        self._handwritten: dict[str, Grammar] = {}
        self._derived: dict[str, Grammar] = {}
        self._aliases: dict[str, str] = {}
        self._lock = threading.Lock()
        self._available: frozenset[str] | None = None

    # ---- registration ----------------------------------------------------
    def register(self, grammar: Grammar, *, aliases: Iterator[str] = ()) -> Grammar:
        """Add a hand-written grammar. It outranks any derived one for its code."""
        if not grammar.code:
            raise ValueError("a grammar needs a code")
        self._handwritten[grammar.code] = grammar
        for alias in aliases:
            self._aliases[alias] = grammar.code
        return grammar

    # ---- lookup ----------------------------------------------------------
    @property
    def available(self) -> frozenset[str]:
        """Every code that can be built, hand-written or derived."""
        if self._available is None:
            derived: set[str] = set()
            if self.db.exists():
                derived = set(self.db.codes(min_tier=self.min_tier))
            self._available = frozenset(derived | set(self._handwritten))
        return self._available

    def get(self, code: str) -> Grammar:
        key = self._aliases.get(code, code)
        hand = self._handwritten.get(key)
        if hand is not None:
            return hand
        with self._lock:
            cached = self._derived.get(key)
            if cached is not None:
                return cached
            if not self.db.exists():
                raise LookupError(
                    f"{key!r} is a derived language and the language database "
                    f"is not built. Run `python scripts/build_langdb.py --fetch` "
                    f"or use one of {sorted(self._handwritten)}.")
            if key not in self.available:
                raise LookupError(
                    f"unknown language {code!r}. "
                    f"{len(self.available)} are available; "
                    f"hand-written: {sorted(self._handwritten)}")
            grammar = DerivedGrammar(self.db, key)
            self._derived[key] = grammar
            return grammar

    def codes(self, *, tier: int | None = None) -> list[str]:
        if tier is None:
            return sorted(self.available)
        out = [c for c in self._handwritten] if tier == 1 else []
        if self.db.exists():
            out += [r["code"] for r in self.db.languages(min_tier=tier)
                    if r["tier"] == tier and r["code"] not in self._handwritten]
        return sorted(set(out))

    def tier_of(self, code: str) -> int:
        if code in self._handwritten:
            return 1
        row = self.db.language(code) if self.db.exists() else None
        return row["tier"] if row else 4

    def summary(self) -> dict[str, int]:
        """How many languages sit at each tier — the coverage claim, computed."""
        out = {1: len(self._handwritten), 2: 0, 3: 0, 4: 0}
        if self.db.exists():
            for row in self.db.languages(min_tier=4):
                if row["code"] in self._handwritten:
                    continue
                out[row["tier"]] = out.get(row["tier"], 0) + 1
        return out

    def __len__(self) -> int:
        return len(self.available)

    def __repr__(self) -> str:
        s = self.summary()
        return (f"<LanguageRegistry {len(self)} languages: "
                f"{s[1]} hand-written, {s[2]}+{s[3]} derived>")


#: the process-wide registry
REGISTRY = LanguageRegistry()


def register_grammar(grammar: Grammar, **kw) -> Grammar:
    return REGISTRY.register(grammar, **kw)


def get_grammar(code: str) -> Grammar:
    return REGISTRY.get(code)


def grammar_codes(*, tier: int | None = None) -> list[str]:
    return REGISTRY.codes(tier=tier)
