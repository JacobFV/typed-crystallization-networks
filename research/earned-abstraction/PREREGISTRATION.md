# Pre-registration — written before the four arms were measured

Written 2026-09-09, after the corpus solver and the selection rule were
implemented and machinery-tested (`test_mine.py`), and **before** any arm of the
later task was run with an earned module. At the time of writing, the only arm
that had been executed was `arm1_none` on the tight scaffold, which reproduces
the retest's published figure (230 400 programs, exhausted, no solution) and is
therefore not an outcome of this track.

## What is fixed in advance

* **The later task** is `maj(a,b,c) xor maj(d,e,f)` over six Boolean inputs, on
  the tight (3-node) and wide (9-node) scaffolds of
  `research/recursive-abstraction-retest`. It was chosen because that track is
  the current state of the art for this question and supplies a measured
  hand-authored arm. The rule never sees it: the corpus is a disjoint family of
  4-input tasks and the rule's only inputs are the solved programs.
* **The rule** is `mine.propose` as documented at the top of `mine.py`, with
  `MAX_NODES = 5`, `MAX_HOLES = 4`, `MIN_TASKS = 2`. These are declared before
  the run and a sensitivity sweep over them will be reported whatever the
  headline says.
* **The arms** differ only in the library:
  1. `arm1_none` — no library;
  2. `arm2_earned` — the rule's rank-1 proposal, published to and loaded from a
     real `tcn.library.Library`;
  3. `arm3_authored` — the retest's hand-authored minimal MAJ3;
  4. `arm4_wrong_authored` — the retest's hand-authored distractor, truth table
     134, same node count and same arity;
  5. `arm4b_wrong_mined` — the rule's highest-ranked runner-up of the same
     arity computing a different function. This is a control the brief did not
     ask for and is the sharper one: same mining machinery, different selection.

## What would falsify the claim

* **The rule found nothing worth abstracting** if `arm2_earned` does not beat
  `arm1_none` — on the tight scaffold that means it does not turn an empty
  conforming set into a non-empty one; on the wide scaffold it means no gain in
  success rate or in steps to first conforming export.
* **The selection contributed nothing** if `arm2_earned` ties
  `arm4_wrong_authored` or `arm4b_wrong_mined`. Then any module of that size
  helps and the result is about scaffold shape, not abstraction.
* **The rule is worse than a human** if `arm2_earned` loses badly to
  `arm3_authored` — a real and publishable negative.
* **The rule is degenerate** if its rank-1 proposal is a 1-node fragment, or if
  no fragment clears `MIN_TASKS`. Either is a legitimate reported outcome.

Whatever comes out is recorded as it comes out. The rule is not to be re-tuned
until the number improves; a sensitivity sweep over the three declared
parameters is reported, and if the headline moves under it, that is reported
too.

## What I already expect to be uninformative

`arm1_none` on the tight scaffold cannot solve the task at all — the flat
minimum is proved >= 7 gates against a 3-node scaffold — so "earned beats
no-library" is guaranteed there and carries no information on its own. The
informative comparisons on the tight scaffold are against arms 3, 4 and 4b; the
non-degenerate search comparison is the wide scaffold, where both routes fit.
