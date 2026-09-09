# Merge queue

Branches that are finished and verified but deliberately **not** merged yet,
with the reason. Read this before merging anything.

## The rule

Do not merge a change to `tcn/` or `generators/` while an agent is running
experiments against the main checkout. Their measurements are taken against
whatever core they started with; changing it underneath them silently mixes two
implementations into one result set. Merge only when the main tree is quiet, or
when the waiting agent's work provably does not touch the changed code.

Check first: `ListAgents`, plus `ps -eo args | grep research/` for detached runs.

## Waiting

### `perturbation-selection` (90dd08a) — SUPERSEDED by `loss-gated-eligibility`

`loss-gated-eligibility` (ec953bc) is built on this commit and contains it
(`git merge-base --is-ancestor` confirms). Merge that branch instead; merging
this one separately is redundant.

Original entry retained below for its verification record.

### `perturbation-selection` (commit 90dd08a) — verified, blocked on quiet tree

Replaces entropy+stability selection with the DARTS-PT removal rule, and adds
`Objective(total, task)` so the connectivity guard probes the unregularized task
loss. 96 tests pass in its worktree.

**Independently verified from the supervising session**, not taken on the agent's
word: `96 passed in 360.92s` run against that worktree, with the gitignored
`node_modules` symlinked in for the four computer tests and the symlink removed
afterwards, leaving the worktree clean at 90dd08a.

Blocked because it changes `tcn/crystallize.py`, `tcn/synthesis.py`, `tcn/cli.py`
and `JointTrainer.episode`, while the perception-ladder and
recursive-abstraction-retest agents are measuring against main's current core.

Merge when both have finished and their results are landed. After merging,
re-run `tcn train --episodes 160` and confirm it still reports 0.2488 -> 0.0022,
4/4 deterministic and 4/4 frozen; the agent reports three disconnection
deferrals instead of four, which is expected from the corrected guard and is
already reflected in its `docs/VALIDATION.md` edit.

Note for a fresh worktree: the four `generators/computer` tests need
`generators/computer/engine/node_modules`, which is gitignored. Symlink it from
the main checkout to run the full suite.

### `positional-reuse` — verified, blocked on quiet tree

Refutes the positional-reuse blocker: one crystallized module applies at every
position of an arbitrarily wide tuple with **three caller nodes**, independent of
width and position count, using only operators already in the registry
(`insert` to bridge into set-land, `pair` for the cartesian product with a
constant set of positions, `map` with the module parameter). No operator, type,
relaxation or loss term is added, and no ARCHITECTURE section 2 amendment is
proposed.

Also fixes a genuine search bug: `legal_candidates` resolved every candidate
with empty parameters, so no parameterized operator could ever be proposed.

**Independently verified from the supervising session, against main's operators
rather than the branch:**

- `legal_candidates` on main proposes **0** candidates for `project`, `map`,
  `filter` and `join`, and 1 for `and`. `project` with empty parameters raises
  `TypeError: operator signature mismatch`; with `index=0` it resolves. The
  entire structural and recursive-abstraction family was outside the search
  space, so positional reuse could only ever be hand-supplied, never discovered.
- The three-node composition runs on **unmodified main**: a registered
  `(index, wide) -> bool` module applied across an 8-wide observation by exactly
  3 caller nodes. Confirmed the claim needs no core change to be *expressible* —
  the branch's changes are what let search *find* it.
- Confirmed the position tag is load-bearing: an untagged `set[BOOL]` output
  collapses to `[False, True]`, losing per-position information, exactly as the
  duplicate-free set semantics require.

Blocked because it changes `tcn/graph.py` and `tcn/scaffold.py` while the
perception-ladder and recursive-abstraction-retest agents measure against main.

Its own reported caveats to preserve when landing: the geometry demonstration's
supervision does not identify the program (1,584 of 32,000 candidates survive
held-out, the renderer's own among them), the whole pattern sits behind a
declared gradient boundary since `map`/`pair`/`insert` have no relaxation, and
73% of apply time is `Value.of` re-encoding because `pair` replicates the
observation per position.

## Merge order, checked in advance

Dry-run with `git merge-tree --write-tree`, no working tree touched:

- `positional-reuse` against main: **clean**
- `perturbation-selection` against main: **clean**
- the two against each other: **clean**

They overlap on `generators/logic/generator.py` and `tests/test_generators.py`
only, and git resolves both without conflict. Either order works.

Merge `perturbation-selection` first: it is the smaller behavioural change to
the shipped path and its re-verification is the sharper tripwire (`tcn train
--episodes 160` must still give 0.2488 -> 0.0022, 4/4 deterministic, 4/4
frozen, with three disconnection deferrals rather than four). If that holds,
merge `positional-reuse` and re-run the suite again.

After both land, two things follow immediately and should not be forgotten:

- The `legal_candidates` parameter fix changes the candidate space, so any
  scaffold that enumerated candidates before the merge may now see more of
  them. Re-run the shipped fixtures and confirm the numbers before trusting
  any comparison that straddles the merge.
- `research/discrete-perception` was measured against pre-merge main and was
  explicitly told to supply `project`/`map`/`filter` candidates by hand. Its
  numbers remain valid for what they measured; do not silently restate them as
  post-merge results.

### `loss-gated-eligibility` (ec953bc) — verified, blocked on quiet tree

Contains `perturbation-selection` and supersedes it. Adds `eligibility` and
`anneal` rules to `Crystallizer`, both **defaulting to the pre-existing
behaviour** (`eligibility="immediate"`, `anneal="round"`), verified by reading
the constructor on the branch. Note the inherited `selection="perturbation"`
default from 90dd08a IS a behavioural change; that belongs to the parent commit.

Verified from the supervising session: contains 90dd08a,
`git merge-tree` shows it conflict-free against `positional-reuse`, and
**`104 passed in 64.89s`** run against that worktree rather than taken on the
agent's word (gitignored `node_modules` symlinked in for the four computer
tests, symlink removed afterwards, worktree left clean at ec953bc).

Its verdict is the third independent confirmation that the scheduler does not
beat plain argmax, and the agent killed its own apparent win: not-annealing
conformed 48/48 against step-matched argmax's 42/48 (p = 0.0265), but that arm
spends about twice argmax's forward passes, and a forward-matched argmax
conforms 32/32. The gain did not survive the second compute axis.

The mechanism it identified is the more general result and should survive into
any future scheduler work: under the gate, perturbation and argmax both name
the reference candidate 11/16 and agree on 9 of 16, whereas at the incumbent
schedule it is 5/16 against 0/16. **The interval in which the perturbation
measurement beats argmax is exactly the interval in which committing is a
mistake.** For DARTS-PT-style selection to pay here, the freeze would have to
be reversible.

Also worth keeping: gating eligibility while leaving the temperature on the
round clock is actively broken, because waiting rounds are counted by the
anneal, so the temperature floors before the objective settles. The two clocks
must come off together.

### `abstraction-preference` — verified, CONFLICTS, merge by hand

Adds `SoftProgram.description_cost()` (expected description bits of the pruned
hardened program under the choice distribution, equal to
`export().pruned().description_bits()` at any one-hot), wires it into
`synthesis.fit(mdl_weight=)` and `TrainConfig.description_weight`, both off by
default; deduplicates `exact_tensor` rows; prunes dead nodes at
`register_module` and `save_program`; and adds
`enumerate_fit(rank='order'|'description'|'cost')` with a `conforming` count.

**`git merge-tree` shows two conflicts against current main**, unlike the other
branches: `tcn/training.py` and `research/FINDINGS.md`. It was cut before the
positional-reuse and loss-gated merges landed, so it does not contain them.
Merge by hand, keeping both sides: main has the merged `Objective(total, task)`
signature and the gate rules in `training.py`, the branch adds
`description_weight`.

Verified from the supervising session: `complexity()` charges a module call its
whole body cost and counts dead nodes, so the branch's correction to the
re-test's "20.43 for both" is right and is now recorded in FINDINGS section 12.

Report its trade honestly when landing: the description term takes the tight
scaffold from 19/24 to 2/24 conformance (p = 1.1e-6), because the pressure is
"make fewer nodes live", which is aligned on an over-provisioned scaffold and
opposed on an exactly-sized one. Defaults are off precisely because of that.
