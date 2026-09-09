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
