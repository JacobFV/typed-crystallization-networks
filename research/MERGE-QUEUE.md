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
