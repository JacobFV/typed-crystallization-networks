# `research/second-family` — does §52 + §54 replicate on a different task family?

§52 (semantic pooling fixes abstraction identity) and §54 (breadth-weighted
ranking fixes transfer) are both measured entirely on one family: §44's six
Boolean tasks around a `MAJ3` window over four inputs. §54's own closing caveat
is *"one family, seven tasks — anecdote-strength on breadth"*, and the partial
second-family check it ran was the same six task shapes with the window swapped.

This track builds a genuinely different family — the shared abstraction is
`W4(w,x,y,z) = (w⊕x) ∧ (y⊕z)`, arity 4 over five Boolean inputs, non-monotone,
non-threshold, three gates, with an 8-element hole-symmetry group instead of all
of S3 — and runs §52's and §54's arms on it unchanged.

**Result: §52 replicates, §54 does not.** See `RESULTS.md`.

## Layout

| file | what it is |
|---|---|
| `PREREGISTRATION.md` | committed before any arm; Amendment 1 appended after the primary tables and labelled exploratory |
| `RESULTS.md` | the family's construction, the arm table, the leave-one-out table, the verdict |
| `_paths.py` | puts §44's, §46's, §52's and §54's harnesses on the import path |
| `family.py` | the family, the off-family twin, and the corpora — replaces §54's `family.py` by module name |
| `evaltasks.py` | the two-node compact scaffold, the held-out tasks and the two controls — replaces §52's by module name |
| `patches.py` | re-points `pool.is_window` at this family, delegating to the original for §52's names; nothing upstream is edited |
| `armlib.py` | the arms, the two hand-authored 4-ary modules, the per-band libraries |
| `mining.py` | drives `mine_multi` / `mine_semantic` and resolves the pre-registered selection rules |
| `run_mine.py` | mines both identities per band and publishes every arm's module |
| `run_arms.py` | §52's arm table on `L1_w4_bdae_xor_c`, plus the controls |
| `run_rank.py` | §54's six corpora × six objectives, per band |
| `run_heldout.py` | the leave-one-out transfer measurement |
| `run_sensitivity.py` | the breadth-exponent sweep and the rule's parameter sweep |
| `run_b1alt.py` | Amendment 1 — the class B1's tie-break did not pick |
| `run_grad.py` | the gradient panel, at the configuration a 4-ary module allows |
| `tables.py`, `show_*.py`, `verify_claims.py` | joins and checks over `out/*.json`; every table in `RESULTS.md` comes from these |

Nothing in `tcn/` or `generators/` is modified, and no upstream research file is
edited: the family is supplied by shadowing two module names and re-pointing one
function.
