# The language capability, measured on the post-audit stream

FINDINGS §39 established that §19's language headline reproduces **only** on the
pre-audit stream (`hardening='none'`) and that the capability **had never been
measured on the post-audit distribution at all**. §43 sharpened why: on the
post-audit stream the *counting* oracle scores exactly the majority while the
*Dyck* oracle scores 1.000 — the §24 re-draw severed counting from balancedness.

This track measures it. Nothing under `tcn/` or `generators/` is touched
(`git diff main...HEAD -- tcn/ generators/` is empty).

**Every episode draw in this directory pins `hardening` explicitly.** That is the
one defect §39 exists to prevent, and `common.dataset` here takes `hardening` as
an argument with no silent inheritance of the generator default.

---

## Verdict, up front

**The language capability does not survive the re-drawn lesson in the form §19
reported it.** The two-stage method, run unchanged on an honestly-posed
post-audit split, **finds no conforming program**. The search **exhausted the
whole space** and returned certificate `complete` with **0 conforming** — so this
is a proved non-existence in the scaffold's family, not a search that ran out of
budget.

The shortfall is an **expressiveness limit, not a search failure**, and it is
precisely localised: §19's program is a *counting* program (accumulate ±1 per
symbol, test the total against a constant), and the §24 re-draw made bracket
counts identical across both classes. A scaffold that adds one thing the algebra
already has — a running **minimum** of the prefix sum, using the existing `min`
and `and` operators and no new domain operator — does contain an exact solution,
and the same unmodified search finds it. **Stage A survives untouched**: lexical
perception is exact and uniquely certified on the post-audit stream.

---

## 1. The stream, profiled before anything was designed

`stream_profile.py` → `stream_profile.json`. 1,200 episodes per row, `hardening`
pinned in every call.

| | lengths emitted | `#( == #)` rate | counting oracle | Dyck oracle | distinct strings |
|---|---|---|---|---|---|
| post-audit (`hardening='context_free_language'`) | 10,12,14,16,18,20,22 | **1.000** | **0.4992** | **1.000** | 841 / 1200 |
| pre-audit (`hardening='none'`) | 2,4,6,8,10,12,14,16 | 0.4992 | **1.000** | 1.000 | 187 / 1200 |

The counting oracle scores 0.4992 post-audit, which is exactly the positive rate:
`#( == #)` holds for **every** string, so "counts match ⇒ balanced" is the
constant `yes`. This reproduces §43's fact on an independent draw. Depths 1–4 are
present at every length in both streams, and post-audit the hardened draw picks
the pair count independently of the depth, so **length and depth are decorrelated
post-audit** — a length holdout no longer holds out depth as it did pre-audit.

Post-audit, `min_prefix` over the string is `0` (599), `-1` (407) or `-2` (194).
Since counts always match, **balanced ⟺ min prefix ≥ 0** on this stream. That is
the fact every result below turns on.

## 2. The split

`splits.py` → `splits.json`. The holdout is **string length**, the generating
structure of a Dyck word, exactly as the original track intended; the original's
training lengths {2,4,6} do not occur post-audit, which is why `final_eval.py`
dies with `ZeroDivisionError`. Train on the three shortest lengths that exist,
hold out the four longest.

| split | n | per-length | per-depth | distinct strings | positive rate | majority constant | share of strings seen in training |
|---|---|---|---|---|---|---|---|
| `train` | **24** | 10:3, 12:11, 14:10 | 1:4, 2:9, 3:5, 4:6 | 22 | 0.5000 | 0.5000 | 1.000 (by definition) |
| `heldout_seen_lengths` | **120** | 10:34, 12:42, 14:44 | 1:30, 2:35, 3:26, 4:29 | 93 | 0.4667 | 0.5333 | 0.125 |
| `heldout_unseen_lengths` | **859** | 16:221, 18:209, 20:215, 22:214 | 1:207, 2:216, 3:213, 4:223 | 700 | 0.5262 | **0.5262** | **0.000** |

**Overlap check: 0.0% of held-out-length strings occur in training.** Drawn from
`split='test'`, `seed0=100000`, disjoint from the training pool's seeds.

This is the **achieved** configuration, counted from the episodes actually run —
not a requested one.

### Splits tried

**One.** `TRAIN_LENGTHS=(10,12,14)`, `HELDOUT_LENGTHS=(16,18,20,22)`. It was
chosen from the length histogram in §1 before any search was run and was not
changed afterwards. The bound in §5 was computed before the large search, as the
brief required, and it predicted the negative result; the split was left alone
rather than tuned until something conformed.

## 3. Stage A — lexical perception. Survives, exactly, uniquely certified.

`run_stage_a.py` → `stage_a.json`. Scaffold `research/language-capability/scaffolds.py:stage_a`,
imported verbatim; search `tcn.search.enumerate_fit`, unmodified.

| | value |
|---|---|
| space size | **10,496** |
| evaluated | **10,496** |
| exhausted | **true** |
| conforming | **1** |
| **certificate** | **`unique`** |
| seconds | 2.65 |
| selection | `base = 14`, `open_byte = 40` (`'('`) |

Training: 12 episodes (lengths 12, 14; depths 1–4), 158 supervised positions.

| readout | episodes | positions | accuracy |
|---|---|---|---|
| held-out, seen lengths (10, 12, 14) | 40 | 482 | **1.000** |
| held-out, **unseen** lengths (16, 18, 20, 22) | 40 | 752 | **1.000** |

The address is computed (`base + pos`), so one module serves every position, and
it transfers to positions 14–21 that were never supervised. **Stage A is not
where the capability breaks.**

## 4. Stage B — the track's own method, run unchanged. No conforming program.

`run_stage_b.py` → `stage_b_p16.json`, `stage_b_p22.json`. Same scaffold, same
`enumerate_fit`, same 24 training episodes, same frozen stage-A module
(execution cost 5.0).

`positions` is the scaffold's own capacity parameter. The pre-audit run used 16,
which covered its longest string (14). Post-audit strings run to 22, so **both**
arms are reported: the literal unchanged default (16) and the length-matched
analogue (22).

| arm | nodes | space size | evaluated | exhausted | conforming | **certificate** | seconds | node evaluations |
|---|---|---|---|---|---|---|---|---|
| `positions=16` (unchanged default) | 84 | 45,375 | **45,375** | **true** | **0** | **`complete`** | 431.2 | 91,476,000 |
| `positions=22` (length-matched) | 114 | 45,375 | **45,375** | **true** | **0** | **`complete`** | 624.5 | 124,146,000 |

`complete` here means: the space was exhausted and the reported conforming set —
which is empty — is the whole of it. **No program in this scaffold's family fits
the 24 training episodes.** Accuracy is therefore not defined: there is nothing to
evaluate. This is falsification arm one of the two stated in advance.

## 5. The bound, computed before the search, and it predicted this

`bound.py` → `bound.json`. With stage A exact, every member of the stage-B family
computes

```
acc = plus * (#open in the first K symbols) + minus * (K - #open),  K = min(positions, length_bytes - c)
ans = op(acc, v),   op ∈ {eq, ge, le}
```

— a thresholded affine function of a bracket count over **one fixed prefix**. All
45,375 members were enumerated directly. The simulator was validated against the
real hardened program first: **150 random members checked, 0 mismatches**,
including the 6 that raise (unsigned `sub` underflow).

| positions | family members | members fitting train exactly | best train accuracy | best held-out-unseen accuracy in the family |
|---|---|---|---|---|
| 16 | 45,375 | **0** | 0.7500 | 0.7031 |
| 22 | 45,375 | **0** | 0.7500 | 0.7031 |

Confirmed against the real program, not the simulator (`verify_best_member.py` →
`verify_best_member.json`):

| member | train | held-out seen | held-out **unseen** (majority 0.5262) |
|---|---|---|---|
| `c=110, plus=-2, minus=+2, le -2` (best in family, selected **on the test split**) | 0.7500 | 0.5833 | **0.7031** |
| `c=101, plus=+1, minus=-1, eq 0` — the honest counting program, i.e. §19's semantics | 0.5000 | 0.4667 | **0.5262** = the majority, exactly |

Two things follow.

**(a) The counting program lands exactly on the majority.** It predicts `yes` on
every episode, because `#( == #)` always holds. That is §43's fact, reproduced
here through the actual typed program rather than through an oracle.

**(b) The family's only remaining signal is a truncation artifact.** The best
member reads only the first `L-9` symbols and asks whether opens outnumber closes
there — a *prefix-sum-at-one-fixed-cut-point* heuristic. It is real signal (a Dyck
word's prefixes are open-heavy) but it is not balancedness, and it is not
reachable honestly: **24 members tie at the best train accuracy of 0.75, and their
held-out accuracies range 0.4738 to 0.7031** (mean 0.6224). Training accuracy does
not identify which of the 24 to take, so even a tolerance-relaxed search would be
choosing by coin flip among members that differ by 23 accuracy points
(`analysis.json: best_train_members_p22`).

**(c) A closed-form ceiling for any member that reads the whole string.** If
`K ≥ L`, then `#open = L/2` always, so `acc = (plus+minus)·L/2` — a function of
length alone. The best predictor from length alone on the held-out split is
**0.5460** (`analysis.json: full_string_ceiling`), barely above the 0.5262
majority. A stage-B program that correctly reads the entire string is *worse* than
one that truncates.

## 6. Accuracy against all three baselines

`baselines.py` → `baselines.json`. Feature set imported verbatim from
`research/language-capability/baselines.py` so no baseline was weakened when the
stream changed. Each fitted number is the optimal deterministic predictor from
that feature, fitted on the 24 training episodes.

Held-out unseen lengths, n = 859, post-audit stream:

| | accuracy |
|---|---|
| **majority constant** | **0.5262** |
| **random** | **0.5000** |
| **best fitted feature** (`constant`; every feature collapses to the training prior) | **0.4738** |
| — *the prior itself is a 12/12 tie broken by insertion order; had it broken the other way every fitted row would read 0.5262, i.e. the majority. Either way no feature beats the majority.* | |
| `length` / `open_count_and_length` / `bigram_multiset` / `first_and_last` / … | 0.4738 (all) |
| training-string lookup (the §24 exploit) | 0.4738 |
| counting oracle | 0.5262 |
| Dyck oracle | **1.0000** |
| **stage-B synthesis, `positions=16`** | **none — no conforming program** |
| **stage-B synthesis, `positions=22`** | **none — no conforming program** |
| stage-B best member *selected on the test split* (not attainable) | 0.7031 |
| pre-audit-trained frozen program of §19, transferred | 0.4342 |

On `heldout_seen_lengths` (n = 120) the best fitted feature is `bigram_multiset`
at 0.600 against a 0.533 majority; on `train` (n = 24) it is `bigram_multiset` at
0.792 against a 0.500 majority. The exploit that §24 re-drew the lesson to kill —
copying the label of an identical training string — scores 0.4738 on held-out
lengths post-audit, against 1.000 on the pre-audit lesson's own training strings.
**The re-draw worked.**

### The §19 program transferred (context, not a result)

`analysis.json: transfer_preaudit_program`. The exported selection from
`research/language-capability/stage_b.json`, run on this honest post-audit split:
**0.4342** against a 0.5262 majority (train 0.500, held-out seen 0.4667). §39
reported 0.4733 on a differently-drawn set; this is the same phenomenon on a
properly posed split. It remains a program evaluated on a distribution it was
never fitted to, and is reported only so the number has a home.

## 7. Control: the same code reproduces §19 on the pre-audit stream

`control_preaudit.py` → `control_preaudit.json`. If the method fails post-audit,
that must be attributable to the stream and not to this re-implementation. The
identical code path — same `common.episode`, same imported scaffolds, same
`enumerate_fit` — with `hardening='none'` and train lengths {2,4,6}:

| | value |
|---|---|
| space size / evaluated / exhausted | 45,375 / 45,375 / true |
| conforming | **10** |
| **certificate** | **`complete`** |
| seconds | 223.4 |
| train accuracy | 1.0000 (majority 0.6250) |
| held-out seen lengths | 1.0000 (majority 0.5667) |
| **held-out unseen lengths** | **0.9986187845303868**, n = **724**, majority **0.5483425414364641** |

That matches FINDINGS §43's corrected figure and §39's table **to every recorded
digit**, including the episode count and the baseline. The pipeline is faithful;
the post-audit negative is about the stream.

Worth recording: the single error in 724 sits at **length 16** (per-length 1.000
at 8/10/12/14, 0.500 at 16), which is the one pre-audit length that exceeds the
16 positions the scaffold reads. The one blemish on §19's headline is the same
truncation mechanism that is the *only* remaining signal post-audit.

## 8. Is the post-audit task solvable at all in this family?

The brief's second question, and the more valuable half. Two different families
have to be distinguished.

**The stage-B scaffold's family: no.** Proved twice and independently — by
exhaustive enumeration with certificate `complete` and 0 conforming (§4), and by
direct enumeration of the family's semantics (§5). The reason is structural: the
scaffold's only accumulator is a **sum**, so every member is a function of a
bracket **count** over one prefix, and the §24 re-draw made bracket counts
identical in both classes by construction (`_count_preserving_negative` is a
transposition, which preserves the multiset exactly). Post-audit,
`balanced ⟺ min prefix ≥ 0`, and a **minimum is not a sum**. This is an
expressiveness limit of the scaffold. It is not a search failure: the search
decided all 45,375 programs, none of them at a budget cut.

**The typed algebra: yes, and the same search reaches it.** `dyck_scaffold.py`
adds exactly one structure to `scaffolds.stage_b` — a second accumulator carrying
the running minimum of the prefix sum — using `min` and `and`, operators already
in `tcn/operators.py`. No new domain operator, no change to core. The answer is a
conjunction of two readouts, and **both readouts are searched** over the same
`{eq, ge, le} × {-2..2}` grid stage B uses; nothing is wired by hand.

<!--DYCK-->

---

## What this means for the record

1. **§19 must always be quoted with its stream.** It is correct, it reproduces to
   six decimals (§7), and it is a statement about `hardening='none'`.
2. **The capability §19 demonstrated was, in the part that carries the headline,
   a counting capability.** Stage A — perception of a symbol at a computed address
   from raw bytes — is untouched and still exact on the post-audit stream. Stage B
   was solving a task whose exploitability §24 identified and removed, and on the
   re-drawn lesson its family provably contains no solution.
3. **The lesson re-draw did what it was for.** Nearest-string lookup fell from
   1.000 to 0.4738; the counting shortcut fell from 1.000 to exactly the majority.
4. **The gap is a scaffold gap, not a substrate gap**, and it is one operator
   wide. That is a concrete, cheap thing to fix, and it is the useful half of a
   negative result.

## Files

| file | what it is |
|---|---|
| `common.py` | episodes and decoding, `hardening` pinned at every call |
| `stream_profile.py` / `.json` | the distribution both streams actually emit |
| `splits.py` / `.json` | the split, per-length counts, overlap check |
| `run_stage_a.py` / `stage_a.json` | stage A, post-audit |
| `run_stage_b.py` / `stage_b_p16.json`, `stage_b_p22.json` | stage B, both position arms |
| `bound.py` / `bound.json` | the family bound, with its simulator validation |
| `verify_best_member.py` / `.json` | the bound's headline re-derived on the real program |
| `baselines.py` / `baselines.json` | majority, random, fitted features, oracles |
| `analysis.py` / `analysis.json` | §19 transfer, best-train members, full-string ceiling |
| `control_preaudit.py` / `.json` | the pre-audit reproduction control |
| `dyck_scaffold.py`, `run_dyck.py` / `dyck_p22.json` | the expressiveness bound |

## Reproduction

```
.venv/bin/python research/language-post-audit/stream_profile.py
.venv/bin/python research/language-post-audit/splits.py
.venv/bin/python research/language-post-audit/run_stage_a.py
.venv/bin/python research/language-post-audit/bound.py
.venv/bin/python research/language-post-audit/run_stage_b.py --positions 16
.venv/bin/python research/language-post-audit/run_stage_b.py --positions 22
.venv/bin/python research/language-post-audit/baselines.py
.venv/bin/python research/language-post-audit/analysis.py
.venv/bin/python research/language-post-audit/verify_best_member.py
.venv/bin/python research/language-post-audit/control_preaudit.py
.venv/bin/python research/language-post-audit/run_dyck.py --positions 22 --rank description
```

## Tests

`.venv/bin/python -m pytest tests/ -q` → **275 passed, 13 failed**. All 13
failures are environmental and pre-existing: they are the `computer` and `panel`
tests, which spawn `node --import tsx`, and neither `tsx` nor a `node_modules`
tree exists in this checkout (`RuntimeError: computer session did not start`).
**Verified by removing `research/language-post-audit/` entirely and re-running:
the same 13 fail, identically.** Nothing in this track is imported by any test,
and `git diff main...HEAD -- tcn/ generators/` is empty.
