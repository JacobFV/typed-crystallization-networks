# Automatic backend selection: the rule, its thresholds, and what it gets wrong

Branch `worktree-agent-afc0484c7c0dc4724`, based on `main` at `d54135c`
(description-cost term, `exact_tensor` row deduplication, dead-node pruning,
`enumerate_fit(rank=)`).

Six tracks measured where enumeration wins and where relaxation wins. This
implements the rule they support, wires it into `synthesis.fit` as an opt-in
`mode="auto"`, and measures the selector itself on eleven cases — including the
five families the task names: small-exact, large-space, continuous-constant,
dead-surrogate and environment-coupled.

**Headline.** The selector agrees with the tracks on **11 of 11** cases when its
decision is scored directly, and **10 of 11** end to end through
`synthesis.fit(mode="auto")`. The one disagreement is not a threshold error: it
is a missing backend capability (`enumerate_fit` has no minimum-Hamming
objective), and the cost of it is 0.05 s on a 0.9 s problem because the failed
sweep returns a completeness certificate and the run falls through to the
relaxation. Every threshold's provenance is below, and the two guesses are
labelled as guesses.

---

## 1. What was added

| file | what |
|---|---|
| `tcn/select.py` | new. `liveness`, `enumeration_cost`, `hybrid_cost`, `select_backend`, `hybrid_fit` |
| `tcn/learning.py` | `carrier_temperature`; `relaxed(..., carrier_scaled=False)`; `SoftProgram.surrogate_scale` and `.scale_surrogates()` |
| `tcn/synthesis.py` | `fit(mode=..., validation=..., rollout_cost=...)`, default `mode='relax'` |
| `tcn/cli.py` | `tcn synthesize --mode`, default `relax` |
| `tests/test_selection.py` | 18 tests |
| `research/search-selection/` | `common.py` (the cases), `liveness.py`, `suite.py`, `fixtures.py`, `report.py`, `out/` |

Nothing shipped changes. `mode` defaults to the gradient path, `surrogate_scale`
defaults to 1.0 everywhere, and `carrier_scaled` defaults to False. Verified two
ways: `examples/mixed.py` through the default `fit` still reports relaxed loss
**1.0066e-06** and exact error **0.0**, the numbers FINDINGS section 10 records;
and the `SoftProgram` forward is **bit-identical** on six fixtures against the
pre-change `relaxed` contract, tensor by tensor (`liveness.py`,
`fixtures_unchanged`).

---

## 2. The rule

```
    0.  no free discrete choice        -> hybrid if a live constant remains, else enumerate
    1.  environment-coupled and a full sweep exceeds the rollout budget
                                       -> relax   (or enumerate, flagged infeasible, if relaxation is dead)
    2.  enumeration affordable         -> hybrid if a live trainable constant exists AND the
                                          hybrid is itself affordable; else enumerate
    3.  otherwise                      -> relax if the relaxation is alive
                                       -> enumerate, flagged infeasible, if it is not
```

The **order** is the claim, and the claim is the asymmetry of the costs.
Choosing enumeration costs at most `projected_enumeration_seconds`, a number
that has already been measured, and it never returns a program that was not
checked. Choosing relaxation costs a run that may return nothing, with no
certificate in either direction. So the affordability test comes before the
relaxation test, and the relaxation is reached only once enumeration is priced
out. Section 6 quantifies that asymmetry: on the eight cases where both arms
ran, choosing enumeration when relaxation was right cost between 0.6x and 176x a
gradient run, while choosing relaxation when enumeration was right cost between
1.6x and 16x **and** lost the certificate — but the enumeration side's worst
case is bounded by a number the selector already knows, and the relaxation
side's is not.

### Every input is measured on the scaffold and the data at hand

| input | where it comes from | measured or declared |
|---|---|---|
| `space_size` | `tcn.search.space_size` | measured |
| seconds per exactly-evaluated program | `enumeration_cost`, 12 sampled selections through the same `evaluate` | measured, on this host |
| seconds per structure for the hybrid | `hybrid_cost`, 3 sampled structures screened and fitted | measured, on this host |
| trainable constants **this supervision can move** | `d probe_loss / d constant` at the initializations | measured |
| `gradient="none"` boundaries | operator contracts, plus the global choice-gradient reading | measured |
| surrogate liveness at the data | section 3 | measured |
| environment coupling | `rollout_cost=` argument | **declared by the caller** |

`rollout_cost` is the one declared input, and it has to be: nothing in a
`Program` records where its examples came from. Everything else is read off the
scaffold and the data.

---

## 3. The surrogate-liveness check

This is the part that would have prevented the wrong "address relaxation is
worse than chance" conclusion (FINDINGS section 16). It is two measurements,
because relaxation fails for two different reasons and only both together say
"relaxation is viable here".

**Local, per candidate, per example.** Each candidate's relaxation is evaluated
on the values that actually reach it in the soft forward pass, and its
derivative with respect to those values is read off by autograd **for each
example separately**. A candidate live on no example is dead: it contributes
nothing to its node's mixture and receives nothing back, so no descent can ever
select it.

Per example and not in aggregate, because that distinction is exactly the
failure. On the recorded rung-3 free-address scaffold the two colour candidates
read:

| candidate | peak gradient over the batch | examples with a representable gradient |
|---|---|---|
| `eq(a, k=24)` | 1.15e-29 | **0 of 12** |
| `eq(a, k=30)` | 2.02e-02 | 5 of 12 at R=2, 1 of 12 at R=4 |

A peak taken over the batch reads the node as differentiable. Per example, the
candidate the reference program needs is dead everywhere. The reference program
is `eq(red, 24) AND eq(green, 30)`, so gradient descent could not have found it
at any budget.

**Global, per node.** The gradient of the *unregularized* probe loss with
respect to each free node's choice logits. A node reading zero has no learning
signal for its choice at all — which is what a `gradient="none"` operator
anywhere downstream produces.

FINDINGS section 3 warns that treating an all-zero gradient as disconnected is
wrong inside the crystallizer, because a legitimately concentrated choice has
zero *entropy* gradient. That objection does not apply here and the code says
why: this is measured at initialization, before any concentration, against the
task objective with no regularizer attached.

### The `eq` reading, checked rather than quoted

| \|a-b\| | `eq` at tau=1 | d/da at tau=1 | `eq` at tau=2^8 | d/da at tau=2^8 |
|---|---|---|---|---|
| 1 | 0.368 | -0.736 | 0.996 | -0.00778 |
| 4 | 1.13e-07 | -9e-07 | 0.939 | -0.0294 |
| 8 | 1.6e-28 | -2.57e-27 | 0.779 | -0.0487 |
| 10 | 3.78e-44 | -7.57e-43 | 0.677 | -0.0529 |
| **11** | **0** | **-0** | 0.623 | -0.0536 |
| 25 | 0 | -0 | 0.087 | -0.017 |
| 64 | 0 | -0 | 1.13e-07 | -5.63e-08 |
| 128 | 0 | -0 | 1.6e-28 | -1.6e-28 |
| 255 | 0 | -0 | **0** | -0 |

The threshold is **|a-b| = 11**, confirming FINDINGS section 16's corrected
figure (the original report said 12) and its note that 1.6e-28 is still
representable at 8. Note the last row: `tau = 2^bits` does not make the surrogate
alive everywhere, it moves the dead zone from 11 to about 90 on a byte carrier.
That is why the check reports a *fraction* rather than a boolean.

### Reachable fraction: how much of the space descent can see

The share of the declared discrete space gradient descent could reach at all —
the product over free nodes of (live candidates)/(all candidates). It is the
single number the check produces, and it is a strictly stronger statement than
"the surrogate is fine".

| case | space | reachable, shipped | reachable, carrier-scaled | dead nodes | candidates dead on every example |
|---|---|---|---|---|---|
| `mixed` | 96 | 0.875 | 0.875 | -- | 2 -> 2 |
| `mixed_constant` | 192 | 0.875 | 0.875 | -- | 2 -> 2 |
| `joint_offline` | 256 | 0.7656 | 0.7656 | -- | 4 -> 4 |
| `rung3_R2` | 576 | **0.25** | **1** | -- | 2 -> 0 |
| `rung3_R4` | 9,216 | **0.25** | **1** | -- | 2 -> 0 |
| `rung3_bytes` | 9,437,184 | **0.0289** | **0.336** | -- | 425 -> 215 |
| `gradient_boundary` | 12 | 1 | 1 | **byte** | 1 -> 1 |
| `boundary_wide` | 480,000 | 1 | 1 | **byte, tail** | 1 -> 1 |
| `noisy_2flip` | 4,096 | 0.6699 | 0.6699 | -- | 6 -> 6 |
| `wide_logic` | 3,388,291,200 | 1 | 1 | -- | 0 -> 0 |

Three things fall out that were not the point of the exercise:

- **`mixed`, `mixed_constant` and `joint_offline` are already at 0.875 and 0.766
  without any images in them.** `truth_0` and `truth_15` are the constant
  functions, whose relaxations are constant in their inputs, so gradient descent
  literally cannot select them at any node. FINDINGS F-bench records that 40% of
  `logic` targets at depth 8 are constant functions; on any such target the
  gradient path is searching a space that does not contain the answer. This is
  a measurable statement about the shipped fixtures, made by the check, and it
  is not in FINDINGS.
- **The gradient boundary is detected as a *node* property, not a program
  property.** On `gradient_boundary` the address node `byte` reads dead while
  the arithmetic node `scale` downstream of `pack` stays live. The relaxation is
  not broken, one choice in it is unmakeable.
- **`rung3_bytes` still only reaches 0.336 after the fix.** Two thirds of the
  256-value colour pool is more than ~90 from every observed byte, so those
  candidates are dead even at `tau = 2^bits`. The selector reports it rather
  than claiming the fix is complete.

### Where the fix belongs: in `relaxed`, behind a separated temperature

The measured fix (FINDINGS section 16) is `tau = 2^bits`, read off the declared
carrier. It is implemented in `relaxed` itself, as `carrier_temperature(t)`
under an opt-in `carrier_scaled` flag — not in the selector — because it is a
property of the operator's contract at that carrier, not of any one search.

But it could not simply be turned on, because of the coupling FINDINGS section
16 flags as D2: `SoftProgram` divided **both** the candidate softmax **and** the
operator relaxation by one temperature per node, so widening a surrogate flattens
that node's choice distribution at the same time. That is why the address-wall
track needed a "compensated" arm with a hand-multiplied learning rate.

**The two are now separated.** `SoftProgram.surrogate_scale[node]` is a second
per-node factor that multiplies the temperature *only* on the path into
`relaxed`; `temperatures[node]` alone still divides the softmax, the entropy and
the description cost. It defaults to 1.0, so:

- every shipped run is bit-identical (verified on six fixtures, tensor by tensor);
- the crystallizer's 0.8x anneal still sharpens surrogates proportionally,
  because it scales the shared term;
- `scale_surrogates()` widens `eq` **without touching the choice distribution** —
  asserted directly in `test_scale_surrogates_names_only_the_eq_nodes_and_leaves_choices_alone`,
  which compares `distributions()` before and after.

Measured effect, at 800 gradient steps with explicit initialization noise 0.5,
3 seeds, held-out exactness:

| arm | `rung3_R2` (576) | `rung3_R4` (9,216) |
|---|---|---|
| shipped surrogate | 0/3 | 0/3 |
| carrier-scaled | **3/3** | 1/3 |

The direction reproduces FINDINGS section 16 (0/12 -> 7/12 at R=2, 0/12 -> 12/12
at R=4). The R=4 number is weaker here and the reason is a harness difference,
not a contradiction: the address-wall track trained on 24 examples, this suite
on 12. So the R=4 relax arm below is **under-trained relative to the recorded
result**, which makes the measured cost of choosing relaxation on that case an
over-estimate. Stated rather than corrected, because the selector's decision on
that case does not depend on it.

---

## 4. Every threshold and where its number comes from

| threshold | value | provenance |
|---|---|---|
| `DEAD_GRADIENT` | 1e-12 | **Measured + derived.** Measured: `eq` at tau=1 is *exactly* 0.0 for \|a-b\| >= 11, so on the case that produced the wrong headline the reading is 0.0 and any threshold in [0, 1e-12] gives the same verdict. Derived: Adam's step is `lr * g / (sqrt(v) + eps)` with eps=1e-8, so at the shipped lr=.05 over 1000 steps a gradient of 1e-12 buys a total logit displacement of 5e-3, which cannot reorder a softmax. It is a bound, not a fit. |
| `ENUMERATION_SECONDS` | 30 s | **Measured.** FINDINGS section 7 records one gradient run on the flagship scaffolds at 10-36 s. 30 s is inside that band, so enumeration is chosen only where it is projected to beat the method it replaces. It fixes *how much enumeration may cost*, never how large a space may be — the space-to-seconds conversion is measured per scaffold. |
| `ENVIRONMENT_BUDGET` | 704 env steps | **Measured.** FINDINGS section 7: the gradient path is 5/5 at its own budget of 704 environment steps where random search over the same 256 candidates is 0/20, and a full sweep needs 16,384. The selector reproduces the 16,384 exactly (256 x 16 episodes x horizon 4). |
| `ENUMERATION_PROBES` | 12 | **A guess**, and labelled as one. Cheap, and nothing downstream is sensitive to the count — what *is* sensitive is the estimator, see section 7. |
| `HYBRID_FULL_FITS` | 8 | **A guess**, and labelled as one. The hybrid screens then re-fits in rank order and stops at the first conforming structure, so the true number is data-dependent. 8 is chosen to over-estimate: the largest observed was 1, so the error is on the side that prices the hybrid *out*. |
| carrier temperature | `2^bits` | **Measured**, FINDINGS section 16: at the failing distance it lifts the surrogate from 0.0 to 7.7e-02, and took a benchmark from 0/12 to 12/12. Read off the declared type, so `bool` returns 1.0 and boolean `eq` is unchanged. |
| liveness initializations | `(0., 0.5)` | **Measured necessity.** At the shipped zero initialization a uniform mixture over the sixteen truth tables is the constant 0.5 on every row, so a *balanced* target cancels the choice gradient exactly — the first version of this check called `noisy_2flip`'s three nodes dead for that reason. It is a property of the initialization, not of the relaxation; `examples/joint.py` carries a residual initialization to break the same cancellation. Liveness is therefore read at the shipped initialization and at a perturbed one, and a candidate is dead only where it is dead at both. Regression test: `test_liveness_does_not_call_a_zero_init_cancellation_dead`. |
| `screen_steps` / `constant_steps` | 25 / 200 | 200 is `synthesis.fit`'s shipped `polish` budget, measured in FINDINGS section 10 at 10/10 conformance. 25 is **a guess** for the screen, mitigated by the screen being an *ordering only* — nothing is discarded, so the worst case is the unscreened search. |

---

## 5. What the selector chooses, and what was right

Eleven cases. `right` is the answer FINDINGS records for that scaffold, cited in
`common.py` case by case, not assigned after seeing the decision.

| case | family | space | chose | right | agrees | why |
|---|---|---|---|---|---|---|
| `mixed` | small-exact | 96 | **enumerate** | enumerate | yes | exhaustible in 0.0036 s, certifies uniqueness |
| `mixed_constant` | continuous | 192 | **hybrid** | hybrid | yes | discrete part exhaustible in 0.0029 s, one live constant `k`, hybrid priced at 2.05 s |
| `joint_offline` | small-exact | 256 | **enumerate** | enumerate | yes | exhaustible in 0.0104 s; the five declared constants receive no gradient from this supervision |
| `joint_environment` | environment | 256 | **relax** | relax | yes | a full sweep costs **16,384** environment steps against a 704-step budget |
| `rung3_R2` | dead-surrogate | 576 | **enumerate** | enumerate | yes | exhaustible in 0.0165 s |
| `rung3_R4` | dead-surrogate | 9,216 | **enumerate** | enumerate | yes | exhaustible in 0.611 s |
| `rung3_bytes` | large-space | 9,437,184 | **relax** | relax | yes | enumeration projects to 346 s; surrogate needs carrier scaling |
| `gradient_boundary` | dead-surrogate | 12 | **enumerate** | enumerate | yes | exhaustible in 1.9e-4 s |
| `boundary_wide` | dead-surrogate | 480,000 | **enumerate** | enumerate | yes | **neither backend is sound**: enumeration projects to 60.7 s and the relaxation is dead at `byte, tail` |
| `noisy_2flip` | small-exact | 4,096 | **enumerate** | enumerate | yes | exhaustible in 0.081 s |
| `wide_logic` | large-space | 3.39e9 | **relax** | relax | yes | enumeration projects to 2.76e4 s |

**11/11 on the decision.** Through `synthesis.fit(mode="auto")` end to end it is
**10/11**: `noisy_2flip`'s enumeration exhausts the space, certifies that no
exact program exists, and the auto path falls through to the relaxation, which
then reports `relax` as the executed backend. See section 8.

### Outcomes, both backends run wherever both were affordable

3 seeds with explicit initialization noise 0.5 for every gradient arm — a
"3-seed" figure without it would be one outcome printed three times.

| case | enumerate | hybrid | relax | relax, shipped surrogate |
|---|---|---|---|---|
| `mixed` | 1/1 in 0.006 s, 1 conforming, certificate | -- | 3/3 in 3.17 s | -- |
| `mixed_constant` | **0/1** in 0.004 s, 0 conforming | **1/1** in 2.91 s | **0/3** in 5.44 s | -- |
| `joint_offline` | 1/1 in 0.046 s, 1 conforming, certificate | 1/1 in 6.19 s | 3/3 in 7.58 s | -- |
| `joint_environment` | 1/1 in 0.021 s, certificate | 1/1 in 14.5 s | 3/3 in 8.21 s | -- |
| `rung3_R2` | 1/1 in 0.027 s, **4 conforming**, no certificate | -- | 3/3 in 3.53 s | **0/3** in 2.68 s |
| `rung3_R4` | 1/1 in 0.88 s, **20 conforming**, no certificate | -- | 1/3 in 9.15 s | **0/3** in 9.83 s |
| `rung3_bytes` | not run, projected 346 s | -- | 0/3 in 84.8 s | 0/3 in 127 s |
| `gradient_boundary` | 1/1 in 0.003 s, certificate | -- | **1/3** in 0.54 s | -- |
| `boundary_wide` | 1/1 in **88.4 s**, 4 conforming | -- | 0/3 in 43.9 s | -- |
| `noisy_2flip` | **0/1** in 0.049 s, 0 conforming, exhausted | -- | 2/3 in 2.67 s | -- |
| `wide_logic` | not run, projected 2.76e4 s | -- | 0/3 in 334 s | -- |

Notes on rows that say more than the decision:

- **`mixed_constant` separates the three backends cleanly.** Enumeration finds
  nothing (the declared constant is wrong, and `enumerate_fit` says so rather
  than searching a subspace). The relaxation is 0/3 — the failure FINDINGS
  section 10 records at 3.9e-2. The hybrid is 1/1 and recovers `k = 1.70001078`
  against a true 1.7, held-out exact error **6.1e-06**.
- **`gradient_boundary` relaxation is at chance, not at zero.** 1/3, and the
  space has 4 addresses: a random initialization lands on the right one about a
  quarter of the time. Behind a `gradient="none"` boundary the gradient path is
  a random draw, which is a sharper statement than "it fails".
- **`boundary_wide` shows the conservative side of the budget.** The selector
  reports `feasible=False` — enumeration projected at 60.7 s against a 30 s
  budget, relaxation dead — and enumeration then solved it in 88.4 s while the
  relaxation was 0/3 in 44 s. Being told "neither is sound, and enumeration is
  the less unsound one" was correct; the budget is a budget, not a capability
  claim.
- **Non-uniqueness is confirmed as the norm** (FINDINGS section 14). `rung3_R2`
  has 4 conforming programs of 576 and `rung3_R4` has 20 of 9,216, so neither
  returns a uniqueness certificate. Both returned programs were nonetheless
  exact on the third, never-scored split, so on these scaffolds the tie-break
  was not the problem it was at 32,000 programs.

---

## 6. The asymmetry, quantified

The task's framing is the right one: a selector right 8/10 but catastrophic on
the 2 is worse than one right 7/10 and cheap when wrong. So here is what each
mistake would have cost, per case, in the currency that case is scarce in.

| case | chose | cost of the discrete backend | cost of the relaxation | relax / discrete |
|---|---|---|---|---|
| `mixed` | enumerate | 0.006 s, solved | 1.06 s/run, 3/3 exact | **176x** |
| `mixed_constant` | hybrid | 2.91 s, solved | 1.81 s/run, **0/3** exact | 0.62x |
| `joint_offline` | enumerate | 0.046 s, solved | 2.53 s/run, 3/3 exact | **55x** |
| `joint_environment` | relax | 0.021 s CPU but **16,384 env steps** | 2.74 s/run, **704 env steps**, 3/3 | see below |
| `rung3_R2` | enumerate | 0.027 s, solved | 1.18 s/run, 3/3 exact | **43x** |
| `rung3_R4` | enumerate | 0.88 s, solved | 3.05 s/run, 1/3 exact | 3.5x |
| `rung3_bytes` | relax | 346 s (projected) | 28.3 s/run, 0/3 exact | 0.08x |
| `gradient_boundary` | enumerate | 0.003 s, solved | 0.18 s/run, 1/3 exact | **60x** |
| `boundary_wide` | enumerate | 88.4 s, solved | 14.6 s/run, 0/3 exact | 0.17x |
| `noisy_2flip` | enumerate | 0.049 s, **not solved** (certificate) | 0.89 s/run, 2/3 exact | 18x |
| `wide_logic` | relax | 2.76e4 s (projected) | 111 s/run, 0/3 exact | 0.004x |

Read down the two error directions:

**Wrong toward enumeration** (choosing the sweep where the relaxation would have
done) costs at most `projected_enumeration_seconds`, which the selector measured
before deciding, and it is bounded by `ENUMERATION_SECONDS = 30 s` by
construction. The worst realised over-run was `boundary_wide` at 88 s against a
60.7 s projection — 1.5x, on a case the selector had already flagged
`feasible=False`. And the sweep never returns a program it did not check.

**Wrong toward relaxation** (choosing descent where the sweep would have done)
costs a run that on this suite failed outright on 5 of 11 cases, returns no
certificate on any, and — the part that matters — **cannot be detected from its
own output**: FINDINGS section 4 records 17/17 failures reaching zero soft BCE
with a wrong argmax, and section 12 records a relaxed loss of 0.002 in a space
certified to contain no solution. A wrong enumeration is loud. A wrong
relaxation is silent.

That is why the rule tests affordability before it tests liveness, and it is why
the two guessed thresholds (`ENUMERATION_PROBES`, `HYBRID_FULL_FITS`) are both
set to err toward the bounded side.

The environment row is the one place the currency changes and the ordering
flips: the sweep is 0.021 s of CPU but **16,384 environment steps**, 23x the
704-step budget at which FINDINGS section 7 measured the gradient path 5/5
against random search's 0/20. The selector computes 16,384 exactly from
`space_size x rollout_cost` and reports it in its reason string.

---

## 7. The selector's real weak point: the projection

`projected_enumeration_seconds` is the noisiest input and the only one that can
flip a decision. Projected against actually-measured sweep time, this run:

| case | projected | actual | projected/actual |
|---|---|---|---|
| `mixed` | 0.0036 s | 0.006 s | 0.60 |
| `mixed_constant` | 0.0029 s | 0.004 s | 0.73 |
| `joint_offline` | 0.0104 s | 0.046 s | 0.23 |
| `rung3_R2` | 0.0165 s | 0.027 s | 0.61 |
| `rung3_R4` | 0.611 s | 0.88 s | 0.69 |
| `gradient_boundary` | 1.9e-4 s | 0.003 s | 0.06 |
| `boundary_wide` | 60.7 s | 88.4 s | 0.69 |
| `noisy_2flip` | 0.081 s | 0.049 s | 1.63 |

So it under-estimates by about 1.4x typically, and that is the benign direction
(it makes enumeration look cheaper than it is, biasing toward the bounded error).
But an **earlier run of this same suite, on a host at load average 77, projected
55.7 s for the `rung3_R4` sweep that then ran in 8.8 s** — a 6.3x over-estimate,
which pushed the case across the 30 s budget and made the selector choose
`relax`, where it got 1/3 against enumeration's 1/1. That is the selector's
worst observed failure and it was pure measurement noise.

**Fixed, and the fix is in `enumeration_cost`:** discard the first probe as
warm-up (it pays program validation and interpreter warm-up the other 9,215 do
not) and project from the **median** rather than the mean, so one descheduled
probe cannot drag the estimate. Both the mean-based and median-based projections
are reported in the result dict so the difference stays visible. With that
change the same case projects 0.611 s and chooses `enumerate`.

This does not make the estimator sound, it makes it robust. **The honest
statement is that any decision whose projection lands within about an order of
magnitude of the budget is not reliable on a shared host**, and the remedy would
be to stop projecting and start enumerating with a wall-clock cap, falling back
when the cap is hit. That is a strictly better design — it pays at most the
budget and gets the certificate whenever it is cheap — and it is not implemented
here.

Decision overhead itself, for completeness: 0.13 s to 14.5 s, dominated by the
liveness pass, which is O(candidates). It is 4.9 s on the 3.4e9-program
`wide_logic` scaffold and 14.5 s on `boundary_wide` (800 address candidates).
Where the answer is "relax", that overhead is pure loss; subsampling candidates
in the liveness pass would remove most of it and is not implemented.

---

## 8. What it gets wrong

**One disagreement out of eleven, and it is a missing capability, not a
threshold.** `noisy_2flip` is the noisy-partial-credit family. FINDINGS section 7
measures minimum-Hamming enumeration recovering the uncorrupted function 4/4,
3/4 and 2/4 against the gradient path's 0.50, 0.12 and 0.12, roughly 1000x
faster — and notes in the same breath that a pure decision encoding *correctly*
returns UNSAT there. `tcn/search.py` is an exact-conformance search with no
minimum-Hamming objective, so **that measured advantage is not reachable through
`enumerate_fit` at all**. The selector picks enumeration on the properties (4,096
programs, exhaustible in 0.08 s), enumeration exhausts and finds nothing, and it
is right that nothing exists.

The auto path handles it the way the asymmetry argument says it should: an
exhausted sweep with no solution is a **completeness certificate**, so `fit`
records it and falls through to the relaxation. Total cost of the "wrong" choice:
0.049 s on top of a 0.89 s gradient run — **5.5%**. That is the strongest single
piece of evidence for ordering the rule the way it is ordered.

The fix is to give `enumerate_fit` a minimum-Hamming ranking alongside
`order`/`description`/`cost`. It is not implemented here.

**Three further things it gets wrong or cannot do:**

1. **It cannot see environment coupling.** `rollout_cost` is declared. A caller
   who forgets it gets `enumerate` on a scaffold whose sweep costs 16,384
   rollouts. There is no way to derive it from a `Program`, and pretending
   otherwise would be worse than requiring the declaration.
2. **The projection is unreliable near the budget** (section 7). Measured 6.3x
   error under load, now mitigated but not solved.
3. **The hybrid's projection under-estimates.** Priced at 2.05 s on
   `mixed_constant`, actual 2.91 s; on an earlier, unscreened implementation it
   was priced at 2.1 s and ran 10.3 s. The screen fixed the level, not the
   estimator. `HYBRID_FULL_FITS = 8` is a guess set to over-estimate to
   compensate.

**Two selector bugs the suite caught and that are now fixed** — recorded because
both were exactly the class of error FINDINGS keeps finding:

- **A trainable constant off every supervised path is not a continuous
  sub-problem.** The first rule chose `hybrid` on `joint_offline` because the
  program declares five trainable constants. All five sit on the policy readout;
  none receives any gradient from the probe signals. The hybrid took 6.2 s where
  enumeration took 0.046 s — **135x** — for the same answer. The rule now
  measures `d probe_loss / d constant` and only counts constants this
  supervision can actually move.
- **The hybrid's price is not enumeration's price.** Enumeration's projection
  prices one exact evaluation per program; the hybrid pays a gradient fit. Left
  unpriced, the rule justified choosing the hybrid on a 30 s budget that the
  hybrid then blew through (29 s measured on a 0.017 s-projected space).
  `hybrid_cost` now measures it, and `hybrid_fit` screens all structures cheaply
  and re-fits in rank order, which brought `mixed_constant` from 29 s to 2.9 s.

---

## 9. Wiring, and what it does to the fixtures

`synthesis.fit(mode=...)` takes `'relax'` (default, unchanged), `'auto'`,
`'enumerate'` and `'hybrid'`. Whatever runs, the report has the same keys plus a
`selection` block recording the decision and the measurements behind it, and the
returned model exports a frozen program either way, so callers do not branch.
`tcn synthesize --mode` exposes it; `stage_runner` passes a `mode` from stage
configuration. Both default to `relax`.

`fit(validation=...)` is used only by the discrete backends, and only because
FINDINGS section 14 measured `enumerate_fit`'s lexicographic pick wrong on fresh
episodes where 2,464 of 32,000 programs conform; scoring on training plus a
validation split is the fix that was measured to work. Regression test:
`test_validation_split_is_scored_by_the_discrete_backend` takes a 4-way
under-determined truth table to a unique solution.

Every case through the public entry point:

| fixture | space | chose | right | agrees | exact error | seconds |
|---|---|---|---|---|---|---|
| `mixed` | 96 | enumerate | enumerate | yes | 0.0 | 0.023 |
| `mixed_constant` | 192 | hybrid | hybrid | yes | 6.1e-06 | 2.25 |
| `joint_offline` | 256 | enumerate | enumerate | yes | 0.0 | 0.070 |
| `joint_environment` | 256 | relax | relax | yes | 0.0 | 1.07 |
| `rung3_R2` | 576 | enumerate | enumerate | yes | 0.0 | 0.058 |
| `rung3_R4` | 9,216 | enumerate | enumerate | yes | 0.0 | 1.04 |
| `rung3_bytes` | 9,437,184 | relax | relax | yes | 1.0 | 15.5 |
| `gradient_boundary` | 12 | enumerate | enumerate | yes | 0.0 | 0.015 |
| `boundary_wide` | 480,000 | enumerate | enumerate | yes | 0.0 | 83.4 |
| `noisy_2flip` | 4,096 | **relax** (after the no-solution certificate) | enumerate | **no** | 1.0 | 0.60 |
| `wide_logic` | 3.39e9 | relax | relax | yes | 1.0 | 16.3 |

The three exact errors of 1.0 are the cases where the selector correctly chose
the relaxation and the relaxation failed at `fit`'s default 200-step budget.
Note the measurement warning applies with full force here: `fit` adds **no**
initialization noise, so each of these is a single deterministic outcome, not a
rate. The suite's noised 3-seed arms in section 5 are the rate.

`examples/mixed.py` through the untouched default path: relaxed loss
1.0066e-06, exact error 0.0, `selection` is `None`.

---

## 10. Reproducing

```
PYTHONPATH=. .venv/bin/python research/search-selection/liveness.py   #  ~2 min
PYTHONPATH=. .venv/bin/python research/search-selection/suite.py      # ~15 min
PYTHONPATH=. .venv/bin/python research/search-selection/fixtures.py   #  ~2 min
PYTHONPATH=. .venv/bin/python research/search-selection/report.py     # renders every table above
```

Raw JSON in `out/`. Every table in this file is generated by `report.py` from
that JSON; nothing is transcribed by hand.

**Host caveat.** These runs shared a machine with several other agents at load
averages between 70 and 80. Absolute wall-clock numbers are therefore upper
bounds and vary by up to 7x between runs of the same measurement — `rung3_bytes`
projected 1.28e4 s on one run and 346 s on the next. Ratios within a run are the
meaningful quantity, and section 7 is about exactly this.

## 11. Tests

`tests/test_selection.py`, 18 tests: the `eq` threshold at 11, carrier
temperature read off the declared type, carrier scaling reviving the surrogate,
per-example liveness, the `gradient="none"` boundary, the zero-init cancellation
regression, `surrogate_scale` defaulting to identity, `scale_surrogates` leaving
the choice distribution untouched, the shipped fixture unchanged end to end,
each branch of the rule, the projection arithmetic, the auto/forced/unknown
modes, and the validation-split fix.

Full suite: **197 passed** (179 before this branch), with the gitignored
`generators/computer/engine/node_modules` symlinked in for the four `computer`
tests and the symlink removed before committing.
