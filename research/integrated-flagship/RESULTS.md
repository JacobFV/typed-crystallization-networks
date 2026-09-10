# The integrated flagship — does the inherited library make an integrated task cheaper to acquire?

`PREREGISTRATION.md` committed at `22e17f8`, before any arm. `generators/` and
`tcn/` have no diff on this branch. Every figure-bearing table sits between
`<!-- BEGIN:… -->` / `<!-- END:… -->` markers and is written by `report.py` from
files under `out/`; `verify.py` re-renders every block, re-derives each headline
claim from raw JSON without importing the run scripts, and fails on any number in
the prose that does not trace to a block.

**Scope of the evidence, stated once and repeated where it matters:** one seed
set (seed 0), one palette (`palette_levels 2`), one relation vocabulary
(right_of, left_of, above), one colour vocabulary (red, green, blue, yellow), 12
training and 36 held-out episodes, resolution 16. Single-configuration evidence
throughout, with a second configuration (`gap 1`) where stated.

---

## VERDICT

**C1 FAILS, as pre-registered.** The inherited library did **not** make the integrated task
materially cheaper to acquire on the metric this track committed to before any arm ran —
expected programs (and episodes) to the first program that conforms on the training episodes.
The schema-plus-learned-prior arm N″ costs far *more* than the flat substrate N, not less
(§1, §2). The success criterion is not met. This is the headline, and nothing below changes it.

**The pre-registered metric was mis-specified — recorded as a lesson, not as a rescue.**
"Programs to the first training-conforming program" counts a spurious fit as a solution. On
12 training episodes the flat and schema-only spaces are dense with programs that fit training
and fail held-out, and those are what N and N′ reach first: their sampled first hits generalize
essentially never (§2). Their low costs are therefore the cost of reaching a *wrong* answer.
The criterion should have been stated on held-out conformance, or with enough training
episodes to make spurious fits rare. That is a defect of this pre-registration, and it is not
repaired after the fact: every verdict in §1 is resolved on the metric as written.

**Exploratory and post-hoc (not pre-registered), labelled wherever it appears:** N″'s first hits
generalize, and N's and N′'s do not. The generalization rates carry Wilson intervals, and §9
bounds the cost to a first *generalizing* program as an exploratory measurement with a proposed
follow-up pre-registration. No verdict rests on it.

**Leakage — checked first, and clean.** Every row the specialization prior was fitted on comes
from the earlier visual, language and computer searches. None comes from the integrated task:

<!-- BEGIN:leakage -->
| prior kind | rows | survivors | from which earlier search |
|---|---|---|---|
| ADDR | 233 | 23 | s19_stage_a (41), s23_policy (56), s23_transform (136) |
| LIT | 424 | 22 | s19_stage_a (256), s23_policy (168) |
| TRUTH | 48 | 5 | s33_s0 (32), s33_s1 (16) |
| STEP | 20 | 6 | s33_s1 (10), s33_s2 (10) |
| GROUND | 0 | 0 | none |

No row comes from the integrated task, its training episodes or its held-out episodes; `verify.py` checks this and that the prior refitted from `sources.json` alone reproduces every stored estimator cell and N″'s tiers from the training episodes' unlabelled text. The integrated task contributes only unlabelled features of its training observations (V: does the addressed byte vary; occurs: does the literal appear there). The held-out episodes enter nothing but the generalization score.
<!-- END:leakage -->

**C2 is met only vacuously.** The pre-registered distractor could not succeed: an exact
48-episode count finds no program in its space that conforms on training and held-out together,
and none of its address expressions reaches the colour word (§9.1). Its failure to help is true by
construction — a straw man — so the "matched distractor does not help" half of the owner's
criterion is carried instead by the post-hoc controls R, U_feat, U_steep, U_Vonly and the per-kind
knockouts (§9.3), not by D′/D″.

**What carries the post-hoc generalization, at one width and one task family.** Knocking out one
learned estimator at a time from N″ leaves the first solution generalizing except when the STEP
estimator — fitted on the 20 step-offset rows of the §33 visual parse — is removed; STEP fitted
alone, with every other estimator uniform, is sufficient. But a one-line human rule
("prefer single-pixel and single-row offsets") does the same, so **those 20 learned rows add
nothing measurable beyond a prior a human would write**. KO_ADDR rules out the §19/§23 *address*
rows only; the literal knockout (identical to OCC_1) rules out their literal rows as well.
Hand features on the address and literal slots — gentle or at N″'s own steepness — never produce
generalization without a step preference (§9.3). **"Generalizes" and "cheaper" are separate
claims**: the step preference makes the first hit correct, not cheap; the cost is set by the
address and literal preferences, and §9.4 gives programs to the first *generalizing* program per
arm, post-hoc. Scope: one seed set, one palette, one relation vocabulary, `gap 0`; the STEP
estimator rests on 20 rows; `gap 1` is §2's second table.

**The post-hoc gap-0 generalization finding does not replicate at the pre-registered second
configuration.** At `gap 1`, N″'s sampled first solutions generalize none of the time (§2's
second table): the result that its first hit generalizes is a `gap 0` result only.

**F6 and the `gap 1` prediction, resolved strictly as pre-registered.** F6 is defined as "0
conforming, exhausted"; the `gap 1` schema space has training-conforming programs, so **F6 is not
triggered as pre-registered**. The prediction that "N′ and N″ have no solution there" is
**falsified as stated**: they do have training-conforming solutions. What the exact 48-episode
count at `gap 1` shows is reported separately in §9.1, post-hoc, and is not F6 whichever way it
comes out.

**Even on the post-hoc metric, the learned prior does not earn its name.** Counting programs to
the first *generalizing* program exactly (§9.4), and restricting to arms whose settings were fixed
before the target was seen, the cheapest route is not N″: the unfitted hand-feature controls
U_feat and U_steep reach a generalizing program sooner. N″ beats the flat and schema-only
baselines on that metric, but a hand prior over the same schema beats N″. The inherited library's
measurable contribution here is the schema's pools plus a step preference that a one-line human
rule reproduces; nothing fitted from the earlier domains' survival rates is shown to be needed.

---

## 1. The pre-registered criteria, resolved

Primary configuration (`gap 0`):

<!-- BEGIN:criteria_gap0 -->
- **C1 (material saving, ≥10× in programs and episodes):** N″/N = 10,925.41 → **FAIL** (episodes are programs × 12, so the same ratio).
- **C2 (the matched distractor does not reach 10×):** D''_0: cost/N = 1.30×10^6; D''_1: cost/N = 3.24×10^7; D''_2: cost/N = 1.30×10^6; D''_3: cost/N = 3.24×10^7; D''_4: cost/N = 2.14×10^14; D': cost/N = 1.96×10^6 → **PASS**.
- **C3 (a solution, not a fit):** the worst of 400 sampled N″ first solutions scores 1.0000 held out (evaluator; live check in the instrumentation table) against the best baseline 0.6944 → **PASS**.
- **N′ vs N:** N′/N = 1.88 → schema is organisational reuse (within 10×).
- **N″ vs N′ (does the prior earn its name, ≤ N′/10?):** N″/N′ = 5,806.89 → **no**.
<!-- END:criteria_gap0 -->

Second configuration (`gap 1`):

<!-- BEGIN:criteria_gap1 -->

<!-- END:criteria_gap1 -->

## 2. Every arm, exact

<!-- BEGIN:arms_gap0 -->
| arm | full space S | conformers K | certificate | first solvable tier | expected programs to first solution | expected episodes | saving vs N | first solution generalizes (of sampled) | held-out accuracy of first solution |
|---|---|---|---|---|---|---|---|---|---|
| N — flat, uniform | 5.03×10^29 | 5.29×10^21 | `complete` | 0 | 9.50×10^7 | 1.14×10^9 | 1× | 0/400 (95% 0.000–0.010) | 0.6444 |
| N′ — schema, uniform | 2.08×10^23 | 1.16×10^15 | `complete` | 0 | 1.79×10^8 | 2.14×10^9 | 0.53× | 0/400 (95% 0.000–0.010) | 0.6138 |
| N″ — schema + learned prior | 2.08×10^23 | 1.16×10^15 | `complete` | 2 | 1.04×10^12 | 1.25×10^13 | 9.15×10^-5× | 400/400 (95% 0.990–1.000) | 1.0000 |
| P — flat + learned prior | 5.03×10^29 | 5.29×10^21 | `complete` | 2 | 4.27×10^17 | 5.12×10^18 | 2.22×10^-10× | 0/400 (95% 0.000–0.010) | 0.6431 |
| D′ — distractor schemas, uniform | 2.08×10^23 | 1.12×10^9 | `complete` | 0 | 1.86×10^14 | 2.24×10^15 | 5.10×10^-7× | 0/400 (95% 0.000–0.010) | 0.3308 |
| N″ without observation features | 2.08×10^23 | 1.16×10^15 | `complete` | 3 | 5.04×10^20 | 6.04×10^21 | 1.89×10^-13× | 1/400 (95% 0.000–0.014) | 0.4515 |
| D″ — distractor schemas + permuted prior (seed 0) | 2.08×10^23 | 1.12×10^9 | `complete` | 1 | 1.23×10^14 | 1.48×10^15 | 7.72×10^-7× | 0/400 (95% 0.000–0.010) | 0.3406 |
| D″ — distractor schemas + permuted prior (seed 1) | 2.08×10^23 | 1.12×10^9 | `complete` | 1 | 3.08×10^15 | 3.69×10^16 | 3.09×10^-8× | 0/400 (95% 0.000–0.010) | 0.2690 |
| D″ — distractor schemas + permuted prior (seed 2) | 2.08×10^23 | 1.12×10^9 | `complete` | 1 | 1.23×10^14 | 1.48×10^15 | 7.72×10^-7× | 0/400 (95% 0.000–0.010) | 0.3393 |
| D″ — distractor schemas + permuted prior (seed 3) | 2.08×10^23 | 1.12×10^9 | `complete` | 1 | 3.08×10^15 | 3.69×10^16 | 3.09×10^-8× | 0/400 (95% 0.000–0.010) | 0.2719 |
| D″ — distractor schemas + permuted prior (seed 4) | 2.08×10^23 | 1.12×10^9 | `complete` | 4 | 2.03×10^22 | 2.44×10^23 | 4.67×10^-15× | 0/400 (95% 0.000–0.010) | 0.3328 |
| R — right schemas + permuted prior (seed 0) | 2.08×10^23 | 1.16×10^15 | `complete` | 3 | 1.73×10^21 | 2.08×10^22 | 5.48×10^-14× | 0/400 (95% 0.000–0.010) | 0.4562 |
| R — right schemas + permuted prior (seed 1) | 2.08×10^23 | 1.16×10^15 | `complete` | 2 | 5.47×10^13 | 6.56×10^14 | 1.74×10^-6× | 400/400 (95% 0.990–1.000) | 1.0000 |
| R — right schemas + permuted prior (seed 2) | 2.08×10^23 | 1.16×10^15 | `complete` | 3 | 3.53×10^19 | 4.24×10^20 | 2.69×10^-12× | 0/400 (95% 0.000–0.010) | 0.6162 |
| R — right schemas + permuted prior (seed 3) | 2.08×10^23 | 1.16×10^15 | `complete` | 3 | 3.64×10^20 | 4.37×10^21 | 2.61×10^-13× | 2/400 (95% 0.001–0.018) | 0.4510 |
| R — right schemas + permuted prior (seed 4) | 2.08×10^23 | 1.16×10^15 | `complete` | 4 | 1.30×10^22 | 1.56×10^23 | 7.30×10^-15× | 0/400 (95% 0.000–0.010) | 0.6172 |
| U — N″'s features, UNFITTED gentle 2× rule (post-hoc control) | 2.08×10^23 | 1.16×10^15 | `complete` | 0 | 2.04×10^6 | 2.45×10^7 | 46.62× | 0/400 (95% 0.000–0.010) | 0.5999 |
| U_steep — hand rules at N″'s steepness, nothing fitted (post-hoc control) | 2.08×10^23 | 1.16×10^15 | `complete` | 0 | 2.04×10^6 | 2.45×10^7 | 46.62× | 0/400 (95% 0.000–0.010) | 0.5999 |
| U_Vonly — one hand rule, V 10:1, all else uniform, nothing fitted (post-hoc control) | 2.08×10^23 | 1.16×10^15 | `complete` | 0 | 4.41×10^7 | 5.29×10^8 | 2.15× | 0/400 (95% 0.000–0.010) | 0.6138 |
| KO_ADDR — N″ with ADDR (§19/§23 rows) knocked out to hand V 10:1 (post-hoc) | 2.08×10^23 | 1.16×10^15 | `complete` | 1 | 3.05×10^13 | 3.66×10^14 | 3.11×10^-6× | 400/400 (95% 0.990–1.000) | 1.0000 |
| KO_TRUTH — N″ with TRUTH (§33 rows) knocked out to uniform (post-hoc) | 2.08×10^23 | 1.16×10^15 | `complete` | 2 | 3.77×10^13 | 4.52×10^14 | 2.52×10^-6× | 400/400 (95% 0.990–1.000) | 1.0000 |
| KO_STEP — N″ with STEP (§33 rows) knocked out to uniform (post-hoc) | 2.08×10^23 | 1.16×10^15 | `complete` | 2 | 1.61×10^13 | 1.93×10^14 | 5.91×10^-6× | 8/400 (95% 0.010–0.039) | 0.4636 |
| STEP_only — only the fitted STEP estimator (20 §33 rows), all else uniform (post-hoc) | 2.08×10^23 | 1.16×10^15 | `complete` | 1 | 1.66×10^21 | 2.00×10^22 | 5.71×10^-14× | 400/400 (95% 0.990–1.000) | 1.0000 |
| STEP_hand — hand rule: pixel or row offsets 1, else 0.1; all else uniform (post-hoc) | 2.08×10^23 | 1.16×10^15 | `complete` | 0 | 3.30×10^16 | 3.96×10^17 | 2.88×10^-9× | 400/400 (95% 0.990–1.000) | 1.0000 |
| N″, occurs ratio pinned at 1 — sensitivity, set on the target task, not inherited | 2.08×10^23 | 1.16×10^15 | `complete` | 2 | 5.51×10^17 | 6.62×10^18 | 1.72×10^-10× | 400/400 (95% 0.990–1.000) | 1.0000 |
| N″, occurs ratio pinned at 3 — sensitivity, set on the target task, not inherited | 2.08×10^23 | 1.16×10^15 | `complete` | 2 | 4.50×10^15 | 5.40×10^16 | 2.11×10^-8× | 400/400 (95% 0.990–1.000) | 1.0000 |
| N″, occurs ratio pinned at 10 — sensitivity, set on the target task, not inherited | 2.08×10^23 | 1.16×10^15 | `complete` | 2 | 1.04×10^12 | 1.25×10^13 | 9.15×10^-5× | 400/400 (95% 0.990–1.000) | 1.0000 |
| N″, occurs ratio pinned at 100 — sensitivity, set on the target task, not inherited | 2.08×10^23 | 1.16×10^15 | `complete` | 2 | 1.04×10^12 | 1.25×10^13 | 9.15×10^-5× | 400/400 (95% 0.990–1.000) | 1.0000 |
| N″, occurs ratio pinned at 3500 — sensitivity, set on the target task, not inherited | 2.08×10^23 | 1.16×10^15 | `complete` | 2 | 1.04×10^12 | 1.25×10^13 | 9.15×10^-5× | 400/400 (95% 0.990–1.000) | 1.0000 |
| N″, V ratio pinned at 1 — sensitivity, set on the target task, not inherited | 2.08×10^23 | 1.16×10^15 | `complete` | 2 | 1.69×10^12 | 2.03×10^13 | 5.62×10^-5× | 400/400 (95% 0.990–1.000) | 1.0000 |
| N″, V ratio pinned at 3 — sensitivity, set on the target task, not inherited | 2.08×10^23 | 1.16×10^15 | `complete` | 1 | 1.96×10^10 | 2.35×10^11 | 4.84×10^-3× | 400/400 (95% 0.990–1.000) | 1.0000 |
| N″, V ratio pinned at 10 — sensitivity, set on the target task, not inherited | 2.08×10^23 | 1.16×10^15 | `complete` | 1 | 1.96×10^10 | 2.35×10^11 | 4.84×10^-3× | 400/400 (95% 0.990–1.000) | 1.0000 |
| N″, V ratio pinned at 100 — sensitivity, set on the target task, not inherited | 2.08×10^23 | 1.16×10^15 | `complete` | 1 | 1.96×10^10 | 2.35×10^11 | 4.84×10^-3× | 400/400 (95% 0.990–1.000) | 1.0000 |
<!-- END:arms_gap0 -->

`gap 1`:

<!-- BEGIN:arms_gap1 -->
| arm | full space S | conformers K | certificate | first solvable tier | expected programs to first solution | expected episodes | saving vs N | first solution generalizes (of sampled) | held-out accuracy of first solution |
|---|---|---|---|---|---|---|---|---|---|
<!-- END:arms_gap1 -->

## 3. The saving against the space size (F3)

<!-- BEGIN:f3_gap0 -->
| arm | space ratio S_N / S_arm | solution retention K_arm / K_N | saving N / arm (expected programs) | saving vs space ratio |
|---|---|---|---|---|
| N′ — schema, uniform | 2.42×10^6 | 2.20×10^-7 | 0.53 | 2.20×10^-7 |
| N″ — schema + learned prior | 2.42×10^6 | 2.20×10^-7 | 9.15×10^-5 | 3.79×10^-11 |
| P — flat + learned prior | 1 | 1 | 2.22×10^-10 | 2.22×10^-10 |
<!-- END:f3_gap0 -->

<!-- BEGIN:f3_gap1 -->
(N not run)
<!-- END:f3_gap1 -->

## 4. The learned prior's tiers

<!-- BEGIN:tiers_npp_gap0 -->
| tier | S_t | K_t | shell S | shell K | ADDR kept | MATCH kept | STEP kept | absent letters kept |
|---|---|---|---|---|---|---|---|---|
| 0 | 2.62×10^5 | 0 | 2.62×10^5 | 0 | 1 | 1 | 4 | no |
| 1 | 1.03×10^12 | 0 | 1.03×10^12 | 0 | 56 | 7 | 8 | no |
| 2 | 6.56×10^14 | 77,280 | 6.55×10^14 | 77,280 | 934 | 16 | 8 | no |
| 3 | 1.02×10^16 | 1.77×10^8 | 9.59×10^15 | 1.77×10^8 | 935 | 16 | 20 | no |
| 4 | 1.02×10^16 | 1.77×10^8 | 0 | 0 | 935 | 16 | 20 | no |
| 5 | 2.56×10^16 | 2.57×10^8 | 1.54×10^16 | 7.97×10^7 | 935 | 40 | 20 | no |
| 6 | 7.18×10^16 | 3.51×10^8 | 4.62×10^16 | 9.38×10^7 | 988 | 112 | 20 | no |
| 7 | 4.50×10^17 | 1.02×10^9 | 3.78×10^17 | 6.69×10^8 | 1244 | 112 | 20 | no |
| 8 | 4.57×10^17 | 1.02×10^9 | 7.00×10^15 | 0 | 1445 | 112 | 20 | no |
| 9 | 1.04×10^18 | 4.79×10^11 | 5.87×10^17 | 4.78×10^11 | 1445 | 256 | 20 | no |
| 10 | 1.04×10^18 | 4.79×10^11 | 0 | 0 | 1445 | 256 | 20 | no |
| 11 | 1.04×10^18 | 4.79×10^11 | 0 | 0 | 1445 | 256 | 20 | no |
| 12 | 2.08×10^23 | 1.16×10^15 | 2.08×10^23 | 1.16×10^15 | 1445 | 256 | 20 | yes |
<!-- END:tiers_npp_gap0 -->

Prior only, over the flat pools:

<!-- BEGIN:tiers_p_gap0 -->
| tier | S_t | K_t | shell S | shell K | ADDR kept | MATCH kept | STEP kept | absent letters kept |
|---|---|---|---|---|---|---|---|---|
| 0 | 2.36×10^6 | 0 | 2.36×10^6 | 0 | 1 | 9 | 4 | no |
| 1 | 4.27×10^17 | 0 | 4.27×10^17 | 0 | 66 | 63 | 254 | no |
| 2 | 2.41×10^22 | 9.65×10^14 | 2.41×10^22 | 9.65×10^14 | 944 | 144 | 1268 | no |
| 3 | 2.48×10^22 | 9.99×10^14 | 7.53×10^20 | 3.39×10^13 | 951 | 144 | 1280 | no |
| 4 | 2.48×10^22 | 9.99×10^14 | 0 | 0 | 952 | 144 | 1280 | no |
| 5 | 6.21×10^22 | 3.49×10^15 | 3.72×10^22 | 2.49×10^15 | 952 | 360 | 1280 | no |
| 6 | 1.74×10^23 | 1.35×10^16 | 1.12×10^23 | 1.01×10^16 | 1005 | 1008 | 1280 | no |
| 7 | 1.07×10^24 | 7.83×10^16 | 8.99×10^23 | 6.47×10^16 | 1261 | 1008 | 1280 | no |
| 8 | 1.09×10^24 | 7.84×10^16 | 1.66×10^22 | 1.06×10^14 | 1462 | 1008 | 1280 | no |
| 9 | 2.49×10^24 | 3.65×10^17 | 1.40×10^24 | 2.86×10^17 | 1462 | 2304 | 1280 | no |
| 10 | 2.49×10^24 | 3.65×10^17 | 0 | 0 | 1462 | 2304 | 1280 | no |
| 11 | 2.49×10^24 | 3.65×10^17 | 0 | 0 | 1462 | 2304 | 1280 | no |
| 12 | 5.03×10^29 | 5.29×10^21 | 5.03×10^29 | 5.29×10^21 | 1462 | 2304 | 1280 | yes |
<!-- END:tiers_p_gap0 -->

What the prior learned from the earlier domains:

<!-- BEGIN:prior -->
| kind | feature tuple | survived / rows | score |
|---|---|---|---|
| ADDR | `('add', ('constant', 'constant'))` | 0/49 | 0.0020 |
| ADDR | `('add', ('constant', 'length'))` | 0/14 | 0.0066 |
| ADDR | `('add', ('constant', 'position'))` | 1/41 | 0.0262 |
| ADDR | `('add', ('length', 'length'))` | 0/1 | 0.0494 |
| ADDR | `('identity', ('constant',))` | 4/13 | 0.2928 |
| ADDR | `('identity', ('length',))` | 0/2 | 0.0329 |
| ADDR | `('sub', ('constant', 'constant'))` | 15/85 | 0.1756 |
| ADDR | `('sub', ('constant', 'length'))` | 2/26 | 0.0777 |
| ADDR | `('sub', ('length', 'length'))` | 1/2 | 0.3662 |
| ADDR | `('add', ('constant', 'constant'), False)` | 0/18 | 0.0052 |
| ADDR | `('add', ('constant', 'constant'), True)` | 0/31 | 0.0031 |
| ADDR | `('add', ('constant', 'length'), False)` | 0/14 | 0.0066 |
| ADDR | `('add', ('constant', 'position'), False)` | 0/11 | 0.0082 |
| ADDR | `('add', ('constant', 'position'), True)` | 1/30 | 0.0354 |
| ADDR | `('add', ('length', 'length'), False)` | 0/1 | 0.0494 |
| ADDR | `('identity', ('constant',), False)` | 0/1 | 0.0494 |
| ADDR | `('identity', ('constant',), True)` | 4/12 | 0.3153 |
| ADDR | `('identity', ('length',), False)` | 0/2 | 0.0329 |
| ADDR | `('sub', ('constant', 'constant'), False)` | 0/42 | 0.0023 |
| ADDR | `('sub', ('constant', 'constant'), True)` | 15/43 | 0.3432 |
| ADDR | `('sub', ('constant', 'length'), False)` | 0/15 | 0.0062 |
| ADDR | `('sub', ('constant', 'length'), True)` | 2/11 | 0.1749 |
| ADDR | `('sub', ('length', 'length'), True)` | 1/2 | 0.3662 |
| LIT | `(False,)` | 0/379 | 0.0001 |
| LIT | `(True,)` | 22/45 | 0.4794 |
| TRUTH | `(0,)` | 0/3 | 0.0260 |
| TRUTH | `(1,)` | 1/3 | 0.2760 |
| TRUTH | `(10,)` | 0/3 | 0.0260 |
| TRUTH | `(11,)` | 0/3 | 0.0260 |
| TRUTH | `(12,)` | 0/3 | 0.0260 |
| TRUTH | `(13,)` | 0/3 | 0.0260 |
| TRUTH | `(14,)` | 0/3 | 0.0260 |
| TRUTH | `(15,)` | 0/3 | 0.0260 |
| TRUTH | `(2,)` | 1/3 | 0.2760 |
| TRUTH | `(3,)` | 0/3 | 0.0260 |
| TRUTH | `(4,)` | 0/3 | 0.0260 |
| TRUTH | `(5,)` | 0/3 | 0.0260 |
| TRUTH | `(6,)` | 0/3 | 0.0260 |
| TRUTH | `(7,)` | 1/3 | 0.2760 |
| TRUTH | `(8,)` | 2/3 | 0.5260 |
| TRUTH | `(9,)` | 0/3 | 0.0260 |
| STEP | `('add',)` | 2/10 | 0.2091 |
| STEP | `('sub',)` | 4/10 | 0.3909 |
| STEP | `('add', '2pixel')` | 0/2 | 0.1000 |
| STEP | `('add', '3pixel')` | 0/2 | 0.1000 |
| STEP | `('add', 'pixel')` | 1/2 | 0.4333 |
| STEP | `('add', 'row')` | 1/2 | 0.4333 |
| STEP | `('add', 'row+pixel')` | 0/2 | 0.1000 |
| STEP | `('sub', '2pixel')` | 0/2 | 0.1000 |
| STEP | `('sub', '3pixel')` | 0/2 | 0.1000 |
| STEP | `('sub', 'pixel')` | 2/2 | 0.7667 |
| STEP | `('sub', 'row')` | 2/2 | 0.7667 |
| STEP | `('sub', 'row+pixel')` | 0/2 | 0.1000 |
<!-- END:prior -->

The source searches it was fitted on reproduce their recorded certificates:

<!-- BEGIN:sources -->
| source | space | conforming | certificate | recorded |
|---|---|---|---|---|
| s19_stage_a | 10496 | 1 | `unique` | {'base': 14, 'byte': 40, 'conforming': 1, 'space': 10496} |
| s23_transform | 7480 | 2 | `complete` | 2 |
| s23_policy | 448 | 21 | `complete` | 21 |
| §33 s0 (res 32, committed) | 256 | 2 | `complete` | same file |
| §33 s1 (res 32, committed) | 400 | 2 | `complete` | same file |
| §33 s2 (res 32, committed) | 25 | 1 | `unique` | same file |

Rows per kind (rows / survivors): ADDR 233/23, GROUND 0/0, LIT 424/22, STEP 20/6, TRUTH 48/5
<!-- END:sources -->

## 5. Hard-transferred vector, baselines, achieved difficulty

<!-- BEGIN:extras_gap0 -->
| quantity | value |
|---|---|
| hard-transferred vector H: training accuracy | 0.2500 |
| H: held-out accuracy | 0.1389 |
| best constant click (chosen on training) held-out | 0.5000 |
| oracle constant click (chosen on held-out) | 0.6944 |
| uniform random pixel, held-out (exact) | 0.3099 |
| uniform random child widget, held-out (exact) | 0.4259 |
| widgets per screen (48 episodes) | {'3': 24, '4': 24} |
| rejection draws per episode (min / median / max) | 1 / 6 / 78 |
| instruction lengths (bytes) | [23, 24, 25, 26, 32, 33, 34, 35, 36] |
| target area in pixels (min / max) | 42 / 140 |
| H's selection | cpos ('sub', 'length', 'k1'), lc1 97, lc2 97, lc3 97, K1 0, K2 0, K3 0, K4 0, ra ('sub', 'length', 'k1'), lr1 97, lr2 97, M (1, 2, 7, 2), X1 ('add', 'lo', 3), X2 ('add', 'lo', 48), X3 ('add', 'lo', 48) |
<!-- END:extras_gap0 -->

<!-- BEGIN:extras_gap1 -->
(not run)
<!-- END:extras_gap1 -->

## 6. Why — instrumentation of the found programs

<!-- BEGIN:instrument_gap0 -->
(not run)
<!-- END:instrument_gap0 -->

## 7. Validation of the instruments

<!-- BEGIN:validation -->
| check | result |
|---|---|
| V2 counter = brute force, gap 0 | 35/35 sub-spaces agree on S and K; 10 contain conformers |
| V2 counter = `tcn.search.enumerate_prefix` | steps_and_relation: S 192, K 12, tcn `complete` 12; grounding_and_steps: S 64, K 1, tcn `unique` 1; matcher_and_literals: S 32, K 2, tcn `complete` 2 — 3/3 agree |
<!-- END:validation -->

## 8. Resources

<!-- BEGIN:resources -->
| capped job | peak RSS (GB) | wall clock | exit |
|---|---|---|---|
| `arm_gap0_A_noobs` | 0.27 | 1:39.54 | 0 |
| `arm_gap0_Dp` | 0.27 | 0:42.70 | 0 |
| `arm_gap0_Dpp_0` | 0.28 | 1:58.48 | 0 |
| `arm_gap0_Dpp_1` | 0.27 | 1:42.20 | 0 |
| `arm_gap0_Dpp_2` | 0.29 | 2:22.74 | 0 |
| `arm_gap0_Dpp_3` | 0.27 | 1:44.46 | 0 |
| `arm_gap0_Dpp_4` | 0.27 | 1:38.02 | 0 |
| `arm_gap0_KO_ADDR` | 0.27 | 1:53.18 | 0 |
| `arm_gap0_KO_STEP` | 0.27 | 2:23.89 | 0 |
| `arm_gap0_KO_TRUTH` | 0.27 | 2:19.11 | 0 |
| `arm_gap0_N` | 4.23 | 1:38.80 | 0 |
| `arm_gap0_Np` | 0.31 | 0:30.33 | 0 |
| `arm_gap0_Npp` | 0.27 | 2:17.59 | 0 |
| `arm_gap0_OCC_1` | 0.27 | 1:59.38 | 0 |
| `arm_gap0_OCC_10` | 0.27 | 2:07.31 | 0 |
| `arm_gap0_OCC_100` | 0.27 | 1:56.55 | 0 |
| `arm_gap0_OCC_3` | 0.27 | 1:58.98 | 0 |
| `arm_gap0_OCC_3500` | 0.27 | 2:19.62 | 0 |
| `arm_gap0_P` | 0.82 | 7:05.20 | 0 |
| `arm_gap0_R_0` | 0.27 | 1:31.89 | 0 |
| `arm_gap0_R_1` | 0.27 | 1:20.11 | 0 |
| `arm_gap0_R_2` | 0.27 | 2:06.01 | 0 |
| `arm_gap0_R_3` | 0.27 | 1:44.80 | 0 |
| `arm_gap0_R_4` | 0.28 | 1:13.48 | 0 |
| `arm_gap0_STEP_hand` | 0.26 | 0:52.31 | 0 |
| `arm_gap0_STEP_only` | 0.27 | 1:16.81 | 0 |
| `arm_gap0_U_Vonly` | 0.31 | 0:53.04 | 0 |
| `arm_gap0_U_feat` | 0.31 | 0:59.10 | 0 |
| `arm_gap0_U_steep` | 0.30 | 1:19.06 | 0 |
| `arm_gap0_V_1` | 0.27 | 1:56.70 | 0 |
| `arm_gap0_V_10` | 0.27 | 2:49.43 | 0 |
| `arm_gap0_V_100` | 0.27 | 3:32.60 | 0 |
| `arm_gap0_V_3` | 0.27 | 2:25.93 | 0 |
| `arm_gap0_extras` | 0.24 | 0:01.64 | 0 |
| `cache` | 0.22 | 0:24.52 | 0 |
| `diag_v1` | 0.25 | 2:29.91 | 0 |
| `first_generalizing_gap0` | 0.38 | 20:40.25 | 0 |
| `generalize_gap0` | 0.39 | 5:31.25 | 0 |
| `sources` | 0.28 | 1:30.87 | 0 |
| `v2brute_gap0` | 0.49 | 0:41.46 | 0 |
| `v2tcn_gap0` | 0.30 | 7:04.85 | 0 |

Memory floor: 52 capped starts logged in `out/memory_floor.log`; MemAvailable at start ranged 33.3–42.2 GB; phases refused by the 25 GB floor: 0.
<!-- END:resources -->

---

## Pre-sweep prediction (written `2026-09-10T12:10:01-07:00`, before any OCC_ or V_ arm has run)

**A conjecture, not a finding:** drawn from three R seeds and crude cell-mean statistics
(`R_0`: occurs ratio `2.4`, V positive, `0/400`; `R_1`: `3.9`, V positive, `400/400`;
`R_2`: `3.9`, V inverted, `0/400`). The coordinator proposed it; it is recorded here so that the
sensitivity sweeps test it rather than being fitted to it.

> Generalization requires BOTH (a) a positive V preference on the address slots AND (b) an occurs
> ratio on the literal slots above a threshold somewhere between `2.4` and `3.9`.

Predicted outcome of each sweep point, stated now:

| sweep point | what is held | predicted |
|---|---|---|
| `OCC_1` | V fitted (positive) | no generalization — (b) fails |
| `OCC_3` | V fitted (positive) | undetermined — inside the threshold band |
| `OCC_10`, `OCC_100`, `OCC_3500` | V fitted (positive) | generalization; `OCC_3500` should reproduce N″ |
| `V_1` | occurs at N″'s fitted ratio | no generalization — (a) fails (zero preference is not positive) |
| `V_3`, `V_10`, `V_100` | occurs at N″'s fitted ratio | generalization |

Any row whose outcome differs is reported as falsifying the corresponding clause.

---

## 9. After the pre-registered arms — controls, the distractor's reach, and exploratory measurements

Everything in this section is **post-hoc** unless it says otherwise. None of it changes §1.

### 9.1 Could the pre-registered distractor have succeeded at all?

Exact counts over all 48 episodes (training and held-out together):

The same count at the second configuration (`gap 1`). **Post-hoc, and not F6**: F6 is "0
conforming, exhausted" on the training episodes, and the `gap 1` schema space has training
conformers, so F6 is not triggered as pre-registered. This count asks a different, stricter
question — does any program in the space conform on training *and* held-out? If the answer is
none, the schema fits training but holds nothing that generalizes: weaker than F6's
counterexample, and proposed as a stricter CEGIS check for next time ("conformance on held-out as
well as training"), not counted as F6 having fired. If some programs do generalize, N″ simply
reached a spurious one first.

<!-- BEGIN:generalize_gap1 -->
<!-- END:generalize_gap1 -->

**What it shows (post-hoc, not F6):** at `gap 1` the schema space holds **no** program that conforms
on all 48 episodes, exhausted. The step schema's offset pool fits the training episodes but
contains nothing that generalizes, so no order over it — learned or hand-written — could have
returned a generalizing program. This is the stricter check this track proposes for next time
(§9.5): a schema instantiation must conform on held-out as well as on training before it counts
as admitted, and a failure is a counterexample against the schema class. By the pre-registered
definition, F6 did not fire.

`gap 0`:

<!-- BEGIN:generalize_gap0 -->
| space | programs | conform on all 48 episodes (train + held-out) | certificate | expected programs to first generalizing, uniform order |
|---|---|---|---|---|
| distractor pools (D′, D″) | 2.08×10^23 | 0 | `complete` | no generalizing program exists |
| schema pools (N′) | 2.08×10^23 | 6.87×10^8 | `complete` | 3.03×10^14 |

N″'s tiered order, to its first generalizing program: 1.04×10^12 programs.
Distractor address expressions equal to `length−5` on every episode: 0 of 1445.
Direct 48-episode counting equals the transform on the 12 training episodes: True.

**The pre-registered distractor could not succeed**: its space contains no program that conforms on all 48 episodes, so D′/D″'s failure to generalize is true by construction and says nothing about the library. The matched-distractor role is carried by R (right schemas, permuted prior) and U (unfitted weights).
<!-- END:generalize_gap0 -->

### 9.2 The pre-sweep prediction, tested

Each sweep point changes exactly one thing in N″ (the literal-slot occurs ratio, or the
address-slot V penalty). The prediction above was committed before any of them ran.

**The sweeps are sensitivity curves, not arms of the library.** Every sweep point was run on the
target task, so whichever point is cheapest was *selected on the target*: it is tuned, not
inherited, and no sweep minimum is quoted anywhere as what the library achieves. Cost claims in
this document are made only for arms whose settings were fixed before the target was seen — N,
N′, N″, P, R, U_feat, U_steep, U_Vonly, the KO_ knockouts and the STEP_ arms. One genuine finding
about the prior does come out of the curves: **N″'s fitted V steepness is not the cost-optimal
setting for this task.** It was fitted on the earlier domains' survival rates, and nothing about
that fitting guarantees a cost-optimal steepness on a new task; the V curve shows it is not.

<!-- BEGIN:sweep -->
| sweep point | predicted (pre-sweep) | observed generalizing / sampled (95% Wilson) | expected programs | verdict |
|---|---|---|---|---|
| OCC_1 | does not | 400/400 (0.990–1.000) | 5.51×10^17 | **FALSIFIES clause (b)** |
| OCC_3 | — | 400/400 (0.990–1.000) | 4.50×10^15 | undetermined by the prediction |
| OCC_10 | generalizes | 400/400 (0.990–1.000) | 1.04×10^12 | as predicted |
| OCC_100 | generalizes | 400/400 (0.990–1.000) | 1.04×10^12 | as predicted |
| OCC_3500 | generalizes | 400/400 (0.990–1.000) | 1.04×10^12 | as predicted |
| V_1 | does not | 400/400 (0.990–1.000) | 1.69×10^12 | **FALSIFIES clause (a)** |
| V_3 | generalizes | 400/400 (0.990–1.000) | 1.96×10^10 | as predicted |
| V_10 | generalizes | 400/400 (0.990–1.000) | 1.96×10^10 | as predicted |
| V_100 | generalizes | 400/400 (0.990–1.000) | 1.96×10^10 | as predicted |

A point counts as generalizing when more than half its sampled first solutions conform on all 36 held-out episodes. Falsified at: OCC_1, V_1.
<!-- END:sweep -->

**The pre-sweep prediction is falsified on both clauses.** Clause (b) — an occurs ratio above a
threshold is required — falls at OCC_1, which generalizes with no occurs preference at all.
Clause (a) — a positive V preference is required — falls at V_1, which generalizes with the V
penalty removed. Every point predicted to generalize does. So, over the ranges tried and with the
rest of N″'s fitted content held fixed, **generalization is insensitive to both the occurs ratio
and the V penalty; only the cost moves.** That agrees with the per-kind knockouts (§9.3): the
step preference decides whether the first hit generalizes, and the address and literal
preferences set how many programs it takes. The prediction was a conjecture drawn from three R
seeds and crude statistics, and it is reported as falsified rather than revised.

### 9.3 What the prior prefers, against how often its first solution generalizes

U_feat (N″'s features with **unfitted** weights) is the primary clean control. R is a
**within-slot permutation that preserves each source slot's base rates**; it weakens the prior's
preferences without destroying them, so it is not a "content destroyed" control.

<!-- BEGIN:mechanism -->
| arm | pools | LIT occurs ratio | ADDR mean V=True / V=False | rank of `sub(length,5)` in cpos | best rank of an address = 13 in ra | absent letters enter at tier | generalizing / 400 | expected programs |
|---|---|---|---|---|---|---|---|---|
| N'' | schema | 3,510.87 | 0.206 / 0.020 | 57 | 2 | 12 | 400 | 1.04×10^12 |
| P | flat | 3,510.87 | 0.206 / 0.020 | — | — | — | 0 | 4.27×10^17 |
| R_0 | schema | 2.45 | 0.171 / 0.079 | 57 | 2 | 2 | 0 | 1.73×10^21 |
| R_1 | schema | 3.87 | 0.183 / 0.129 | 2 | 214 | 2 | 400 | 5.47×10^13 |
| R_2 | schema | 3.87 | 0.101 / 0.158 | 2 | 214 | 2 | 0 | 3.53×10^19 |
| R_3 | schema | 3.11 | 0.086 / 0.118 | 202 | 213 | 2 | 2 | 3.64×10^20 |
| R_4 | schema | 1.85 | 0.136 / 0.047 | 1 | 12 | 1 | 0 | 1.30×10^22 |
| D''_0 | distractor | 2.45 | 0.171 / 0.079 | — | — | — | 0 | 1.23×10^14 |
| D''_1 | distractor | 3.87 | 0.183 / 0.129 | — | — | — | 0 | 3.08×10^15 |
| D''_2 | distractor | 3.87 | 0.101 / 0.158 | — | — | — | 0 | 1.23×10^14 |
| D''_3 | distractor | 3.11 | 0.086 / 0.118 | — | — | — | 0 | 3.08×10^15 |
| D''_4 | distractor | 1.85 | 0.136 / 0.047 | — | — | — | 0 | 2.03×10^22 |
| U_feat | schema | 2 | rule | 1 | 1 | 1 | 0 | 2.04×10^6 |
| U_steep | schema | 2 | rule | 1 | 1 | 12 | 0 | 2.04×10^6 |
| U_Vonly | schema | 1 | rule | 1 | 1 | 0 | 0 | 4.41×10^7 |
| KO_ADDR | schema | 3,510.87 | rule | 1 | 1 | 12 | 400 | 3.05×10^13 |
| KO_TRUTH | schema | 3,510.87 | 0.206 / 0.020 | 57 | 2 | 12 | 400 | 3.77×10^13 |
| KO_STEP | schema | 3,510.87 | 0.206 / 0.020 | 57 | 2 | 12 | 8 | 1.61×10^13 |
| STEP_only | schema | 1 | rule | 1 | 1 | 0 | 400 | 1.66×10^21 |
| STEP_hand | schema | 1 | rule | 1 | 1 | 0 | 400 | 3.30×10^16 |
| A_noobs | schema | 1 | rule | 1125 | 2 | 0 | 1 | 5.04×10^20 |
| OCC_1 | schema | 1 | 0.206 / 0.020 | 57 | 2 | 0 | 400 | 5.51×10^17 |
| OCC_3 | schema | 3 | 0.206 / 0.020 | 57 | 2 | 2 | 400 | 4.50×10^15 |
| OCC_10 | schema | 10 | 0.206 / 0.020 | 57 | 2 | 4 | 400 | 1.04×10^12 |
| OCC_100 | schema | 100 | 0.206 / 0.020 | 57 | 2 | 7 | 400 | 1.04×10^12 |
| OCC_3500 | schema | 3,500.00 | 0.206 / 0.020 | 57 | 2 | 12 | 400 | 1.04×10^12 |
| V_1 | schema | 3,510.87 | 0.206 / 0.020 | 258 | 2 | 12 | 400 | 1.69×10^12 |
| V_3 | schema | 3,510.87 | 0.206 / 0.020 | 56 | 1 | 12 | 400 | 1.96×10^10 |
| V_10 | schema | 3,510.87 | 0.206 / 0.020 | 56 | 1 | 12 | 400 | 1.96×10^10 |
| V_100 | schema | 3,510.87 | 0.206 / 0.020 | 56 | 1 | 12 | 400 | 1.96×10^10 |

Over the 24 schema-pool arms, Spearman(log occurs ratio, generalizing) = 0.458; Spearman(−colour-address rank, generalizing) = -0.227.
<!-- END:mechanism -->

### 9.4 Programs to the first GENERALIZING program — exploratory, not a verdict

**Post-hoc throughout; C1's pre-registered failure stands as the first line of this document.**
One column per arm: the expected programs a search following that arm's order evaluates before
reaching a program that conforms on all 48 episodes. Exact where the arm's tiers can be counted
over 48 episodes; a Wilson-interval bound from exact uniform samples for the flat pools; infinite
where the space holds no generalizing program.

How to read it. The column assumes a search that checks each candidate against held-out
conformance, so a spurious training fit is rejected rather than returned; that is the same
assumption for every arm. Arms whose settings were fixed before the target was seen are the only
ones whose figures are claims; the OCC_/V_ rows are target-tuned sensitivity points and their
minimum is not quoted as a result. Among the pre-fixed arms, the hand-feature controls come out
cheaper than N″, and N″ cheaper than N′.

<!-- BEGIN:first_generalizing_gap0 -->
| arm | programs to first training-conforming (pre-registered metric) | programs to first GENERALIZING program (post-hoc) | how obtained |
|---|---|---|---|
| N — flat, uniform | 9.50×10^7 | pending | — |
| N′ — schema, uniform | 1.79×10^8 | 3.03×10^14 | exact, 48-episode count, tier 0 |
| N″ — schema + learned prior | 1.04×10^12 | 1.04×10^12 | exact, 48-episode count, tier 2 |
| P — flat + learned prior | 4.27×10^17 | pending | — |
| D′ — distractor schemas, uniform | 1.86×10^14 | ∞ — no generalizing program exists | exact, 48-episode count |
| N″ without observation features | 5.04×10^20 | 5.04×10^20 | exact, 48-episode count, tier 3 |
| D″ — distractor schemas + permuted prior (seed 0) | 1.23×10^14 | ∞ — no generalizing program exists | exact, 48-episode count |
| D″ — distractor schemas + permuted prior (seed 1) | 3.08×10^15 | ∞ — no generalizing program exists | exact, 48-episode count |
| D″ — distractor schemas + permuted prior (seed 2) | 1.23×10^14 | ∞ — no generalizing program exists | exact, 48-episode count |
| D″ — distractor schemas + permuted prior (seed 3) | 3.08×10^15 | ∞ — no generalizing program exists | exact, 48-episode count |
| D″ — distractor schemas + permuted prior (seed 4) | 2.03×10^22 | ∞ — no generalizing program exists | exact, 48-episode count |
| R — right schemas + permuted prior (seed 0) | 1.73×10^21 | 2.01×10^22 | exact, 48-episode count, tier 6 |
| R — right schemas + permuted prior (seed 1) | 5.47×10^13 | 5.47×10^13 | exact, 48-episode count, tier 2 |
| R — right schemas + permuted prior (seed 2) | 3.53×10^19 | 3.53×10^19 | exact, 48-episode count, tier 3 |
| R — right schemas + permuted prior (seed 3) | 3.64×10^20 | 3.64×10^20 | exact, 48-episode count, tier 3 |
| R — right schemas + permuted prior (seed 4) | 1.30×10^22 | 1.30×10^22 | exact, 48-episode count, tier 4 |
| U — N″'s features, UNFITTED gentle 2× rule (post-hoc control) | 2.04×10^6 | 3.36×10^10 | exact, 48-episode count, tier 0 |
| U_steep — hand rules at N″'s steepness, nothing fitted (post-hoc control) | 2.04×10^6 | 3.36×10^10 | exact, 48-episode count, tier 0 |
| U_Vonly — one hand rule, V 10:1, all else uniform, nothing fitted (post-hoc control) | 4.41×10^7 | 7.48×10^13 | exact, 48-episode count, tier 0 |
| KO_ADDR — N″ with ADDR (§19/§23 rows) knocked out to hand V 10:1 (post-hoc) | 3.05×10^13 | 3.05×10^13 | exact, 48-episode count, tier 1 |
| KO_TRUTH — N″ with TRUTH (§33 rows) knocked out to uniform (post-hoc) | 3.77×10^13 | 3.77×10^13 | exact, 48-episode count, tier 2 |
| KO_STEP — N″ with STEP (§33 rows) knocked out to uniform (post-hoc) | 1.61×10^13 | 1.61×10^13 | exact, 48-episode count, tier 2 |
| STEP_only — only the fitted STEP estimator (20 §33 rows), all else uniform (post-hoc) | 1.66×10^21 | 1.66×10^21 | exact, 48-episode count, tier 1 |
| STEP_hand — hand rule: pixel or row offsets 1, else 0.1; all else uniform (post-hoc) | 3.30×10^16 | 3.30×10^16 | exact, 48-episode count, tier 0 |
| N″, occurs ratio pinned at 1 — sensitivity, set on the target task, not inherited | 5.51×10^17 | 5.51×10^17 | exact, 48-episode count, tier 2 |
| N″, occurs ratio pinned at 3 — sensitivity, set on the target task, not inherited | 4.50×10^15 | 4.50×10^15 | exact, 48-episode count, tier 2 |
| N″, occurs ratio pinned at 10 — sensitivity, set on the target task, not inherited | 1.04×10^12 | 1.04×10^12 | exact, 48-episode count, tier 2 |
| N″, occurs ratio pinned at 100 — sensitivity, set on the target task, not inherited | 1.04×10^12 | 1.04×10^12 | exact, 48-episode count, tier 2 |
| N″, occurs ratio pinned at 3500 — sensitivity, set on the target task, not inherited | 1.04×10^12 | 1.04×10^12 | exact, 48-episode count, tier 2 |
| N″, V ratio pinned at 1 — sensitivity, set on the target task, not inherited | 1.69×10^12 | 1.69×10^12 | exact, 48-episode count, tier 2 |
| N″, V ratio pinned at 3 — sensitivity, set on the target task, not inherited | 1.96×10^10 | 1.96×10^10 | exact, 48-episode count, tier 1 |
| N″, V ratio pinned at 10 — sensitivity, set on the target task, not inherited | 1.96×10^10 | 1.96×10^10 | exact, 48-episode count, tier 1 |
| N″, V ratio pinned at 100 — sensitivity, set on the target task, not inherited | 1.96×10^10 | 1.96×10^10 | exact, 48-episode count, tier 1 |
<!-- END:first_generalizing_gap0 -->

Sampled first-shell conformers, for the arms whose first hit does not always generalize:

<!-- BEGIN:posthoc_gap0 -->
| arm | exact cost to a training-conforming program | sampled first-shell conformers that generalize | cost to a first generalizing program (estimate / bounds) | exact, from 48-episode counts |
|---|---|---|---|---|
<!-- END:posthoc_gap0 -->

### 9.5 Proposed follow-up pre-registration (a proposal, not run here)

1. **Criterion on held-out conformance.** Cost is programs and episodes to the first program that
   conforms on the training episodes **and** on a disjoint validation split, with the verdict
   scored on a third, untouched split.
2. **Enough training episodes that spurious fits are rare**, sized from this track's measured
   spurious-fit densities, so that the flat substrate's first hit is not a spurious one.
3. **The distractor must be shown able to succeed** (its space must contain a generalizing
   program, counted exactly) before its failure is read as evidence.
4. **The observation-conditioned features (V, occurs) are part of the library under test**, with
   the unfitted-weight control (U) and the sweep as pre-registered arms, not added afterwards.
5. **A stricter CEGIS admission check for schemas.** A schema instantiation counts as admitted
   only if its space contains a program that conforms on held-out episodes as well as on
   training, counted exactly; a failure is a counterexample recorded against the schema class,
   which is then refined or split. F6 as written here ("0 conforming, exhausted" on training)
   could not see the `gap 1` failure of the step schema's offset pool (§9.1).
