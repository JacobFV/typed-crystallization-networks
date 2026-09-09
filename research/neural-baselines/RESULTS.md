# Matched neural baselines for the visual, language and computer artifacts

`research/FINDINGS.md` §36 ends with the largest unsupported claim in this
project: *"no matched neural baseline exists for the visual, computer or
language artifacts, so **'cheaper than a model' remains unmeasured for all
three**."* This track replaces that sentence with numbers.

It is the second matched-baseline pass in the repository. The first,
`research/baselines/RESULTS.md`, covered only the `mixed` and `joint` fixtures;
its method — same observable inputs, same supervision, same splits, trivial
reference beside every quality figure, tuning disclosed on both sides, both
sides re-timed on one host — is followed here deliberately and is cited where a
choice is inherited from it.

**The question is not "can a typed program beat a model."** It is: *for equal
observable information and comparable task quality, what representation,
execution and training cost does each method require?* The short answer is that
**the premise of the §36 sentence does not survive**: on these three artifacts
the two methods never reach comparable task quality, so "cheaper at equal
quality" is not a question that has an answer here. What can be reported, and is,
is the cost of each method at the quality it actually achieves — and on that
reading the typed program wins task quality on all three arms, loses execution
cost on two of them by up to 556×, wins deployment footprint on two, and wins
training wall clock on one. Every one of those is reported the same way.

Everything below was produced in this worktree on 2026-09-09 with the repository
`.venv` (Python 3.13.15, torch 2.14.0+cpu, `torch.set_num_threads(1)`), Linux,
20 cores. **The host was heavily loaded throughout** — a dozen other jobs, load
average 32–65 — so every wall clock is a median beside a load-independent count
where one exists, and the latency caveat of `research/baselines/RESULTS.md` §5
applies unchanged. Nothing under `tcn/` or `generators/` was modified and no
other track's files were written.

---

## 0. Verdict, one line per artifact

Full tables in §3–§5, cost in §6.1, the reasoning in §6.2.

* **Visual** (screenshot → exact widget hierarchy, 12 held-out screens, 227
  widgets). The typed program is **227/227 rectangles, 227/227 links, 12/12 trees
  from 6 training screens**; the best matched CNN reaches **211/227 rectangles and
  0/12 trees from 768 screens** — 128× the data and still not exact. At *its*
  quality the CNN costs **300–556× less execution time** (31–61 ms against
  18,375 ms per screen) and **250× less on disk** (160 KB of weights against
  40 MB of JSON, or 184 KB gzipped). **The typed program wins quality and sample
  efficiency; the neural baseline wins latency and size. The qualities are not
  equal, so there is no single winner — and that is the finding.**
* **Language** (bracket balancedness from raw prompt bytes, 724 held-out
  episodes, 0.548 majority). Typed program **0.9986**; the best matched net,
  selected by validation, **0.657** (a 3,330-parameter GRU); the best single seed
  anywhere in a nine-architecture sweep, 0.782. **No cost trade exists because no
  baseline reaches comparable quality.** The cheapest net that gets near
  (a 9,123-parameter CNN at 0.604) is 26× cheaper per inference and 80× cheaper to
  fit; neither is solving the task.
* **Computer** (typed policy in a live OS, 10 held-out documents, program-only
  path). Typed agent **10/10 solved, mean return 2.00/2**; the best matched net
  **5/10**, which is the *computed ceiling* for any 256-way classifier on this
  held-out set and which it attains. Program-only latency 13.35 ms against
  0.39–9.85 ms — same order — and 18,927 B gzipped against 13,492–30,832 B of
  weights — a tie. **The typed program wins quality; cost is a wash.**

---

## 1. What "matched" means here, and how it was enforced

Four conditions, stated per arm in §3–§5 and enforced in code:

1. **Exactly the same observable inputs.** Raw bytes where the TCN program gets
   raw bytes. The baseline's input tensor is built from the same `Value` the
   program's input port receives, and from nothing else.
2. **No privileged probes as model inputs.** `probes` and `latent_states` build
   supervision targets and never reach a model input, on either side. This is the
   project's standing rule (`AGENTS.md`); each arm's script asserts the
   observation set it read.
3. **The same train / validation / test episodes**, by index, as the track that
   froze the artifact. Model selection uses training fit (what the TCN search
   had) and, where a validation split exists, that split — never the held-out set.
4. **The same supervision available to both.** The TCN path gets dense
   hierarchical supervision; every baseline is offered the same channel, and in
   two arms strictly more of it. Denying the baseline supervision the TCN had
   would be a strawman in the other direction.

And two disciplines carried over from the earlier pass:

* **A trivial reference beside every quality number** — majority constant,
  random, `parent = root`, always-write-"5" — because a quality figure without
  its trivial reference is uninterpretable.
* **The achieved configuration is reported, never the requested one.** Where a
  sweep was cut for wall-clock reasons the cut is stated in §7.

### What the baselines were NOT given

Three things were withheld from every matched arm and reported separately, in the
same spirit as `research/baselines/RESULTS.md` §6 declining Fourier features on
the mixed fixture: bag-of-bytes **counting features** (language), a **computed
relative address** `length - k` (computer), and **colour-bijection augmentation**
(visual). Each of these hands the model the abstraction the TCN's search had to
find. They are run and reported as *declared structural hints*, apart from the
matched tables, because the size of the gap each one closes is the most
informative number in this track.

---

## 2. Which language stream this measures, and why it has to be said

**The language lesson was re-drawn after its artifact was frozen.** FINDINGS §24
re-drew 14 exploitable lessons, `context_free_language` among them. The re-draw
moved string length from 2–16 to **10–22**, and the language-capability track
holds out *length*, training on {2,4,6} — so on today's default stream that
track's training split is empty. This track hit that wall independently while
building the dataset, and it is now recorded on main as **FINDINGS §39**
(`research/language-capability/reproduce/stream_check.py`), which reaches the same
conclusion from the supervising session.

**Decision, stated once and held to.** §3's comparison is run entirely on the
**pre-audit stream, `hardening="none"`** — the stream §19's artifact was searched
and scored on. `langdata.py` pins it explicitly rather than inheriting the
generator default, which is what §39 recommends the track itself should do. Both
methods train on it and both are held out on it. The reproduction here matches the
recorded splits exactly: 24 training episodes over 12 distinct strings, 120
seen-length validation episodes, **724** held-out episodes at lengths 8–16 with a
**0.5483** majority constant and 0% string overlap with training.

**Rebuilt on that stream, the frozen program scores 0.9986187845303868**, which is
`final_eval.json`'s recorded value to every digit and §39's reproduction to five.
FINDINGS §19 and `STATUS.md` §1 row 9 both round it to "**1.000**"; the per-length
breakdown is `{8: 1.0, 10: 1.0, 12: 1.0, 14: 1.0, 16: 0.5}` — the accumulator is
16 steps deep and misses one of the two length-16 episodes. 723/724 against a
0.548 majority is still decisive, and no conclusion below turns on it, but the
prose figure should be 0.9986.

**The post-audit stream is reported separately, and only as transfer.** Neither
method has been *trained* on it. What §3.4 reports is the legacy-trained artifact
and the legacy-trained baselines both run on 400 post-audit episodes — an
out-of-distribution transfer measurement for both sides, on identical terms. It is
never quoted as a comparison on the current task, per §39's instruction. One new
fact this track adds to §39:

| on the post-audit stream, 400 episodes | accuracy |
|---|---|
| majority constant | 0.5150 |
| **bracket-counting oracle** (the rule the frozen program implements) | **0.5150** |
| true Dyck-membership oracle | **1.0000** |
| frozen TCN program (trained on the legacy stream) | 0.4650 |

The counting oracle is *exactly* the majority constant. So the hardening did not
merely shift the lengths: it severed counting from balancedness, which §19 had
measured as agreeing on 20,000 of 20,000 seeds. The task remains answerable — the
Dyck oracle is 1.000 — but not by counting, and §19 itself says the artifact
"demonstrates counting and agreement, **not recursion**." The transfer failure is
the one its own track predicted, and the capability on the post-audit
distribution remains unmeasured for the TCN side. §3.5 reports what a small
neural baseline does when it is *trained* there, precisely because there is no
TCN number to put beside it.

---

## 3. Language — grammaticality of a bracket string from raw prompt bytes

### 3.1 Matching conditions

| | TCN (`research/language-capability` stage B) | every neural baseline |
|---|---|---|
| input | `observations['text']` — `(length: u32, 128 × byte)` | the same 129 numbers: 128 byte ids and the length |
| never seen as input | `latent_states['construction']`, `probes['answer']`, seed, split, the extracted string | identical |
| supervision | staged: 44 in-string positions labelled "is this byte an opening bracket" (from `construction`), then the yes/no answer | `aux=0`: the answer only. `aux=1`: the answer **plus** a per-position running-bracket-depth target derived from the same `construction` latent — strictly denser than stage A's |
| training episodes | 24, over 12 distinct strings, lengths {2,4,6} | identical episodes |
| held out | 724 episodes, lengths 8–16, 0% string overlap | identical |
| selection | none — enumeration exhausted the space | training accuracy only, and (reported separately) the 120-episode seen-length validation split |
| stream | `hardening="none"` on both sides — see §2 | |

### 3.2 The table

Quality is accuracy on the same 724 held-out episodes. **Majority constant
0.5483, random 0.500** — every row must be read against those.

| method | params | serialized | learned content | batch-1 warm | batch-1 cold | peak RSS | training budget | **held-out accuracy** |
|---|---|---|---|---|---|---|---|---|
| **TCN frozen program** | **0 trainable** | 505,444 B JSON (`description_bits` 4,043,552); **6,634 B** gzipped | **28.8 bits** (log₂ of 10,496 × 45,375) | **2.820 ms** | 2.951 ms | 50.0 MB | 24 episodes, 44 probe positions, **55,871 programs enumerated in 361.6 s**, 0 optimizer steps | **0.9986** (723/724) |
| **best matched net, selected on validation** — `gru_16`, aux, lr 0.03, 1,000 steps | **3,330** | 16,017 B `torch.save`; 13,252 B f32 | 106,560 bits of weights | 31.1 ms (min 5.1) | 28.8 ms | 270 MB (torch) | 24 episodes × 1,000 Adam steps, 174–233 s | **0.657** (0.551 / 0.762 over 2 seeds; validation 0.888) |
| best convolutional arm, selected on validation — `cnn_32x3`, aux, lr 0.01 | 9,123 | 40,258 B; 36,492 B f32 | 291,936 bits | **0.110 ms** | 0.364 ms | 270 MB | 24 × 400, 4.5 s | 0.604 (0.565–0.634 over 3 seeds; validation 0.808) |
| best matched net, selected on training fit only — `transformer_d16`, aux | 8,403 | 39,213 B | 268,896 bits | 18.4 ms | 136.9 ms | 270 MB | 24 × 400, 4.9 s | 0.554 |
| best *any* matched arm and seed (post-hoc, not a protocol) — `gru_16`, aux | 3,330 | | | | | | | 0.782 |
| the recurrent arms at lr 0.01, 400 steps | 2,489–8,962 | | | | | | 18 runs | **17 of 18 collapsed to the training majority** (train 0.625, test 0.548); the 18th reached train 0.875, test 0.782 |
| over-budget: the same nets on all 461 short-length episodes | 3,313–22,785 | | | | | | 461 episodes | 0.351–0.548 — **worse**, not better |
| *(declared structural hint)* linear readout over bag-of-256-byte counts | **258** | 1,177 B | 8,256 bits | | | | 24 × 400 | 0.790 |
| *(reference)* the counting rule itself, as an oracle | 0 | ~60 bits of source | — | — | — | — | 0 | **1.000** |
| *(reference)* best fitted single feature, from the track's own `baselines.json` | 0 | — | — | — | — | — | 24 | 0.648 |
| majority constant | 0 | 1 bit | — | — | — | — | 0 | 0.548 |

**Reading.** No matched neural baseline comes close. Nine architectures across
four families, two supervision modes, three learning rates and three seeds:
the best *protocol-selected* arm reaches **0.657** where the frozen program
reaches **0.9986**, and the best single seed anywhere in the sweep reaches 0.782.
The MLPs and transformers fit the 24 training examples perfectly (train accuracy
1.000) and land within a few points of the 0.548 majority off-distribution —
textbook memorisation of 12 strings. More data of the same lengths makes it
*worse*: at 461 episodes the CNN's held-out accuracy falls to 0.351, because the
extra episodes add only 15 new distinct strings and sharpen the memorisation.

Two honest qualifications. First, **the recurrent arm is unstable rather than
incapable, and finding that out changed the headline.** At lr 0.01 for 400 steps,
17 of 18 GRU runs collapsed to the constant predictor — they did not fit even the
24 training examples — and a first pass would have recorded "the family fails".
Re-run at lr 0.03 for 1,000 steps with the depth supervision, `gru_16` fits the
training set exactly on 2 of 2 seeds and becomes the **best matched arm in the
whole language study by validation accuracy** (0.888) at a held-out 0.657. That
is still 0.34 below the frozen program, but it is 0.05 above the best CNN, and it
cost 40× the training wall clock to get there. Second, the counting structural
hint reaches only
0.790 — but the counting *rule* is 1.000, so the shortfall is the linear readout
(balancedness is an equality test on a count difference, not a half-space), not
the feature. Hand the model the aggregation and the remaining gap is one
non-linearity wide; make it discover the aggregation from 24 examples and no arm
here gets near.

### 3.3 Cost

Latency, size and MAC counts for every architecture are in §6.1, measured for all
three arms back to back in one process. On this arm the typed program is **26×
slower per inference** than the CNN that is closest to it in quality (2.820 ms
against 0.110 ms) and **5.5× smaller on disk once both are compressed** (6,634 B
against 33,904 B). The program's `.pyz` runs in 23.9 MB of RSS under a
stdlib-only interpreter with a 91–108 ms cold start; the CNN needs 270 MB and
4.2–4.8 s just to import torch.

### 3.4 Transfer to the post-audit stream (both sides, neither trained there)

400 episodes of the current default lesson, lengths 10–22, majority 0.5150.

| method (all trained on the legacy stream) | accuracy |
|---|---|
| frozen TCN program | 0.4650 |
| best matched net (`cnn_32x3`, aux) | 0.5175 |
| counting structural hint | 0.5150 |
| majority constant / counting oracle | 0.5150 |
| true Dyck oracle | 1.0000 |

Everything that implements counting lands on the majority, because on this
stream counting *is* the majority (§2). Neither side transfers.

### 3.5 The post-audit stream, trained there — where the TCN has no number

Per FINDINGS §39 the language capability has never been measured on the
post-audit distribution, so this is one side only, reported so a later TCN
measurement has something to land against. Same holdout discipline, transposed:
train on lengths {10,12,14}, hold out {16,18,20,22} (**688 episodes**, majority
0.5189, counting oracle 0.5189, Dyck oracle 1.000).

| arm | params | held-out accuracy |
|---|---|---|
| matched budget, 24 episodes — `cnn_32x3`, aux, lr 0.01 | 9,123 | **0.732** |
| over-budget, 535 episodes — `cnn_32x3`, aux, lr 0.03 | 9,123 | **0.876** |
| selected on the seen-length validation split — `mlp_32`, aux | 22,785 | 0.704 |
| majority constant | 0 | 0.5189 |
| **frozen TCN program** | — | **not measured — no artifact exists for this stream** |

So on the harder, non-exploitable lesson a 9,123-parameter CNN gets to 0.876
from 535 episodes. Whether a typed program does better there is an open
question this track cannot answer.

---

## 4. Visual — a 32×32 screenshot parsed to an exact widget hierarchy

### 4.1 Matching conditions

The TCN artifact (`rung3_root.py`) runs three frozen modules at all 1,024 pixel
addresses: `same(a,b)` (S0, 256 programs), `corner(pos)` (S1′, 400), and
`rect(pos) → (x, y, w, h, own_key, parent_key)` (S2′, 25). The tree is then
resolved from those rows by `score_with_root`: a row is the root when
`parent_key == own_key`, otherwise its parent is the smallest parsed rectangle
containing the pixel at `(x−1, y)`.

| | TCN | CNN baseline |
|---|---|---|
| input | the 3,072 raw RGB bytes of `observations['pixels']` | identical |
| never seen as input | `probes['hierarchy']`, `probes['owner']`, seed, split, widget count | identical |
| output | per position: corner, and (x, y, w, h, own_key, parent_key) | per position: corner logit, w (33-way), h (33-way). `own_key`/`parent_key` are packed from the raw image by the **same declared, unsearched rule** S2′ uses |
| scoring | `rung3_root.score_with_root`, unmodified | the identical function, on the identical rows format |
| supervision | the `hierarchy` probe, subsampled to 120 corner positions per training screen | the same probe, **densely**: a corner label at all 1,024 positions and (w, h) at every widget corner — *more* than S1′ consumed |
| training | seeds 0–5 `split='train'` (6 screens) | the same 6 screens, plus over-budget arms at 48, 192 and 768 screens |
| validation | seeds 50–52 `split='validation'` | the same 3 screens; the corner threshold is selected on them and nothing else |
| held out | seeds 200–211 `split='test'` — 12 screens, **227 widgets** | identical |

### 4.2 What is recoverable before any model is trained

`visual_bounds.py`, using the repository's own `lookup_bound` method. A lookup
table over a context is an upper bound on every function of that context.

| context for the corner label | distinct keys, train → test | unseen keys at test | transfer accuracy | advantage over the 0.9815 majority |
|---|---|---|---|---|
| the 27 raw bytes of the 3×3 window | 2,001 → 3,837 | **81.2%** | 0.9824 | **+0.0009** |
| same-as-left / same-as-up / same-as-up-left | 7 → 7 | 0.0% | **1.0000** | **+0.0185** (the full oracle advantage) |
| the rule written out in plain Python | — | — | 227 TP, **0 FP, 0 FN** over 12,288 positions | — |

**This is the whole visual result in one table.** The information is in the
*equality relation*, not in the raw values: four fifths of the held-out raw pixel
contexts never occur in the 6 training screens, and a table over raw values
transfers at one tenth of one percent above the majority. The TCN scaffold has
equality as a searched primitive (S0 selects it from 256 candidates on those same
6 screens); a CNN over raw values must induce it.

### 4.3 The table

Quality on the same 12 held-out screens, 227 widgets. Trivial references:
`parent = root` gives link accuracy **0.1395** over the 215 non-root widgets
(**0.132** over all 227, the figure FINDINGS §33 quotes); "no corner anywhere" is
right at **0.9815** of the 12,288 positions; the empty parse gives 0 rectangles and 0
exact trees. Medians over 2 seeds; corner precision/recall and extent accuracy
are seed 0.

| method | train screens | params | serialized | learned content | batch-1 latency | peak RSS | **rects exact / 227** | **links / 227** | **trees / 12** | corner P/R | extent |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **TCN frozen parse** | **6** | **0 trainable** | 39,994,512 B JSON (`description_bits` 319,956,096); **184,259 B** gzipped; 627 static nodes | **21.3 bits** (log₂ of 256 × 400 × 25) | **18,375 ms / screen** | 405 MB | **227 (1.000)** | **227 (1.000)** | **12 / 12** | 1.00 / 1.00 | 1.00 |
| CNN `cnn_w16_d5` | 6 | 10,867 | 48,435 B | 347,744 bits | 12.2 ms (min 3.1) | 270 MB | 5 (0.022) | 3 (0.015) | 0 | 0.11 / 0.19 | 0.05 |
| CNN `cnn_w32_d5` | 6 | 40,099 | 165,363 B | 1,283,168 bits | 61.3 ms (min 31.1) | 270 MB | 3 (0.013) | 3 (0.013) | 0 | 0.06 / 0.15 | 0.06 |
| CNN `cnn_w64_d5` | 6 | 153,859 | 620,403 B | 4,923,488 bits | 75.5 ms (min 34.0) | 270 MB | 4 (0.018) | 3 (0.015) | 0 | 0.11 / 0.12 | 0.19 |
| the same, 48 screens (8×) | 48 | 10,867–153,859 | | | | | 15–24 (0.068–0.104) | 10–14 (0.044–0.062) | 0 | up to 0.78 / 0.97 | 0.17 |
| the same, 192 screens (32×) | 192 | 40,099 | | | | | 141 (0.619) | 62 (0.275) | 0 | 0.47 / 0.98 | 0.70 |
| the same, 192 screens | 192 | 153,859 | | | | | **151 (0.665)** | **106 (0.469)** | **0** | 0.93 / 0.99 | 0.68 |
| the same, 768 screens (128×) | 768 | 40,099 | | | | | **211 (0.926)** | **174 (0.767)** | **0** | 0.84 / 1.00 | 0.94 |
| *(declared structural hint)* colour-bijection augmentation, `cnn_w32_d5` | 6 | 40,099 | | | | | 18 (0.081) | 14 (0.062) | 0 | 0.17 / 0.50 | 0.16 |
| *(same hint)* | 48 | 40,099 | | | | | 49 (0.214) | 19 (0.084) | 0 | 0.22 / 0.82 | 0.26 |
| `parent = root` | — | 0 | — | — | — | — | 0 | **30 of the 215 non-root widgets (0.1395); 0.132 of all 227**, matching FINDINGS §33 | 0 | — | — |
| empty parse | — | 0 | — | — | — | — | 0 | 0 | 0 | — | — |

**Training budget.** The TCN side consumed **6 training screens and 3 validation
screens, 681 program evaluations in 55.3 s** (S0 4.1 s over 256, S1′ 19.8 s over
400, S2′ 31.4 s over 25) and **zero optimizer steps**. The CNN rows consumed
2,500–4,000 Adam steps at batch 6, 73–754 s of wall clock each.

**Reading.** At the matched budget the CNN is not close: it fits the 6 training
screens (train link accuracy 0.95–1.00) and recovers **3 of 227** parent links on
held-out screens, against the program's 227. The scaling curve is the honest part
of the answer — exact rectangles go 3 → 24 → 151 → 211 as the training set goes
6 → 48 → 192 → 768 screens — but **no arm at any budget got a single one of the
12 trees exactly right**, because a tree is exact only when every rectangle on
the screen and every link is exact, and the best arm still misses ~7% of
rectangles. The typed program reaches 12/12 from 6 screens.

The colour-augmentation hint moves corner *recall* from 0.15 to 0.50 at 6 screens
and 0.82 at 48, confirming the bound in §4.2 — colour invariance is the missing
inductive bias — but it does not close the gap on its own, because the extent
heads still have to learn a 32-step run length from a convolution.

One generator note, reported rather than worked around: at the `FLAT`
configuration some `gui` seeds raise `ValueError: y1 must be greater than or
equal to y0` from `generators/gui/render.py:164` (a widget laid out with
non-positive height). Nothing under `generators/` was modified; `screen_pool`
skips those seeds and records them in `out/visual*.json` as
`generator_seeds_refused` so the training budget is not silently overstated.

---

## 5. Computer — a policy acting in a live OS

### 5.1 Matching conditions

The task (FINDINGS §23): `/home/agent/task.txt` holds `<name> = <digit>` and the
objective is met when the file contains `str(digit+1)`. At tick 0 the terminal
shows only the setup write's JSON, which carries the file's *length*, not its
content — so the policy must read, then write a value computed from what it saw.
Max return 2.

| | TCN frozen agent | every neural baseline |
|---|---|---|
| input | `observations['terminal']` — `(length: u32, 4096 × byte)` — and the previous action's 3-float one-hot | the same two ports. The baseline reads the first **64** bytes; `assert_padding_is_constant` checks on all 45 train and test decision points that every byte at or beyond 64 is zero (longest terminal: 45 bytes) and aborts otherwise, so the truncation discards nothing and is a restriction, never an enlargement |
| never seen as input | `probes`, `latent_states`, the document string, the digit, the name, the episode index, the split | identical |
| supervision | `task.examples_from`'s `reference` verb index and `reference_byte` — the two targets `search_run.py` enumerated against | byte for byte the same two targets |
| training | 5 documents × 3 ticks = 15 decision points, of which 5 are write decisions | identical |
| held out | `task.TEST_DOCUMENTS` — 10 documents, names and digits never seen, run at `index=1000+i, split='test'` as `closed_loop.run_agent` does | identical |
| quality | the shipped reward for the action the policy chose, executed by the live kernel | identical |
| selection | none — enumeration exhausted | training fit only; the held-out closed loop never chooses an arm |

### 5.2 The table

**Program-only cost.** `Agent.act` runs the frozen program twice per step (choose,
then commit) for 46 operator applications; the kernel round trip is measured
separately at **866.3 ms** median per step and is charged to neither method. The
neural policy figure is one forward pass, which is the whole of its decision.

| method | params | serialized | learned content | policy latency / step | peak RSS | training budget | **solved / 10** | mean return / 2 |
|---|---|---|---|---|---|---|---|---|
| **TCN frozen agent** | **0 trainable** | 4,246,492 B JSON (`description_bits` 33,971,936); **18,927 B** gzipped | **21.7 bits** (log₂ of 7,480 × 448) | **13.35 ms** median (93.6 ms first call), 46 operator applications | 249.6 MB | 5 documents / 15 decision points, **7,928 programs enumerated in 194.7 s**, 0 optimizer steps | **10 / 10** | **2.00** |
| matched net, selected on training fit — `gru_h16`, regression byte head | 3,373 | 16,777 B | 107,936 bits | 9.25 ms | 270 MB (torch) | 15 decision points × 800 Adam steps | **0 / 10** | 0.00 |
| `gru_h32`, regression head | 6,221 | 30,177 B | 199,072 bits | 8.68 ms | 270 MB | same | 0–1 / 10 | 0.10 |
| **best matched net run live** — `gru_h16`, 256-way classifier head *(selected on held-out supervised accuracy — a protocol the TCN did not have)* | 7,708 | 34,057 B | 246,656 bits | 4.10 ms | 270 MB | same | **4–5 / 10** | 0.90 |
| `cnn_w32`, classifier head, same protocol | 14,508 | 61,257 B | 464,256 bits | 0.59 ms | 270 MB | same | 2–4 / 10 | 0.60 |
| *(declared structural hint)* `mlp_relative_hint` — handed the computed address `length − k` | 4,333 | 20,617 B | 138,656 bits | 0.69 ms | 270 MB | same | 3–6 / 10 | 0.90 |
| **ceiling for any 256-way classifier on this held-out set** | — | — | — | — | — | — | **5 / 10** | 1.00 |
| uniform random verb and digit | 0 | — | — | — | — | 0 | 1 / 10 | 0.60 |
| always write "5" | 0 | — | — | — | — | 0 | 0 / 10 | 0.30 |
| read, then write the modal digit "5" | 0 | — | — | — | — | 0 | 1 / 10 | 0.20 |
| always wait | 0 | — | — | — | — | 0 | 0 / 10 | 0.00 |

The four model-free references reproduce `closed_loop.json`'s recorded values
exactly (0.30 / 0.20 / 0.60), which independently confirms §23's baseline row.

The `policy latency / step` column is measured **inside the live rollout**, one
`perf_counter_ns` around the forward pass that chose each action, so it is the
cost as actually incurred. §6.1's dedicated bench, 100 warm calls with the same
protocol as `tcn/runtime.py:benchmark`, gives 0.39–9.85 ms for the same
architectures; the two agree to within the host's noise.

### 5.3 Where the neural policy fails, precisely

It is not the perception and not the plan. **Every** matched arm reaches verb
accuracy 1.00 on training and 0.90–1.00 on the held-out documents: read first,
then write, then wait. What none of them gets is the *byte*.

* The **regression** head must learn `written = observed + 1` as an affine map
  from 5 examples. It gets held-out byte accuracy **0.00–0.05**.
* The **classifier** head can only emit a byte it saw as a training label.
  Training targets are {'1','3','4','6','7'}; 5 of the 10 held-out targets fall
  outside that set, so **0.50 is a hard ceiling**, computed in
  `computer_live_extra.py` and printed with the run. The classifier *attains* the
  ceiling — 0.50 held-out byte accuracy, 4–5 of 10 solved live.

So the neural side's best achievable score on this task, with matched inputs, is
5/10 and it reaches it. The TCN's `shift = add(value, 1)` is one node chosen from
a pool that also contained `identity`; that single discrete choice is the whole
difference between 5/10 and 10/10, and it is not a quantity of parameters.

The structural hint confirms the diagnosis is about arithmetic rather than
addressing: handing the model the computed address `length − k` — the thing
`pos` had to *search* for, against 15 constant alternatives — moves it to 3–6 of
10, not to 10 of 10.

---

## 6. Verdicts

### 6.1 Cost, all three arms, measured back to back

Both sides on this host, batch one, one thread. **The host was at load 32–60
throughout**, so each wall clock is quoted with its minimum and beside a
load-independent count — operator applications for the typed program (the unit
`research/inference-cost/RESULTS.md` uses), multiply-accumulates for the net.
The neural rows are torch; `research/baselines/RESULTS.md` §5 measured the same
weights 10–20× faster as plain numpy, so every neural latency here is an **upper
bound** on its deployed cost — the conservative direction for the question being
asked.

| arm | method | params | MACs or operator applications | warm p50 | warm min | cold first call | serialized (f32 / JSON) | gzipped |
|---|---|---|---|---|---|---|---|---|
| language | **TCN frozen program** | **0** | 148 `execution_cost`; 164 real operator applications (inherited, `inference-cost` §2) | **2.820 ms** | 2.759 ms | 2.951 ms | 505,444 B JSON | **6,634 B** |
| language | `cnn_32x3` (best matched) | 9,123 | 888,865 MACs | **0.110 ms** | 0.107 ms | 0.364 ms | 36,492 B | 33,904 B |
| language | `cnn_16x2` | 3,267 | 149,521 | 0.302 ms | 0.054 ms | 0.341 ms | 13,068 B | 12,265 B |
| language | `mlp_32` | 18,561 | 17,472 | 0.307 ms | 0.264 ms | 0.534 ms | 74,244 B | 69,108 B |
| language | `transformer_d16` | 8,403 | 788,497 | 18.36 ms | 9.31 ms | 136.9 ms | 33,612 B | 30,886 B |
| language | `gru_16` (best matched on validation, with the aux head: 3,330 params) | 3,313 | 147,472 | 31.12 ms | 5.11 ms | 28.84 ms | 13,252 B | 12,395 B |
| visual | **TCN frozen parse** | **0** | 1,540,098 `execution_cost`; 64,346 real operator applications at 961 positions (inherited) | **18,374.8 ms** | 17,285.7 ms | 18,481.8 ms | 39,994,512 B JSON | **184,259 B** |
| visual | `cnn_w16_d5` | 10,867 | 10,977,280 MACs | 12.17 ms | **3.09 ms** | 16.68 ms | 43,468 B | 39,812 B |
| visual | `cnn_w32_d5` (reaches 0.93 of rectangles at 768 screens) | 40,099 | 40,828,928 | 61.25 ms | **31.09 ms** | 77.01 ms | 160,396 B | 148,450 B |
| visual | `cnn_w64_d5` | 153,859 | 157,155,328 | 75.50 ms | 34.02 ms | 43.05 ms | 615,436 B | 563,578 B |
| computer | **TCN frozen agent** (per step, two program passes) | **0** | 23 `execution_cost`; 46 real operator applications (inherited) | **13.350 ms** | — | 93.6 ms | 4,246,492 B JSON | **18,927 B** |
| computer | `gru_h16` classifier (best matched run live) | 7,708 | 77,881 MACs | 9.85 ms | 4.31 ms | 8.25 ms | 30,832 B | 28,499 B |
| computer | `gru_h16` regression | 3,373 | 73,801 | 4.37 ms | 2.36 ms | 11.90 ms | 13,492 B | 12,616 B |
| computer | `cnn_w32` classifier | 14,508 | 254,057 | 0.43 ms | 0.37 ms | 2.96 ms | 58,032 B | 53,904 B |
| computer | `mlp_relative_hint` *(hint)* | 4,333 | 2,217 | 0.39 ms | 0.36 ms | 3.75 ms | 17,332 B | 16,345 B |
| computer | *(neither method)* the live kernel round trip | — | — | 866.3 ms | — | — | — | — |

Out-of-process cold start, fresh interpreter, one inference:

| method | interpreter | import + build | first inference | peak RSS |
|---|---|---|---|---|
| every neural row | repo `.venv` python, `import torch` | **1.7–7.2 s** | 0.5–1,066 ms | **270 MB** |
| `language.pyz` *(inherited, `inference-cost/out/pyz_language.json`)* | `/usr/bin/python3 -I`, stdlib only | 91–108 ms | 14.6 ms | **23.9 MB** |
| `computer.pyz` *(inherited)* | same | 410–479 ms | 210 ms | 60.4 MB |
| `visual.pyz` *(inherited)* | same | 5.2–5.4 s | 66.6 s | 377.5 MB |

The warm medians on this host are noisy — `cnn_w64_d5` medians above `cnn_w32_d5`
while having 3.8× the MACs — which is why the MAC and operator-application
columns are there and why minima are quoted beside every median.

**Where the cost actually lands, and it is not close on either side.**

* **Representation.** The typed programs ship 0 trainable parameters and 21–29
  bits of learned content, but **39,994,512 / 4,246,492 / 505,444 bytes of JSON**
  (visual / computer / language). The matched nets ship 3,267–153,859 parameters
  = **13,068–615,436 bytes** of float32. Gzipped, the typed artifacts come back
  to 184,259 / 18,927 / 6,634 bytes. So on the language and computer arms the
  typed program is *smaller* once compressed and far larger as shipped; on the
  visual arm it is larger either way. This is FINDINGS §36's serialization
  finding reproduced from the other side: `description_bits` measures JSON
  verbosity, and the 21.3 bits of learned visual content occupy 40 MB.
* **Execution.** The visual parse costs **18,375 ms per screen** (min 17,286)
  against **61 ms median, 31 ms minimum** for the CNN that reaches 93% of its
  rectangles — **300× on medians, 556× on minima**. That is the one place a
  neural baseline wins a cost axis outright. On the language arm the program is
  2.82 ms against 0.11 ms for the best matched CNN — **26×**. On the computer arm
  the frozen program is 13.35 ms per step against 0.39–9.85 ms for the nets — the
  same order, and both are dwarfed by the 866 ms kernel round trip that neither
  is charged for. For reference, `research/inference-cost/RESULTS.md` §2 measures
  a plain-Python implementation of the same visual rule at **0.17 ms**, so the
  CNN is itself 180× slower than the rule written out; the program is 108,000×.
* **Memory and dependencies.** Every neural row costs **~270 MB of RSS** and
  1.7–7.2 s just to `import torch` in a fresh process. The typed artifacts run
  under `/usr/bin/python3 -I` with no torch, no numpy and no repository:
  24–60 MB and 91–479 ms cold for the language and computer `.pyz`, 377 MB and
  5.4 s for the visual one (inherited from
  `research/inference-cost/out/pyz_*.json`, and labelled as inherited). On
  deployment footprint the typed artifact wins two of three arms decisively.
* **Training.** The typed side consumed 55,871 / 7,928 / 681 program
  evaluations in 362 / 195 / 55 s and **zero optimizer steps**. The neural side
  consumed 400–800 Adam steps in 2–5 s (language, computer) and 2,500–4,000 steps
  in 70–750 s (visual). On the two small arms the neural side is the cheaper to
  fit by an order of magnitude; on the visual arm the typed search is cheaper.

### 6.2 The verdict per artifact

| artifact | which method costs less, on what axis, at the quality each actually reaches |
|---|---|
| **Language** | **Quality is not comparable, so no cost trade exists.** 0.9986 against **0.657** for the best protocol-selected net (a 3,330-parameter GRU) and 0.604 for the best CNN, on 724 held-out episodes with a 0.548 majority. The CNN is cheaper to fit (4.5 s against 362 s of search) and 26× cheaper per inference; the GRU that scores best is *more* expensive than the program on both (174–233 s to fit, 31.1 ms per inference against 2.82 ms). Serialized, the program is smaller than either once gzipped (6,634 B against 12,395–33,904 B). **The typed program wins on quality decisively and on size; the neural side wins training wall-clock only if you take the CNN, which is also the weaker of the two.** |
| **Visual** | **Split, and this is the clearest trade in the track.** The typed program is exact — 227/227 rectangles, 227/227 links, 12/12 trees from 6 screens — where the best CNN at 128× the data reaches 211/227 and **0/12 trees**. But the program costs **18,375 ms per screen** (min 17,286) against 61 ms median / 31 ms min, and ships 40 MB (184 KB gzipped) against 160 KB of weights. **The typed program wins on quality and sample efficiency by a wide margin; the neural baseline wins on execution latency by 300× on medians and 556× on minima, and on shipped size.** If you need the tree, only one method produces it; if you need it fast, only one method is fast. |
| **Computer** | **The typed program wins on quality and ties on cost.** 10/10 against a hard ceiling of 5/10 for any 256-way classifier over matched inputs, which the best net attains. Program-only latency 13.35 ms against 0.39–9.85 ms — the same order, and 1.5% of the 866 ms kernel step either way. Serialized: 18,927 B gzipped against 13,492–30,832 B of weights — a tie. Training: 195 s of enumeration against 3 s of Adam. **Quality: typed, decisively. Latency and size: a wash. Training wall-clock: neural.** |

### 6.3 What this settles about FINDINGS §36

§36's unsupported sentence can now be replaced. The honest replacement is that
**"cheaper than a model" was the wrong question for these three artifacts**,
because no matched baseline reaches comparable quality on any of them; what the
measurement does settle is each axis separately:

* **Execution cost: refuted, again, and worse than on the mixed fixture.** The
  visual parse is 18.4 s per screen where a 40,099-parameter CNN is 31–61 ms
  and a plain-Python implementation of the same rule is 0.17 ms
  (`research/inference-cost/RESULTS.md` §2). §36 already identified the cause —
  the typed value layer, 97.7% of execution — and this track confirms it is not
  a matched-baseline artifact: a model carrying 1.28 million bits of weights
  against the program's 21.3 bits of learned content runs 300–556× faster.
* **Deployment footprint: survives.** 24–60 MB of RSS and 91–479 ms of cold
  start for `language.pyz` and `computer.pyz` under a stdlib-only interpreter,
  against ~270 MB and 1.7–7.2 s to import torch. On the two small
  artifacts this is a real, large win that no amount of interpreter optimisation
  on the neural side removes.
* **Sample efficiency and exactness: survives, and is the strongest result
  here.** From 6 screens and 681 program evaluations the typed parse is exact on
  227 held-out widgets; the CNN needs 128× the screens to get to 93% and never
  gets a tree. From 24 episodes the typed program is 723/724; no net is above
  0.61. From 5 documents the typed policy is 10/10; the best matched net's
  ceiling is 5/10.
* **And the cause is identified, the same one `research/baselines/RESULTS.md` §7
  found on the mixed fixture:** the operator library contains the answer. `eq`
  over typed values is what makes the visual parse exact — §4.2 measures that
  the equality signature transfers at 1.000 where raw pixel values transfer at
  +0.0009 — and `add(value, 1)` chosen over `identity` is the whole difference
  between 5/10 and 10/10 on the computer arm. That is a real and reproducible
  advantage, and it is an advantage of the hypothesis class, not of the search.

---

## 7. Tuning disclosure, and what was cut

**TCN side — tuning it received (all of it pre-existing, none of it by me).** The
scaffold shape, the operator library, the staging decomposition, the offset and
address pools, the tolerance, the position sets and the region labels are all
authored in each artifact's own track. In particular the operator library
contains the answer in every arm: `eq` (equality of typed values) for the visual
parse, `eq` over the 0–255 byte alphabet plus an accumulator chain for language,
and `sub` / `add` over a length field for the computer address. I changed
nothing; each artifact was rebuilt from the selections its own track froze and
re-scored on that track's own splits.

**Baseline side — tuning I gave it.** Disclosed per arm in §3–§5 and in the
scripts' docstrings. Across all three arms: Adam, grad-clip 5.0 (matching
`tcn/training.py`), tanh or ReLU activations, no normalisation layers, no
learning-rate schedule, no weight decay, no ensembling, no architecture search
beyond the small ladders listed. Model selection is on training fit or on the
artifact's own validation split, never on the held-out set. Where a threshold was
selected (the visual arm's corner logit) it was selected on the 3 validation
screens only, and the un-thresholded number is recorded beside it in the JSON.

**What was cut, and why.** The host was at load 32–60 throughout, shared with a
dozen unrelated jobs, and a live kernel step for the computer arm costs about a
second. The sweeps were trimmed as follows. **These are the achieved
configurations, not the requested ones.**

| arm | achieved | what a fuller run would have added |
|---|---|---|
| language, matched | 9 architectures × 2 supervision modes × 3 seeds, 400 Adam steps, lr 0.01; plus `gru_16` × 2 modes × 2 seeds at lr 0.03 for 1,000 steps (the lr 0.1 cell was cut for wall clock) | the same lr/step budget for every architecture. The recurrent arm's second pass moved the best matched score from 0.604 to 0.657, so this is the one cut that demonstrably changed a headline |
| language, over-budget | 3 architectures × 2 modes × 2 seeds on 461 episodes | little: the over-budget arm was *worse*, and only 15 new distinct strings exist |
| language, post-audit stream | 4 architectures × 2 modes × 2 seeds × lr ∈ {0.01, 0.03} | a TCN measurement on that stream, which does not exist |
| visual | 3 widths × 2 seeds at 6, 48, 192 screens (2,500 steps) and 768 screens (4,000 steps) | more widths and a longer schedule; the 6→768 curve is the load-bearing measurement and it is complete |
| computer | 4 architectures × 2 byte heads × 2 seeds offline (800 steps); the live kernel spent only on the arms selected by training fit, plus both classifier heads and the hint arm | live rollouts for the arms whose held-out byte accuracy is 0.10 or below, which cannot beat the 5/10 ceiling |
| latency | one back-to-back pass on a loaded host | a quiet host; minimum and p95 are recorded in `out/latency.json` beside every median, and MAC counts are given so a reader on another machine can reprice every row |

**Three things a fuller run should do, in priority order.** (1) Give every
language architecture the lr 0.03 / 1,000-step budget the recurrent arm needed —
that budget moved the best matched score from 0.604 to 0.657 and no other
architecture was re-run at it. (2) Re-time everything on a quiet host.
(3) Measure the TCN language capability on the post-audit stream, where §3.5
leaves a neural number with nothing beside it.

---

## 8. Files and how to reproduce

Every number above comes from `out/*.json`, and `summarise.py` prints the rows
this document quotes so they can be regenerated and diffed rather than trusted.

| file | what it does |
|---|---|
| `common.py` | the shared harness: `bench` (mirrors `tcn/runtime.py:benchmark`, warm-up separated), `torch_size_report`, `program_size_report`, `subprocess_cold_start`, `majority_reference` |
| `langdata.py` | the language episodes on the stream the §19 artifact was searched on (`hardening="none"`), and the current default stream as a second held-out set |
| `language_baseline.py` | the language model ladder, both supervision modes, both budgets, the counting structural-hint arm |
| `visual_baseline.py` | the screenshot-parse CNN ladder, scored through `rung3_root.score_with_root` unmodified |
| `visual_bounds.py` | what is recoverable from the raw 3×3 window and from the equality signature, before any model is trained |
| `visual_augmented.py` | the colour-bijection structural-hint arm |
| `computer_baseline.py` | the live-OS policy ladder, its offline fit and its closed loop, with the kernel round trip separated |
| `tcn_reference.py` | all three TCN artifacts rebuilt from their frozen selections, re-scored and re-timed here |
| `latency_pass.py` | every architecture's batch-one cost and serialized size, measured back to back |
| `verify.py` | recomputes each headline from the per-row records; non-zero exit on any mismatch |
| `summarise.py` | prints the tables above from `out/*.json` |

```bash
# from the repository root, with the repo .venv
.venv/bin/python research/neural-baselines/tcn_reference.py language
.venv/bin/python research/neural-baselines/tcn_reference.py visual      # ~4 min
.venv/bin/python research/neural-baselines/tcn_reference.py computer    # live OS
.venv/bin/python research/neural-baselines/visual_bounds.py
.venv/bin/python research/neural-baselines/language_baseline.py --seeds 3 --steps 400
.venv/bin/python research/neural-baselines/visual_baseline.py --seeds 2 --steps 2500 \
    --models cnn_w16_d5 cnn_w32_d5 cnn_w64_d5 \
    --budgets matched_6_screens over_budget_48_screens over_budget_192_screens
.venv/bin/python research/neural-baselines/visual_augmented.py
.venv/bin/python research/neural-baselines/computer_baseline.py --seeds 2 --steps 800
.venv/bin/python research/neural-baselines/latency_pass.py
.venv/bin/python research/neural-baselines/verify.py
.venv/bin/python research/neural-baselines/summarise.py
```

`out/computer_trivial_cache.json` caches the four model-free closed-loop
references so a re-run does not spend 120 live kernel steps re-measuring them;
delete it to re-measure.
