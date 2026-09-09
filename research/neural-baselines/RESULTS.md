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
execution and training cost does each method require?* Two of the three arms
come out in the typed program's favour and one does not, and both kinds are
reported the same way.

Everything below was produced in this worktree on 2026-09-09 with the repository
`.venv` (Python 3.13.15, torch 2.14.0+cpu, `torch.set_num_threads(1)`), Linux,
20 cores. **The host was heavily loaded throughout** — a dozen other jobs, load
average 32–39 — so every wall clock is a median beside a load-independent count
where one exists, and the latency caveat of `research/baselines/RESULTS.md` §5
applies unchanged. Nothing under `tcn/` or `generators/` was modified and no
other track's files were written.

---

## 0. Verdict, one line per artifact

*(filled in at §6)*

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

## 2. A correction and a caveat that had to be settled first

**The language artifact's headline is 0.9986, not 1.000.** FINDINGS §19 and
`STATUS.md` §1 row 9 both say "**1.000** on 724 held-out episodes." The track's
own `research/language-capability/final_eval.json` records
`heldout_unseen_lengths.accuracy = 0.9986187845303868`, with `per_length`
`{8: 1.0, 10: 1.0, 12: 1.0, 14: 1.0, 16: 0.5}` — the frozen program's accumulator
is 16 steps deep and it gets one of the two length-16 episodes wrong. Rebuilding
the program here from `stage_b.json` and re-scoring reproduces
**0.9986187845303868** exactly. 1.000 holds on the 242-episode set at lengths
8–14 that `run_stage_b.py` wrote; it does not hold on the 724-episode set the
headline cites. This does not change any conclusion below — 723/724 against a
0.548 majority is still a decisive result — but the record should say 0.9986.

**The language lesson was re-drawn after the artifact was frozen, and the
artifact does not survive the re-draw.** FINDINGS §24 re-drew 14 exploitable
lessons, `context_free_language` among them, and changed the default stream. On
the current default the lesson emits string lengths {10…22} and the §19 splits
(train on {2,4,6}, hold out {8…16}) **do not exist in it at all**. The pre-audit
stream is preserved exactly as `hardening="none"`, which is what every §19 number
was measured on, so that is what `langdata.py` uses and what both methods face in
§3. The current stream is then reported as a *second, harder* held-out set:

| stream | frozen TCN program | counting oracle | true Dyck oracle | majority |
|---|---|---|---|---|
| legacy (`hardening="none"`), 724 episodes | **0.9986** | 1.0000 | 1.0000 | 0.5483 |
| current default, 400 episodes | **0.4650** | **0.5150** | **1.0000** | 0.5150 |

The counting oracle on the current stream is *exactly* the majority constant, and
the true Dyck oracle is 1.000. So the hardening removed the counting shortcut and
left the task answerable — and the frozen program, which §19 itself says
"demonstrates counting and agreement, **not recursion**", drops to just below
chance. The artifact fails on the current lesson for precisely the reason its own
track predicted. Both methods are scored on both streams below.

---

## 3. Language — grammaticality of a bracket string from raw prompt bytes

*(filled in)*

---

## 4. Visual — a 32×32 screenshot parsed to an exact widget hierarchy

*(filled in)*

---

## 5. Computer — a policy acting in a live OS

*(filled in)*

---

## 6. Verdicts

*(filled in)*

---

## 7. Tuning disclosure, and what was cut

*(filled in)*

---

## 8. Files and how to reproduce

*(filled in)*
