# Track 7 — What the literature actually establishes about DLGNs

> **Correction (2026-09-08, after this report was written).** The project lead
> reports the Mario demonstration was posted to Twitter/X on 2026-09-07. Every
> search line below covered indexed sources — papers, GitHub, blogs, author
> publication lists — none of which index a day-old social media post. Read the
> null finding in section 1 as "not present in indexed sources as of this
> search", not as "does not exist". The conflation candidates listed there are
> retained as context, not as an explanation. The rest of this report, which
> concerns DLGN capabilities, scaling limits and prior art, is unaffected.

**Method.** Web search and direct source retrieval only, 2026-09-08. No experiments.
Every claim below carries a URL. Evidence is tagged:

- **[PR]** peer-reviewed / accepted at a refereed venue
- **[WS]** refereed workshop paper (lighter review)
- **[PP]** arXiv preprint, no confirmed venue
- **[BLOG]** blog, forum, video, or vendor page
- **[UNVERIFIED]** asserted by a secondary source; primary not retrieved

Where a number could not be read from the primary source, it is marked as such
rather than repeated.

---

## 1. The Mario claim

### Verdict

**No differentiable logic gate network result on Super Mario Bros exists in the
public literature, and none exists as a demo, talk, blog post, or GitHub project
that I could locate.** The claim does not survive contact with the sources.

This is a null finding, not a "could not confirm." I searched from six
independent directions and every one came back empty:

1. Petersen's own publication list — the origin of the DLGN line — contains
   **no** reinforcement learning, control, or game-playing work of any kind, and
   no mention of Mario.
   [petersen.ai](https://petersen.ai/) (retrieved 2026-09-08).
2. The `difflogic` reference library README supports Adult, Breast Cancer,
   MNIST, MNIST-20x20, and CIFAR-10 only. No RL, no control, no games, no Mario.
   [github.com/Felix-Petersen/difflogic](https://github.com/Felix-Petersen/difflogic)
   (retrieved 2026-09-08).
3. The NeurIPS 2024 Oral (`Convolutional Differentiable Logic Gate Networks`)
   and its 5-minute oral video are entirely image classification. No game demo.
   [neurips.cc/virtual/2024/oral/97997](https://neurips.cc/virtual/2024/oral/97997)
4. The nearest real logic-circuit control paper — Differentiable Weightless
   Controllers, ICML 2026 — explicitly evaluates on **MuJoCo only** and
   mentions no video game anywhere in the paper.
   [arXiv 2512.01467](https://arxiv.org/abs/2512.01467)
5. The main Hacker News discussion of DLGNs (a comment subtree from
   2025-05-28 under "Compiling a neural net to C for a speedup", 296 points)
   contains **no reference to games, Mario, RL, or control**.
   [news.ycombinator.com/item?id=44118846](https://news.ycombinator.com/item?id=44118846)
6. Repeated targeted searches for "logic gate network" + Mario / NES / Atari /
   game / agent returned only unrelated neuroevolution and DQN Mario projects.

### The four things it is most likely a conflation with

Each of these is real, and each is one associative step from "DLGNs beating
Super Mario Bros."

**(a) Logic gates built *inside* Super Mario Maker.** Ceave Gaming, "Logic
Gates, Computer Logic and Calculators in Super Mario Maker!", published
2016-12-03; and Helgefan's "Cluttered-Chaos Calculator", a working digital
circuit in Mario Maker using Shelmets as bits.
[youtube.com/watch?v=jx59oHXdXBU](https://www.youtube.com/watch?v=jx59oHXdXBU) **[BLOG]**
This is *logic gates + Mario* with **zero learning**. The gates are hand-built
level geometry. If the recollection is visual — circuits and Mario on screen at
the same time — this is by far the most likely source.

**(b) A non-standard hardware network that genuinely does play Super Mario
Bros: diffractive *optical* networks.** Qiu, Xiao, Huang, Miroshnichenko,
Zhang, Liu, Yu, "Decision-making and control with diffractive optical networks",
*Advanced Photonics Nexus* 3(4), 2024 (submitted 2022-12-21).
[arXiv 2212.11278](https://arxiv.org/abs/2212.11278) **[PR]**
They train diffractive optical networks with deep RL and validate on
Tic-Tac-Toe, **Super Mario Bros.**, and Car Racing; only Tic-Tac-Toe was
demonstrated on physical hardware (spatial light modulator), Mario is
simulation. This is the strongest candidate for "an exotic
non-von-Neumann-substrate network beat Super Mario Bros" — but it is an
*optical* network, not a logic gate network, and it is trained by standard deep
RL, not by discrete program crystallization.

**(c) Differentiable Logic Cellular Automata — DLGNs learning "the Game of
Life".** Miotti, Niklasson, Randazzo, Mordvintsev (Google, Paradigms of
Intelligence), ALIFE 2025.
[arXiv 2506.04912](https://arxiv.org/abs/2506.04912) ·
[project page](https://google-research.github.io/self-organising-systems/difflogic-ca/) **[PR]**
DLGN gates as cellular automata update rules. Achievements: fully learning
Conway's Game of Life rules, noise/damage-resilient checkerboards, growing a
lizard shape, multi-colour pattern generation. This is a real, high-profile
DLGN result whose headline contains the word **"Game"**. "DLGNs beating the
Game of Life" → "DLGNs beating a game" is a very short slip.

**(d) MarI/O and the neuroevolution Mario genre.** SethBling's MarI/O (June
2015) evolved a NEAT network that cleared a Super Mario World level in 34
generations, with a famous on-screen visualisation of a **sparse network of
nodes and wires** that looks like a circuit diagram. The final network is
"less than a dozen neurons."
[hackaday.com/2015/06/14/neural-networks-and-mario/](https://hackaday.com/2015/06/14/neural-networks-and-mario/) **[BLOG]**
A tiny, sparse, discrete-looking graph beating Mario, widely seen. Visually
this is what people remember when they think "a circuit learned to play Mario."

### Bonus: a genuine acronym collision

**"DLGN" also means Deep Linearly Gated Network** — an unrelated
interpretability architecture (gating network of ReLUs + weight network of
gated linear units, analysed via the Neural Path Kernel), from
Lakshminarayanan et al.
[arXiv 2203.16455](https://arxiv.org/pdf/2203.16455),
[arXiv 2110.03403](https://arxiv.org/pdf/2110.03403) **[PP/PR]**
If the claim arrived as an acronym rather than a full name, this is a second
possible source of confusion. Neither DLGN sense has a Mario result.

### What the nearest real thing actually is

If the motivation is "a discrete learned circuit can drive a nontrivial
sequential control task," that claim **is** supported — by MuJoCo, not Mario:

> **Differentiable Weightless Controllers: Learning Logic Circuits for
> Continuous Control.** Fabian Kresse, Christoph H. Lampert. **ICML 2026**.
> [arXiv 2512.01467](https://arxiv.org/abs/2512.01467) **[PR]**

Reported SAC returns (higher is better), DWC vs full-precision and quantized
NN policies:

| Env | FP baseline | Quantized | DWC |
|---|---|---|---|
| Ant | 5.598k | 4.717k | **5.677k** |
| HalfCheetah | 11.529k | 10.465k | 7.549k |
| Hopper | 2.797k | 1.931k | **3.120k** |
| Humanoid | 6.186k | 5.954k | 6.141k |
| Walker2d | 5.044k | 4.656k | 5.025k |

On an Artix-7 XC7A15T FPGA at 100 MHz: ~0.8–3.2k LUTs, 2–3 clock cycles of
latency, ~10^8 actions/s, 1.02–2.28 nJ per action, ~0.1–0.2 W.

Three things about this paper matter more to TCN than the returns do:

- **HalfCheetah is a clear capacity failure** (7.5k vs 11.5k), which the
  authors attribute to circuit capacity limits. Discrete circuits are not
  uniformly competitive; they fall over on the task that most rewards fine
  continuous control.
- Inputs are **thermometer-encoded** from normalised, clipped observations, and
  outputs come from a **popcount → SRAM lookup**. The learned circuit sits
  between two hand-designed non-learned representation layers. TCN's
  constitutional rule that no domain gets a primitive unavailable to others is
  *stricter* than what the state of the art actually does.
- Stated limitation, quoted by the authors: training cost "significantly
  exceeds the cost at deployment, because of the relaxations required to allow
  for gradient-based training," and training is therefore "only feasible in
  simulated environments, not interactively on-device."

There is also exactly one prior DLGN-for-RL paper, and it is a **workshop**
paper, not a conference result:

> **Efficient Reinforcement Learning Agents with Differentiable Logic Gate
> Networks.** Felix Petersen, Christian Borgelt, Stefano Ermon. CoRL 2024
> Workshop on Differentiable Optimization (DiffOpt), paper #122.
> [workshop paper list](https://sites.google.com/seas.upenn.edu/corl-2024-workshop-diff/papers) **[WS]**

Existence, authorship, venue and paper number are confirmed from the workshop's
own page. The PDF is behind a Google Drive viewer that would not render; **its
benchmarks and numbers are therefore [UNVERIFIED] here.** It is cited by the
DWC paper (as Petersen et al. 2024a) alongside Valencia et al. 2019
("Using neuroevolved binary neural networks to solve reinforcement learning
environments"), and DWC characterises prior logic-network RL work as sparse —
"most DWN research focuses on classification rather than continuous control."

### How to state this honestly in project documents

> No differentiable-logic-gate-network result on Super Mario Bros has been
> located. The nearest verified results are (i) logic *circuits* learned as
> continuous-control policies on MuJoCo, competitive with deep policies on 4 of
> 5 tasks (Kresse & Lampert, ICML 2026), and (ii) a one-workshop-paper DLGN RL
> line whose numbers we have not read. Super Mario Bros *has* been played by an
> exotic-substrate network — a diffractive optical network trained with deep RL
> (Qiu et al., Advanced Photonics Nexus 2024) — but that is not a logic gate
> network.

The existing wording in `docs/LESSONS.md` ("was not identified in the initial
search... remains motivation supplied by the user") should be **upgraded from
"not found" to "searched thoroughly, does not exist"**, with the conflation
candidates named so the question does not get reopened.

---

## 2. DLGN state of the art

### 2.0 Bottom line

DLGNs are a real, active line with genuine nanosecond-scale inference results.
The headline numbers are considerably softer than they first appear:

1. **The best CIFAR-10 number (86.29%) is not a pure logic-gate result.** The
   LogicTreeNet-B/L/G models use **hand-designed edge and curvature feature
   detectors on 5-bit inputs** *and* **knowledge distillation from a neural
   network teacher**. Neither fact is in the abstract. **The code was never
   released and nobody has reproduced anything above LogicTreeNet-M (71%).**
2. **There is no ImageNet result at all.** Not at 224px, not at 1000 classes.
3. **Real practical depth is 4–6 layers.** Two independent 2026 papers show
   randomly-wired DLGNs get *zero* benefit from depth and actively degrade.
4. **Parameter efficiency versus CNNs is bad, not good.** 61M gates for 86.29%
   vs ResNet-20's 0.27M parameters for ~92%.
5. **Training cost is the dominant hidden cost** — 90 GPU-hours for a 62%
   CIFAR-10 model, "days to weeks" per the ETH group.
6. **The independent hardware record is much worse than the papers' own.**
   DiffLogicNet models repeatedly **fail to place-and-route on real FPGAs.**

### 2.1 CIFAR-10 — the central benchmark

| Model | Acc. | Size | Source | Status |
|---|---|---|---|---|
| DiffLogic Net (large×4) | 62.14% | 5.12M gates, 5 layers | Petersen 2022 | **[PR]** NeurIPS'22 |
| LogicTreeNet-S | 60.38% | 0.40M gates | Petersen 2024 | **[PR]** NeurIPS'24 Oral |
| LogicTreeNet-M | 71.01% | 3.08M gates | Petersen 2024 | **[PR]** |
| LogicTreeNet-B | 80.17% | 16.0M gates | Petersen 2024 | **[PR]** + feature detectors + NN teacher |
| LogicTreeNet-L | 84.99% | 28.9M gates | Petersen 2024 | **[PR]** + feature detectors + NN teacher |
| **LogicTreeNet-G** | **86.29%** | **61.0M gates** | Petersen 2024 | **[PR]** + feature detectors + NN teacher |
| CompactLogic (Ours-L) | **72.56%** | 1.08M BOPs | ETH SRI, [arXiv 2602.05830](https://arxiv.org/abs/2602.05830) | **[PP]** — *best independently verifiable pure-Boolean result* |
| IALGN (D=80, W=128k) | 59.61% | 10.24M gates | [arXiv 2607.21633](https://arxiv.org/abs/2607.21633) | **[PP]** |
| LILogicNet | 60.98% | 256K gates | [arXiv 2511.12340](https://arxiv.org/abs/2511.12340) | CVPR'26 Workshops |
| DLGN 6×64k baseline | 50.88 ± 0.87% | 384K gates | [arXiv 2509.25933](https://arxiv.org/abs/2509.25933) | **[PP]** |
| *Reference:* **ResNet-20** | **~91.3–92.6%** | **0.27M params** | He et al. 2015 | **[PR]** |

**The critical caveat, from Petersen 2024 §3.5:** for the larger CIFAR-10 models
(B, L, G) the authors use **5-bit precision inputs processed with edge and
curvature detector kernels with thresholds**, and Appendix A.2 states they use
**a neural network teacher supervising the class scores.** An independent ETH SRI
group excludes those models from comparison on the grounds that they include
floating-point feature-extraction layers and are therefore not Boolean networks;
Petersen's paper states the detectors are converted into LGNs and their gates
counted. **This is a genuine open dispute.** What is not in dispute: **the
86.29% figure depends on hand-engineered vision priors and a CNN teacher — you
still need a conventional network to get it.**

**Reproduction status** (the only independent attempt): TreeLogicNet-S reproduces
at 59.84% (reported 60.38%) and M at 69.15–71.37% (reported 71.01%).
**B, L and G have never been reproduced by anyone.**

This is directly relevant to TCN's constitutional rule in ARCHITECTURE.md that
"connected components, edge detectors, tokenizers … are not initial learned
primitives." **The DLGN state of the art violates the equivalent of that rule to
get its headline number.** TCN is holding itself to a stricter standard than the
work it cites as precedent — which is honourable, and which means TCN should not
expect to match those numbers.

### 2.2 MNIST, tabular, and where DLGNs genuinely win

MNIST tops out at 99.35–99.38% (LogicTreeNet-L, 1.27M gates; CompactLogic,
251K BOPs). The interesting story is a **~50× gate-count collapse driven
entirely by learning the connectivity**: Mommen et al.
([arXiv 2607.09399](https://arxiv.org/abs/2607.09399) **[PP]**, *Neural
Networks*) match Petersen's 98.47%/384k with **98.45% using one layer of 8,000
gates**. Fashion-MNIST ~90.6%; CIFAR-100 best ~30.6%.

**Tabular is where DLGNs look genuinely good.** Petersen 2022: MONK-1 100%,
Adult 84.8% with **1,280 parameters** matching a 3,810-parameter NN; Breast
Cancer 76.1% with **640 parameters**, beating the NN's 75.3% — at 20–50× lower
inference time. Yue & Jha (**[PR]** IEEE TCASAI, and *ACM TODAES* 31(5),
[DOI 10.1145/3795533](https://dl.acm.org/doi/pdf/10.1145/3795533), 2026-04-10)
report up to **1000× fewer gate-level ops** than NNs across 20 classification /
15 regression benchmarks — though on regression **random forest still ranks
first**, DLNs second.

**No ImageNet result exists.** The nearest points: on **ImageNet-32**, a DLGN
with 256,000 gates/layer × 6 layers *does not achieve performance comparable to
feed-forward networks* and **loses to a 3-layer 512-wide MLP beyond ~100
classes**; doubling to 512,000 gates/layer "does not significantly improve
performance." Tiny ImageNet best is **23.54%**. The one paper reaching 2000
classes does so on a **synthetic linearly-separable binary dataset**, not images.

### 2.3 Inference speed — the direction is real, the magnitudes are not apples-to-apples

Claims: "beyond a million MNIST images/s on a single CPU core"; 41.6M FPS on
FPGA (24 ns/image, VU13P); "1900× inference-speed improvement over SOTA"; and
the author's "4 nanoseconds" announcement.

Six problems, each verifiable in the papers:

1. **The 1M/s figure is for the 48K-gate, 97.69% model** at 625 ns/image. The
   98.47% model runs at 7 µs/image = **143k/s**, seven times slower.
2. **The flagship CIFAR-10 timings in Petersen 2022 were never measured.** The
   CPU and GPU times for large/large×2/large×4 are **all in parentheses =
   extrapolated**, because compilation to binaries did not finish.
3. **The "12× faster than FINN" claim compares an NVIDIA A6000 GPU to a
   2016-era Zynq FPGA** — different node, different decade, ~30× different
   power. The paper also notes only "7% utilization of the GPU."
4. **The OPs-based speedup rests on assumed conversion factors.** Petersen 2022
   states it *assumes* 100 OPs per FLOP, and 1000 binary OPs per sparse float32
   FLOP. The "two orders of magnitude" is a consequence of these assumptions,
   not a measurement.
5. **FPGA times are I/O-bound by the authors' own admission**, and no clock
   frequency is given to back-derive the 24 ns / 4 ns.
6. **Independent measurements diverge by orders of magnitude**, and LILogicNet
   measures DLGN FPGA **fmax at 26–40 MHz** on a 7-series part — a 6–8×
   discrepancy with the reciprocals implied by the published per-image times.

### 2.4 How deep do these networks actually go — **the answer is: shallow**

| Work | Depth actually used | Width |
|---|---|---|
| Petersen 2022 MNIST | **6 layers** | 8k–64k |
| Petersen 2022 CIFAR-10 | **4–5 layers** | 12k–1024k |
| Petersen 2024 LogicTreeNet | **23 logical layers, 15 trainable** | k=32…2560 |
| Mommen et al. 2026 | **1–6 layers** (stable to 10) | 1k–8k |
| LILogicNet | **1–4 layers** | 2k–32k |
| **BitLogic (TMLR 2026)** | **2 layers, deliberately** | 4k–64k |
| DiffLogic CA | 23-layer update net | 128 (3,199 gates, **336 active**) |

Petersen 2022 grid-searched depth over {2,…,10} and **selected 4–6**. Their own
appendix: logic gate networks can generally be trained efficiently **up to
around 8–10 layers, when training starts to suffer from vanishing gradients.**
Petersen 2024's residual initializations enable training beyond 6 layers "for the
first time," but the payoff is modest: 9 trainable → 82.68%, 11 → 83.32%,
15 → 84.99%. **+2.3 pp for a 1.7× depth increase.**

**The decisive experiment** — *On the Depth Scalability of Logic Gate Networks*,
An, Kim, Lee, Joo (Korea University), [arXiv 2607.21633](https://arxiv.org/abs/2607.21633)
**[PP]**, v3 2026-08-28. Fixed-width depth sweeps, 5 seeds. RWLGN = standard
randomly-wired DLGN; LDLGN = Light-DLGN parameterization; IALGN = their fix:

| Dataset | Width | Depth | Gates | RWLGN | LDLGN | IALGN |
|---|---|---|---|---|---|---|
| CIFAR-10 | 12k | 4 | 48k | 50.56 | 50.22 | 53.59 |
| CIFAR-10 | 12k | 50 | 600k | 50.16 | 45.40 | 56.12 |
| CIFAR-10 | 12k | **150** | 1.8M | **47.91** | **22.72** | 56.50 |
| CIFAR-10 | 128k | 4 | 512k | 57.01 | 54.01 | 57.31 |
| CIFAR-10 | 128k | 80 | 10.24M | 55.66 | 52.05 | 59.61 |
| MNIST | 8k | 6 | 48k | 97.29 | 97.54 | 98.22 |
| MNIST | 8k | 100 | 800k | 97.12 | 94.31 | 98.33 |

**A standard DLGN gains nothing from depth and slowly loses accuracy.** 37× more
gates buys **−2.65 pp**. The Light-DLGN parameterization **collapses 50.22% →
22.72%**. Even the fix buys +2.9 pp for 37× the gates — and their operation
analysis explains why: **88.4% of gates in the trained deep circuit select the
pass-through operation 'A'**, i.e. the network works by mostly doing nothing.

Two named obstacles: **optimization collapse** (gradients vanish through relaxed
Boolean ops) and **topology-induced credit degradation** (random wiring merges
output paths).

Corroboration from four independent groups: Mommen et al. — "for networks with
more than three layers, the accuracy saturates… the wider the network, the
better the performance… the reason why width plays such a major role compared to
depth is currently unknown"; BitLogic — "every configuration we evaluate is two
layers deep, and we deliberately did not sweep depth. Randomly-initialized LUT
stacks lose accuracy rapidly with depth"; LILogicNet — "at fixed gate budgets,
shallow and wide architectures often outperform deeper ones"; Light DLGN —
"scaling these networks in depth did not yield large expressivity benefits."

> **This is the most important finding in this report for TCN's architecture.**
> TCN's entire value proposition — compose crystallized modules into deeper
> programs, then recursively abstract — depends on depth paying off. In the
> substrate TCN cites as precedent, **depth demonstrably does not pay off**, and
> the diagnosed cause (credit degradation through random wiring, gradient
> vanishing through relaxed operators) applies to TCN's soft phase too.
> TCN's *typed, structured* wiring is a plausible answer to the topology half of
> that diagnosis — and demonstrating that would be a genuine contribution.

**Parameter efficiency:** LogicTreeNet-G is 61M gates → 86.29%; ResNet-20 is
0.27M parameters → ~92%. Even at a generous 10 gates per parameter, DLGNs are
~20× larger for 6 points less. The best independently-verifiable Boolean net is
1.08M BOPs → 72.56%.

### 2.5 Discretization

The original method is **softmax over 16 relaxed gates during training, hard
argmax at the end.** No annealing, no straight-through.

**Measured gap** (Mind the Gap Table 3, 6 layers, width 64k): MNIST 98.33 →
98.16 (**0.17**); CIFAR-10 52.04 → 50.72 (**1.31**); CIFAR-100 23.86 → 23.10
(**0.76**); FashionMNIST **0.36**; EMNIST-letters **0.38**.

**At depth 6 the gap is small — under 1.5 points. The problem is that it grows
with depth.** The largest gap on a real task is RDDLGN's WMT'14: 5.00 → 4.39
BLEU (**−12% relative**).

**Mind the Gap** (Yousefi, Plesner, Aczel, Wattenhofer, ETH; **NeurIPS 2025
[PR]**, [arXiv 2506.07500](https://arxiv.org/abs/2506.07500)) injects **Gumbel
noise + a straight-through estimator** — hard argmax forward, softmax gradients
backward. Results: **4.5× faster wall-clock training** (4.75× fewer iterations);
**98% reduction in the discretization gap**, stable across depths 6–12; **unused
gates 49.81% → 0.00%** — i.e. **nearly half of a standard DLGN's gates never
commit to anything.** Theory: Gumbel smoothing implicitly regularizes the
**trace of the Hessian**. Cost: τ needs tuning (goldilocks ~0.25; τ=0.01 and τ=2
degrade), and they disclose a **48 GPU-hour budget per run, 1,284 GPU-hours
total**, with the baseline DLGN *still improving* at 48 hours.

**Four competing 2025–26 approaches:**

| Approach | Mechanism | Result |
|---|---|---|
| **Light DLGN** [PP] | input-wise reparameterization (4 params/gate not 16) | Root cause is the parameterization: *"the logic gate that is rounded to is not necessarily the one that the neuron is closest to."* 4× smaller, 8.5× fewer steps — **but collapses at depth** (An et al.) |
| **Mommen et al.** [PP] | STE on gates **and** connections + remove constant-output gates + residual init + lr 0.1 | Gap **0.0–1.0 pp** at depths 1–6; **gap shrinks with width, not depth**. STE alone was not enough for 6-LUTNs |
| ⭐ **CompactLogic** [PP] | **Progressive layer-wise discretization — freeze each layer as it converges, shallow → deep** | **+2.2 pp on CIFAR-10 (60.74 → 62.92)**, and **1.8× faster training** once conv layers freeze |
| **CAGE** [PP] | confidence-adaptive gradient estimation; factorizes forward structure × stochasticity | MNIST L=6 selection gap: Hard-ST **0%**, Gumbel-ST 1%, Soft-Mix **24%**, Soft-Gumbel **40%**. Claim: *"noise alone does not reduce the gap"* — the hard forward pass is what matters |

⭐ **CompactLogic is the closest published thing to TCN's crystallization, inside
the DLGN literature, and it is a positive result.** Progressive layer-by-layer
freezing with continued training of the rest gains +2.2 pp *and* makes training
1.8× cheaper. It does **not** do TCN's rollback, per-node granularity, or
degradation-threshold acceptance test. **That gap is TCN's Track 1.**

Note also a real live disagreement: Mind the Gap credits **Gumbel noise**; CAGE's
analysis credits the **hard forward pass** and shows Gumbel-ST *without* CAGE
suffers a 47-point collapse at low temperature. TCN should not assume this is
settled.

### 2.6 Documented failure modes and scaling limits

1. **Temperature (τ) hypersensitivity — the most brittle knob in the method.**
   Same model, varying only τ: EMNIST-Balanced **51.56 → 80.31 → 64.29**;
   CIFAR-10 CLGN **27.80 → 65.23**; CIFAR-100 CLGN 15.88 → **30.96** → 25.60.
   **Swings of 25–40 points from one hyperparameter, with the optimum moving in
   opposite directions across datasets and class counts.** The scalability paper
   concludes τ *"should be treated as a primary optimization target."* This is a
   reproducibility hazard, not a tuning nicety. TCN has the same knob (§5 of
   ARCHITECTURE) and should treat it the same way.
2. **Learning-rate sensitivity.** With fully trainable connections at the
   *default* lr=0.01, Mommen et al.'s MNIST accuracy for 1–6 layers is
   **98% / 82% / 45% / 32% / 16% / 15%.** lr=0.1 is required. Their words: "by
   far, the biggest impact on making deep networks fully trainable is the
   increase of the learning rate."
3. **Vanishing gradients, quantified.** Light DLGN: gradient norm with the
   original parameterization *"undercuts machine precision after 16 logic layers
   already, and vanishes to 10⁻³⁴ over 40 layers"*; even with residual
   initialization it reaches 10⁻¹⁶ at 40 layers. Mechanism: sign symmetries in
   the 16-gate parameterization produce self-cancellations in the partial
   derivatives. A sharper theoretical version
   ([arXiv 2605.08657](https://arxiv.org/abs/2605.08657) **[PP]**) argues the
   16-dim softmax has a structural pathology where **11 of 15 simplex directions
   carry nullspace gradient, and at uniform initialization the backward signal
   vanishes exactly** — and proves no affine product reparameterization fixes
   it. The `difflogic` README's `grad_factor` knob is the authors' own
   acknowledgement.
   **TCN's `examples/joint.py` note that "initialization copies a predecessor to
   avoid cancellation from a perfectly uniform gate mixture" is the same
   pathology, independently rediscovered.**
4. **Dead and redundant gates.** 49.81% unused in a standard DLGN. At low τ,
   neurons are "consistently inactive ('dead'), fully active ('saturated'), or
   toggle in a synchronized manner." In deep IALGN, 88.4% are pass-through.
   LogicTreeNet-L MNIST: 3.2M gates during training → 698k after simplification.
5. **The receptive-field limit.** Each DLGN output neuron depends on only **2ⁿ
   inputs across n layers** — with six layers, 64 inputs, **~0.7% of an
   ImageNet-32 input vector**. Depth would fix it; vanishing gradients prevent
   depth. This is the cleanest statement of why DLGNs cannot scale to large
   inputs, and it is structural.
6. **Fixed random wiring is the core structural limitation, and learning it is
   expensive.** Petersen 2022's own justification is that relaxed connectivity
   "would add additional complexity to the relaxation, which would degrade
   performance." The cost estimate: a dense connectivity weight matrix for a
   single layer of a medium 2024 model *"can consume over 3400 GB of memory."*
   Fixing it is the single biggest accuracy-per-gate lever found (~50× gate
   reduction on MNIST, found independently by three groups) — but **BitLogic
   warns that full-layer learnable connectivity mode-collapses 7–8 pp below
   every bounded-candidate variant, with k=8–16 candidate pools best.**
   **This is directly load-bearing for TCN.** TCN learns operator *and input
   binding* jointly at every node, from "bounded predecessor pools." The
   literature says bounded pools are right and unbounded is a trap — TCN's
   design is on the correct side, and TCN's Track 3 (search scaling) is asking
   exactly the question the field has answered as "k=8–16."
7. **Generalization gap.** Light DLGN: *"DLGNs still have a considerable
   generalization gap despite data augmentations… standard techniques like
   dropout, random interventions, or residual connections fail to improve test
   performance. Designing constraints that promote generalizable functionality
   in DLGNs remains an open problem."*
8. **Training cost — the elephant.** Petersen 2022 CIFAR-10 large×4 (62.14%):
   **90.3 hours on an A6000** vs **0.8 hours** for their NN baseline — **113×**.
   Mind the Gap: *"days to weeks to train."* CompactLogic: 5.7–25.8 h per model
   for 62–72%. For calibration, ResNet-20 reaches ~92% in well under an hour.
   **The DLGN training bill is 20–100× higher for 20–30 points less accuracy.
   The inference win must be amortized over an enormous number of deployments to
   break even.** This is the single most important economic fact about the
   approach, and it applies to TCN's soft phase directly.
9. **Hardware: the models fail to route.** DWN (**[PR]** ICML 2024) reports for
   DiffLogicNet: *"Model could not be synthesized; hardware values are
   approximate,"* cause being *"routing congestion… the irregular connectivity
   between layers in DiffLogicNet and DWNs with n=2 proved impossible to map to
   the FPGA's programmable interconnect."* All n=6 DWNs routed successfully. An
   independent U. Florida study
   ([arXiv 2605.04109](https://arxiv.org/abs/2605.04109) **[PP]**, 390 models,
   Alveo U200) found **only 31.2% of hardware builds completed.** LILogicNet
   measures **routing at 75–87% of the critical path** and fmax at 20–40 MHz;
   the popcount/GroupSum head is **40–67% of hardware cost** (independently
   confirmed by three groups). The standard 6×64,000 topology **could not be
   routed in SKY130** ([arXiv 2604.19334](https://arxiv.org/abs/2604.19334)).
10. **Under a matched protocol, DiffLogic ranks last.** BitLogic (**[PR]** TMLR
    06/2026) retrained six methods in one framework on MNIST. At every width,
    DiffLogic is bottom: w=4K **75.92** vs DWN 87.11, PolyLUT 86.24, NeuraLUT
    85.91, LILogicNet 80.40; w=64K **93.68** vs PolyLUT 96.22. Read carefully —
    *every* method drops far below its published number under the shared
    protocol, so this is a fair **relative ranking under one budget, not a
    refutation of published absolutes.** Their diagnosis is the useful part:
    **fan-in is the dominant axis** — the n=2 → n=4 step explains most of the
    gap, and *"once n is matched, the relaxation family matters much less than
    the literature suggests."*
    **Implication for TCN: the *arity* of your operators may matter more than
    the cleverness of your relaxation.**
11. **Reproducibility.** The `difflogic` repo was **last pushed 2024-03-19**.
    **The NeurIPS 2024 convolutional code has never been released** (open
    requests #35 Mar'25, #41 Jun'25, #48 Nov'25, #50 Dec'25). Issue #45, asking
    how inference time was measured, is unanswered. Everyone now uses
    third-party reimplementations.
12. **Patent encumbrance — actionable.** MIT-licensed code, patent-pending
    method: **WO 2023143707 A1**, **US 18/883,354** (pending), **TW 112101045**
    (granted). Verify independently before commercial deployment of a plain
    relaxed-gate-choice layer.

### 2.7 Published criticism

**Correction to a common assumption, including my own earlier note:** Hacker
News `item?id=44118846` is **not a submission** — it is a comment by *enricozb*,
2025-05-28, on the story "Compiling a neural net to C for a speedup"
(id 44118373). No paper author responded anywhere in the subtree.

Substantive criticisms from it:
- *enricozb*: **"I still don't like that the wiring is fixed initially rather
  than learned. I did some extremely rough research into doing learnable
  wirings, but couldn't get past even learning ~4-bit addition."**
- *thesz*, the sharpest technical point: **"selecting a wire has all same
  problems as selecting an operation, such as vanishing gradient, but no
  apparent mitigation like residual initialization. And for wire selection
  gradient vanishes quicker, because there are more wires… than operations
  (16)."**
- *Lerc*, on novelty: "I remember reading about adaptive boolean logic networks
  in the 90's… It probably goes back considerably earlier."

**Named-expert skepticism** (MIT Technology Review, Grace Huckins, 2024-12-20
**[BLOG/journalism]**): **Farinaz Koushanfar (UC San Diego): "It's a cute idea,
but I'm not sure how well it scales"** — the relaxation is only an approximation
and may break down as networks grow. **Zhiru Zhang (Cornell)**, conditionally
positive: "If we can close the gap, then this could potentially open up a lot of
possibilities."

**The unanswered objection**, from a commenter on the DiffLogic-CA thread:
*"Circuit synthesis is a really well-researched field… I wonder if you have
tried your learner against the IWLS competition data sets."* **The DLGN line has
never answered the classical-logic-synthesis-baseline objection.** This is
exactly TerpreT's finding (§4.3) reappearing in a different field, and it is the
question a skeptical reviewer will put to TCN.

**Practitioner reproduction reports [BLOG]:** Isaac Clayton
(slightknack.dev) — "the model would not converge"; "Hyperparameters suck!";
learning wiring: "Poorly"; and he self-debunks his own headline speedup. Enrico
Borba (ezb.io) — could not exceed 4-bit addition.

**What nobody has done:** no paper attempts and fails to replicate the
86.29%/61M-gate result — it has been **neither refuted nor independently
confirmed, because the code was never released.** No independent LUT count,
fmax, or power for any LogicTreeNet exists. **No returned-silicon measurement of
any logic gate network exists.** All power figures are EDA-estimated.

Note the shape of the field: **a sustained multi-paper critique from ETH Zürich
(Wattenhofer's lab: Mind the Gap, Light DLGN, From MNIST to ImageNet, RDDLGN,
BitLogic), a second ETH lab (Vechev's SRI: CompactLogic), Korea University (depth
scalability), and independent hardware audits from UT Austin, U. Florida and
Fulda.** The critical literature is now larger than the original line.

---

## 3. RL and sequential control with discrete / logic-gate / program-synthesis policies

### 3.0 One-line summary

Differentiable program synthesis has been combined with policy learning
repeatedly since 2018. **It works on gridworlds, classic control, and
low-dimensional MuJoCo. It has never produced a competitive pixel-input Atari
result from a discrete/logical program policy.** The universal failure mode is
the **hardening cliff**: the soft relaxation trains fine and discretization
destroys it. The specific niche of *logic gate networks as an RL policy* is
essentially unoccupied — one ICML 2026 paper, one workshop abstract.

### 3.1 Logic gate networks / Boolean circuits as a control policy — the gap

Everything that exists:

- **Differentiable Weightless Controllers.** Kresse & Lampert (ISTA).
  **ICML 2026 [PR]**. [arXiv 2512.01467](https://arxiv.org/abs/2512.01467).
  Thermometer-encoded observations (63 thresholds/dim) → two layers of sparsely
  connected **6-input Boolean lookup tables** → group summation + learned affine
  action head. **Trained end-to-end with SAC directly — not distilled** (critics
  stay floating point; only the policy is a circuit). Full table in §1. Hardware:
  2–3 clock cycles, **1.05–2.28 nJ/action**, 0.8–3.2k LUTs, up to 100M actions/s,
  no DSP/BRAM; ~1000× lower latency and energy than the quantized-NN approach.
  **Stated limitations:** HalfCheetah is a representation-capacity/output-
  resolution bottleneck (−34%); training is up to **7.7× slower than FP** on
  Humanoid; training feasible only in simulation.
- **Efficient Reinforcement Learning Agents with Differentiable Logic Gate
  Networks.** Petersen, Borgelt, Ermon. **CoRL 2024 DiffOpt workshop**,
  spotlight, 2024-11-09, paper #122.
  [workshop page](https://sites.google.com/seas.upenn.edu/corl-2024-workshop-diff/papers) ·
  [talk video](https://www.youtube.com/watch?v=YEAQlkV9JAw) **[WS]**.
  The talk claims **nanosecond latencies for RL agents**. **The PDF was not
  retrievable and no archival publication has followed as of 2026-09.** Treat as
  an unpublished workshop abstract. *(Worth emailing the authors.)*
- **Logic Gate Neural Networks are Good for Verification.** Kresse, Yu, Lampert,
  Henzinger. **NeSy 2025 [PR]**, [PMLR v288:90–103](https://proceedings.mlr.press/v288/kresse25a.html).
  SAT encoding for global robustness/fairness of LGNs. Not RL, but it is the
  argument for why discrete circuits are worth the accuracy concession.
- **Differentiable Logic Cellular Automata.** ALIFE 2025 **[PR]**. Discrete at
  inference, recurrent logic circuits as a CA update rule — but **no reward, no
  sequential decision problem, no agent.** The maze-steering demo is interactive
  steering, not a learned policy. Note that the *non-logic* version already does
  control: NCA + deep Q-learning on cart-pole with regeneration after damage
  ([arXiv 2106.15240](https://arxiv.org/abs/2106.15240) **[PP]**).
  **DiffLogic-CA + RL is an obvious unoccupied combination.**
- Adjacent, small-scale: **WiSARD / weightless neural networks** with RL
  variants (LogicWiSARD, ASAP 2022) — DWC is the modern descendant; and
  **Tsetlin Machines** with off-policy and on-policy RL variants, small
  benchmarks only.

> **The gap, stated plainly: there is no published logic-gate-network or
> Boolean-circuit RL policy on pixel input, on Atari, or on any discrete-action
> high-dimensional environment.** The entire intersection is one ICML 2026 paper
> on state-vector MuJoCo, one workshop abstract with no archival follow-up, and
> an ALIFE paper with no reward signal.

### 3.2 Programmatic RL — real, and firmly capped

| Work | Venue | Domain | Outcome |
|---|---|---|---|
| **PIRL / NDPS**, Verma, Murali, Singh, Kohli, Chaudhuri | **ICML 2018 [PR]** [PMLR](https://proceedings.mlr.press/v80/verma18a.html) | TORCS | Train a DDPG oracle, then locally search a hand-written **PID-controller sketch** minimizing distance to it. CG-Speedway-1: NDPS lap **1:01.56** / reward 115.32 vs DRL lap **54.27** / reward 118.39. **Strictly worse than its own teacher on reward and much slower on lap time.** Wins claimed on smoothness, noise tolerance, transfer. |
| **PROPEL**, Verma, Le, Yue, Chaudhuri | **NeurIPS 2019 [PR]** | TORCS (5 tracks), MountainCar, Pendulum | Mirror descent: gradient step in unconstrained neural space, then **project back into program space via imitation learning**. Has convergence theory. G-Track lap/crash: DDPG **78.82/0.24**, NDPS 108.25/0.24, VIPER 83.60/0.24, **PropelTree 78.33/0.04**. On Ruudskogen and Alpine-2 **DDPG, NDPS and VIPER all crash on a majority of seeds** while PROPEL completes. Authors: "on an absolute level, the generalization ability of Propel still leaves much room for improvement." |
| **π-PRL**, Qiu & Zhu | **ICLR 2022 Spotlight [PR]** | — | Removes NDPS's oracle requirement. **Program size ceiling: trees of depth 6.** |
| **LEAPS**, Trivedi, Zhang, Sun, Lim | **NeurIPS 2021 [PR]** | Karel, 8×8 grids | VAE over random DSL programs → latent space; **CEM search in the latent space**. **Max program length 44 tokens, mean 17.9.** Trained on 8×8, transfers to **100×100** StairClimber/Maze. Loses to plain DRL on Harvester (0.45 vs 0.90). Authors: cannot synthesize programs more complex than the training program distribution; the 44-token cap is load-bearing. |
| **HPRL**, Liu et al. | **ICML 2023 [PR]** | Karel, Karel-Hard | A meta-policy composes a *sequence* of LEAPS programs. **Ceiling ~5 × 40 = 200 tokens.** Introduced Karel-Hard because the original six tasks were saturated. |
| **Reclaiming the Source of Programmatic Policies**, Carvalho, Tjhia, Lelis | **ICLR 2024 [PR]** [arXiv 2410.12166](https://arxiv.org/abs/2410.12166) | Karel + Karel-Hard | **The correction.** Plain **hill climbing directly in the DSL** — no VAE, no training — beats LEAPS and HPRL on every task: DoorKey **0.84** vs 0.50/0.50, Snake **0.65** vs 0.23/0.33. Diagnosis: latent-space search gets trapped in local maxima; their appendix confirms high-reward policies **exist** in the latent space, search just cannot find them. |
| **LLM-GS**, Liu et al. | **ICLR 2025 [PR]** [arXiv 2405.16450](https://arxiv.org/abs/2405.16450) | Karel | GPT-4 emits Python → DSL, then scheduled hill climbing. DoorKey converges at ~500K evaluations where **LEAPS/HPRL/CEBS/HC never converge within 1M**. Authors: explicitly "difficult to apply to low-level control tasks, such as motor torque control." |
| **InnateCoder**, Moraes et al. | **IJCAI 2025 [PR]** | Karel, MicroRTS to 64×64 | LLM generates programmatic *options* zero-shot to induce a semantic search space. |
| **Semantic-space search**, Moraes & Lelis | **IJCAI 2024 [PR]** | MicroRTS | The strongest "programs beat humans" result in the field: **beat the human-written winners of the two most recent MicroRTS competitions.** |
| **VIPER**, Bastani, Pu, Solar-Lezama | **NeurIPS 2018 [PR]** | CartPole, symbolic Pong, HalfCheetah | Decision trees distilled from a DNN oracle. CartPole 200.0 in **3 nodes**. Atari Pong 21.0 = DQN — **on a 7-D hand-built symbolic state, not pixels — in a 769-node tree.** HalfCheetah 4014 vs 4189 in **9,757 nodes.** Z3 safety proof on cart-pole in 1.5 s. *(Correction to a common belief: VIPER has no half-field-offense experiment.)* |
| **Deep Symbolic Policy**, Landajuela et al. | **ICML 2021 [PR]** | 8 control envs | Autoregressive RNN + risk-seeking policy gradient. **Average policy length 8.25 tokens per action dimension.** Competitive with SAC/TD3 on CartPole/LunarLander/BipedalWalker, loses on Hopper (2122 vs 2744). **Sample efficiency is catastrophic: 2,000,000 episodes for CartPole** (ESPL: 500). **Multi-dimensional actions require "anchoring" — distilling a pre-trained NN one action dimension at a time**, so beyond 1-D it is distillation wearing search's clothes. |
| **Neural DNF-MT**, Baugh, Dickens, Russo | **AAMAS 2025 [PR]** | various | Differentiable DNF layers → extractable ASP/ProbLog, **editable** and re-insertable. **Blackjack: −0.050 soft → −0.099 after ProbLog extraction (~2× worse purely from discretization).** Taxi **could not be trained end-to-end at all** and had to be distilled from an MLP. |

**Karel's honest status:** an 8×8 grid, ~4-bit observation, 5 actions. Saturated.
Karel-Hard nearly saturated. This line has not scaled past gridworlds and
MicroRTS.

### 3.3 Neurosymbolic / differentiable-logic RL — and the Atari reality check

| Work | Venue | Scale actually demonstrated |
|---|---|---|
| **Neural Logic Machines**, Dong et al. | ICLR 2019 **[PR]** | Blocks world, sorting, shortest path over symbolic predicates. Generalizes m≤12 → m=50. **But object count must stay <1000, complexity O(m^B) with arity B≤3, and Blocks World "graduated" only 40% of 10 seeds.** |
| **NLRL**, Jiang & Luo | ICML 2019 **[PR]** | **4 blocks, 5×5 cliff-walking.** The memory-reduced variant still uses "less than 10 GB" — **on a 25-state MDP.** Hand-designed rule templates are hyperparameters. |
| **dNL-ILP for RL**, Payani & Fekri | **[PP]** [arXiv 2003.10386](https://arxiv.org/abs/2003.10386) | GridWorld 12×12 in **700 episodes vs A2C >10⁸** — but requires **20–50 hand-labeled scenes and a pre-abstracted 8×12 grid**. Ceiling: 5 objects. |
| **Differentiable Logic Machines**, Zimmer et al. | **[PP]** → TMLR | The honest scaling study: **Sorting 0.939 @ m=10 → 0.559 @ m=50; Blocksworld 0.230 @ m=50.** Authors name a "contradiction between learning lessons and converging to interpretable policies." |
| **Logical Optimal Actions** | ACL 2021 **Demo** | **Contains no scores at all.** Do not cite for performance. |
| **Scallop** | PLDI 2023 **[PR]** | Its RL evaluation is **PacMan-Maze only.** |

**The decisive table — INSIGHT (ICML 2024 Spotlight) re-running all baselines on
Atari:**

| Game | INSIGHT | Neural PPO | **NUDGE (pure logic)** | DSP | CGP |
|---|---|---|---|---|---|
| Pong | 20.9 | 20.4 | **−7.2** | −1 | 20 |
| Seaquest | 2665.7 | 1804.8 | **0** | 193.3 | 724 |
| Breakout | 409.6 | 259.6 | **3.4** | 4.3 | 13.2 |
| Enduro | 843.7 | 676.9 | **2.4** | 41.1 | 2 |
| SpaceInvaders | 1232.6 | 1184.6 | **80** | 222.6 | 1001 |

**A purely logical differentiable policy on Atari scores 0.5%–9% of DQN, and on
Pong is barely above random.** And INSIGHT — the one neurosymbolic method
reaching PPO parity across nine games — gets there by **abandoning logic**: it
uses an **EQL symbolic regressor** (see §4.4), and its authors state the EQL
network "cannot express logical operations required by some reasoning tasks."

Everything that "works" on Atari consumes **OCAtari ground-truth object lists,
not pixels**: NUDGE (NeurIPS 2023), SCoBots (NeurIPS 2024 — and its final policy
is a VIPER-distilled tree), blendRL (ICLR 2025 Spotlight — which **bolts a
neural policy back on**, an admission in itself). The one genuine pixels→rules
result ([arXiv 2410.14371](https://arxiv.org/abs/2410.14371) **[PP]**) gets
**Pong 14.4** (neural 17–19) and **Boxing 51.8** (neural 78.8–97), and Boxing
drops **91.2 → 51.8** purely from swapping ground-truth objects for learned
ones.

**The only real pixel-input Atari results from synthesized discrete programs come
from evolutionary genetic programming, not differentiable synthesis:**
**CGP** (Wilson, Cussat-Blanc, Luga, Miller, **GECCO 2018 [PR]**,
[arXiv 1806.05695](https://arxiv.org/abs/1806.05695)) — best among artificial
agents on **8 of 61 games** from raw RGB, with genomes of 40 nodes of which only
a fraction are active; and **TPG** (Kelly & Heywood, GECCO/EuroGP 2017–18,
IJCAI 2018 **[PR]**) — best score in **14 of 20 games**, exceeding deep learning
on 15/20 and human on 7 of those, in real time without a GPU. CGP's own caveats
are damning: total failure on exploration games (Montezuma's Revenge, Venture =
0), evolution trapped by "simple strategies," and **some high-scoring agents did
not use the pixel input at all — they just repeated fixed actions.**

> **Nobody has a readable discrete program policy playing an Atari game from
> pixels at DQN parity.** Which is a second, independent reason to disbelieve
> the Mario claim: the field is nowhere near it.

### 3.4 Binary / quantized policies

- **QuaRL** ([arXiv 1910.01055](https://arxiv.org/abs/1910.01055) **[PP]**):
  int8 post-training quantization is essentially **free** (2–5% relative error,
  no reward loss); **QAT holds to 5–6 bits, then reward drops.** QAT sometimes
  *beats* fp32 — quantization as regularizer.
- **Quantized Continuous Controllers for Integer Hardware**
  ([arXiv 2511.07046](https://arxiv.org/abs/2511.07046) **[PP]**, Nov 2025):
  SAC at **3 bits matches FP32 on most MuJoCo**; 2 bits works on
  Walker2d/Ant/Hopper and **collapses on HalfCheetah**. **Key insight: the
  bottleneck is observation precision, not weights** — internals tolerate 2–3
  bits while inputs need 3–8.
- **Deep Binary RL for Scalable Verification** (Lazarus & Kochenderfer,
  [arXiv 2203.05704](https://arxiv.org/abs/2203.05704) **[PP]**) — the decisive
  binary-on-Atari numbers: Beam Rider **401 vs DQN 7456 (5.4%)**, Q*bert **179
  vs 18900 (0.9%)**, Seaquest **133 vs 28010 (0.5%)**, Enduro **0 (0%)**. Naive
  binarization **fails to converge on most games**; recovery needs a
  full-precision target network plus widening. **The real payoff is
  verification, not reward: DQN verified 0/50 properties in 30 min; the binary
  net verified 39/50.**

### 3.5 Progressive hardening — the mechanism TCN's Track 1 tests

**The hardening cliff is universal, and it is a cliff, not a slope.**

Differentiable decision trees, post-hoc discretized (Silva & Gombolay et al.,
**AISTATS 2020 [PR]**, [PMLR v108](https://proceedings.mlr.press/v108/silva20a.html)):

| Domain | DDT (soft) | **Discretized** |
|---|---|---|
| Cart Pole | 500 ± 0 | 499.5 ± 0.8 |
| **Lunar Lander (8-D)** | **97.9 ± 10.5** | **−88 ± 20.4** |
| **FindAndDefeatZerglings (SC2)** | 6.6 ± 1.1 | **4.2 ± 1.6 (−36%)** |

ICCT (Paleja et al., **RSS 2022 [PR]**,
[PDF](https://www.roboticsproceedings.org/rss18/p068.pdf)):

| Domain | CDDT (soft) | **CDDT-crisp** | ICCT (trained crisp) |
|---|---|---|---|
| Inverted Pendulum | 1000.0 | **5.0** | 1000.0 |
| Lunar Lander | 226.4 ± 44.5 | **−451.6 ± 97.3** | 300.5 ± 1.2 |
| Lane Keeping | 464.7 ± 5.4 | **−43,526.0 ± 15,905.0** | 476.6 ± 3.1 |

**1000 → 5. 464 → −43,526.** The only known fix is to train in (or toward) the
crisp representation from the start: ICCT's differentiable crispification,
SYMPOL's straight-through PPO ([arXiv 2408.08761](https://arxiv.org/abs/2408.08761)
**[PP]**, train→test information loss Cohen's d **−0.019** vs soft-DT **−3.126**
and VIPER-style **−3.449**).

**And there is now a paper doing almost exactly TCN's Track 1, six months ago:**

> **DiPRL: Learning Discrete Programmatic Policies via Architecture Entropy
> Regularization.** Chengpeng Hu, Yingqian Zhang, Hendrik Baier.
> [arXiv 2605.18508](https://arxiv.org/abs/2605.18508), **2026-05-18. [PP]**, no
> venue listed.

Its abstract states the problem in TCN's own terms: gradient-based methods
optimizing continuous relaxations of programs *"face a significant performance
drop when converting the continuous relaxations back into discrete programs.
Post-hoc discretization can discard optimized branches and parameters in a
program, which results in a collapse of policy expressivity and lowered task
performance."*

Its fix is **not** a hardening schedule with rollback. It is **programmatic
architecture entropy regularization**, annealed with automatic coefficient
tuning, so the policy is already near-discrete when training ends — no post-hoc
step at all. Architecture entropy before extraction: π-PRL **0.284–0.861**,
DiPRL **0.000–0.004**. Results vs π-PRL: MountainCar **−110.79** vs −171.83;
MiniGrid DoorKey **0.95** vs 0.66; HalfCheetah Hurdle **723.88** vs **−443.80**;
Ant RandomGoal **413.12** vs 264.75. Their motivating figure shows that on Ant
RandomGoal, post-hoc discretization causes a sudden drop and **"even after
another 50% of training, its performance is never recovered."**
Author-stated limitation: if the optimal strategy needs constructs the DSL lacks
(loops, temporal operators for long horizons), DiPRL converges to an
interpretable but suboptimal program.

**Practical synthesis of the whole hardening literature** (DiPRL + Mind the Gap +
ICCT + SYMPOL + INQ):

1. **Do not discretize post-hoc — it does not recover.**
2. Inject **Gumbel noise + straight-through** during training.
3. Anneal temperature inside a **goldilocks band** — Mind the Gap reports that
   τ > 1 or τ < 0.1 converges much slower.
4. **Regularize toward low architecture/selection entropy** so the model is
   already near-discrete before you harden.
5. **Overparameterize: width is the cheapest way to shrink the hardening gap**
   (Mind the Gap: gap contracts from 48K → 512K neurons on CIFAR-10). Depth is
   not — see §2.4.

### 3.6 The other failure modes, blunt

1. **Program-size ceilings are small and consistent.** LEAPS 44 tokens; HPRL
   ~200; π-PRL depth-6 trees; DTPO 16 leaves (CartPole near-optimal at **4**,
   CartPoleSwingup needs **64+**); DSP 8.25 tokens/action-dim; SYMPOL ~50 nodes
   post-pruning. Where programs get big they stop being interpretable: VIPER's
   769-node Pong tree and 9,757-node HalfCheetah tree are compressed neural nets
   with worse ergonomics. **Verifiability ≠ interpretability.**
2. **State dimensionality is the binding constraint, and the wall is ~20-D.** A
   systematic study ([arXiv 2503.08322](https://arxiv.org/abs/2503.08322)
   **[PP]**) finds state dimensionality carries **80.87%** of the feature
   importance for whether an interpretable policy exists, and states that for
   very high-dimensional environments like Seaquest (180 state dimensions) **no
   baseline can solve the game.** Everything that works lives in 2–24 dims.
3. **Sample efficiency is often catastrophically worse.** DSP: 2M episodes for
   CartPole. LLM-GS's framing of prior programmatic-RL SOTA: "tens of millions
   of program-environment interactions."
4. **Seed variance is the unreported killer.** NLM Blocks World graduated **40%
   of 10 seeds**; NLM family-tree tasks 20–100%; Neural DNF-MT Door Corridor
   fails to finish in 6/32 runs; SCoBots Seaquest fails 1 of 3 seeds. "100%
   accuracy" is conditioned on convergence.
5. **The generalization advantage is partly an artifact.** *Common Benchmarks
   Undervalue the Generalization Power of Programmatic Policies*
   ([arXiv 2506.14162](https://arxiv.org/abs/2506.14162) **[PP]**, position
   paper) shows TORCS DRL going from **0% OOD success to 69–100%** just by using
   a cautious reward (β=0.5), and Karel PPO from ~0.00 to **0.04–1.00** on
   100×100 with sparse observations and a simpler architecture. The residual
   real advantage is narrow: tasks needing algorithmic constructs (stacks,
   queues), for which they had to invent a new benchmark.
6. **Search space matters more than representation learning.** LEAPS's core
   premise — that a learned latent program space is easier to search — was
   **empirically falsified** at ICLR 2024 by plain hill climbing in the raw DSL.
7. **Multi-dimensional continuous action collapses "search" into
   "distillation."** DSP's anchoring needs a pre-trained NN; PIRL/NDPS needs a
   DRL oracle; VIPER, SCoBots and Neural-DNF-MT-on-Taxi are all distillation.
   **Direct RL over discrete program/circuit structure at scale is done by
   exactly one paper in this survey: DWC.**

### 3.7 What worked

- **Distillation, not direct synthesis.** PIRL, VIPER, SCoBots, Neural DNF-MT,
  DSP-with-anchoring: train something soft first, then *extract* a program. This
  is by far the most reliable recipe in the literature.
- **Projection loops with theory.** PROPEL's mirror-descent-with-projection is
  the principled version of "alternate soft optimization and hardening," and it
  is the closest published theory for what TCN's crystallization loop attempts.
- **Programs transfer and are safe rather than high-scoring.** PROPEL's crash
  ratios (0.04 vs DDPG's 0.24–0.92) and the binary-net verification result
  (39/50 vs 0/50 properties proved) are the durable wins.
  **If TCN wants a defensible headline it is cost, verifiability and
  generalization — not return.**

---

## 4. Neighbouring fields TCN should not reinvent

This is the section with the most direct consequences for the codebase.

### 4.1 Neural program synthesis and induction — status: the differentiable branch is dead

| Work | Venue | Status |
|---|---|---|
| Neural Turing Machines, [arXiv 1410.5401](https://arxiv.org/pdf/1410.5401) | 2014 **[PP]** | existence proof, did not scale |
| Differentiable Neural Computer | Nature 538, 2016 **[PR]** | did not scale |
| Neural Programmer-Interpreter, [arXiv 1511.06279](https://arxiv.org/abs/1511.06279) | ICLR 2016 **[PR]** | key-value *program memory*; direct ancestor of recursive abstraction |
| Neural GPU | ICLR 2016 **[PR]** | learned long arithmetic |
| Extensions and Limitations of the Neural GPU, [arXiv 1611.00736](https://arxiv.org/abs/1611.00736) | 2016 **[PP]** | **the negative result** |
| DeepCoder | ICLR 2017 **[PR]** | neural-*guided* classical search — the part that survived |
| RobustFill, [ICML 2017](https://proceedings.mlr.press/v70/devlin17a/devlin17a.pdf) | **[PR]** | ditto |
| ExeDec, [arXiv 2307.13883](https://arxiv.org/pdf/2307.13883) | ICLR 2024 **[PR]** | neural synthesizers still fail compositional generalization |
| ARC Prize 2025 Technical Report, [arXiv 2601.10904](https://arxiv.org/abs/2601.10904) | 2026 **[PP]** | current state of program-search-for-tasks |

**The single most useful negative result here.** Price, Zaremba & Sutskever
(OpenAI) found a Neural GPU that generalised near-perfectly to 100-digit decimal
multiplication and yet **failed on `000…002 × 000…002` while succeeding on
`2 × 2`**. Their own framing is that these failures are reminiscent of
adversarial examples. **A synthesized program can pass every length- and
held-out-generalization test you thought to design and still not implement the
algorithm.** TCN's `docs/VALIDATION.md` currently reports 4/4 on 16 held-out
episodes of a fixed-structure task; that is exactly the regime where this
failure mode is invisible.

What survived from this line, per Chaudhuri's 2025 neurosymbolic program
synthesis handbook chapter
([UT Austin](https://www.cs.utexas.edu/~swarat/pubs/ns-handbook-2025.pdf) **[PP]**):
neural relaxation is retained only as a **heuristic inside a discrete search**,
never as the synthesis mechanism. ARC Prize 2025's defining theme was the
**refinement loop** — per-task iterative program optimization under execution
feedback; top ARC-AGI-2 private-eval score was **24%** across 1,455 teams.
Notably **CompressARC** (pure MDL/code-golf per puzzle, no pretraining) reaches
~20–34% on ARC-AGI-1 but **~4% on ARC-AGI-2** — evidence that a pure
description-length objective over discrete programs plateaus hard as task
complexity rises. TCN's `lambda_mdl` term should be read in that light.

### 4.2 DreamCoder-style library learning — directly relevant to track 5

| Work | Venue | Link |
|---|---|---|
| DreamCoder | PLDI 2021 **[PR]** | [ACM](https://dl.acm.org/doi/10.1145/3453483.3454080) |
| DreamCoder (journal) | Phil. Trans. R. Soc. A 381:20220050, 2023 **[PR]** | [RS](https://royalsocietypublishing.org/rsta/article/381/2251/20220050/112456/) |
| LAPS | ICML 2021 **[PR]** | [arXiv 2106.11053](https://arxiv.org/abs/2106.11053) |
| **STITCH** | POPL 2023 **[PR]** | [arXiv 2211.16605](https://arxiv.org/abs/2211.16605) |
| babble (e-graphs + anti-unification) | POPL 2023 **[PR]** | [arXiv 2212.04596](https://arxiv.org/pdf/2212.04596) |
| LILO | ICLR 2024 **[PR]** | [arXiv 2310.19791](https://arxiv.org/abs/2310.19791) |
| ReGAL | ICML 2024 **[PR]** | [PMLR](https://proceedings.mlr.press/v235/stengel-eskin24a.html) |
| **LLM Library Learning Fails (LEGO-Prover case study)** | 2025 **[PP]** | [arXiv 2504.03048](https://arxiv.org/abs/2504.03048) |

**Does library learning pay off? Conditionally yes.** DreamCoder's own ablation:
in generative structure-building domains (LOGO graphics, tower building),
ablations *without* library learning never solve more than **60%** of held-out
tasks while full DreamCoder solves nearly **100%**. LILO over DreamCoder:
**+33.14 REGEX, +20.42 LOGO, +2.26 CLEVR** solve rate. ReGAL (gradient-free,
pure refactoring): **+11.5% LOGO, +26.1% date understanding, +8.1% TextCraft**.

**But four findings should change how track 5 is designed:**

1. **STITCH shows the abstraction-discovery step does not need to be neural or
   differentiable.** Replacing DreamCoder's compression with corpus-guided
   top-down synthesis plus syntactic pruning is **1000–10000× faster and ~100×
   less memory**, producing libraries of comparable or better quality across
   Lists/Text/LOGO/Towers/Physics and 8 graphics corpora. DreamCoder's
   compression takes minutes-to-hours; STITCH takes milliseconds. STITCH's own
   limitation: it cannot natively learn **higher-order** abstractions.
2. **A newly registered abstraction is a net *loss* by default.** LILO's early
   experiments handing raw STITCH abstractions to the solver:
   **−30.60 (REGEX), −2.91 (CLEVR), −11.11 (LOGO)**. Only after AutoDoc
   (auto-generated names + docstrings) did abstractions become net-positive
   (+9.73 REGEX, +2.27 CLEVR). LILO also documents semantic drift in generated
   names ("looping move and rotate" for what is actually "draw polygon").
   **TCN's typed signature alone is not enough to make a crystallized module
   usable by later synthesis.**
3. **Library-learning gains can be entirely illusory under compute-matching.**
   Berlot-Attwell, Rudzicz & Si re-examined LEGO-Prover and found it does not
   improve over simply prompting the model once computational cost is
   accounted for, with no evidence of direct lemma reuse and evidence
   *contradicting* indirect reuse.
4. **You must start with a good DSL.** DreamCoder explicitly requires a base
   language expressive enough to solve all training problems in principle.
   Library learning recombines; it does not invent the primitive vocabulary.

Cost anchor: DreamCoder to convergence is **~1 day on 20–100 CPUs**, up to
**~5 days on 64 CPUs** for the hardest 20-problem domains.

### 4.3 Differentiable programming languages — the TerpreT verdict

> **TerpreT: A Probabilistic Programming Language for Program Induction.**
> Gaunt, Brockschmidt, Singh, Kushman, Kohli, Taylor, Tarlow. Microsoft
> Research, submitted 2016-08-15. [arXiv 1608.04428](https://arxiv.org/abs/1608.04428) **[PP]**
> — *never accepted at a main venue; its influence was entirely through this
> negative result.*

TerpreT is a controlled experiment: one model specification, four inference
back-ends (gradient descent, LP relaxation, SAT solving, and the Sketch
synthesizer). Verbatim from the abstract:

> **"Our key empirical finding is that constraint solvers dominate the gradient
> descent and LP-based formulations."**

This was built by the group that was building differentiable interpreters, and
it says that on the task of recovering a discrete program from I/O examples,
**gradient descent on a continuous relaxation of program choice loses to a SAT
solver.** It is the load-bearing negative result for TCN's framing. The
differentiable-interpreter line at MSR wound down shortly after.

Adjacent:
- **∂4 / Differentiable Forth**, ICML 2017 **[PR]**,
  [arXiv 1605.06640](https://arxiv.org/abs/1605.06640) — works, but only because
  a human writes a program *sketch* with small trainable slots. The search
  burden moved to the human.
- **Differentiable Programs with Neural Libraries** (NTPT), Gaunt,
  Brockschmidt, Kushman, Tarlow, 2016/2017 **[PP]**,
  [arXiv 1611.02109](https://arxiv.org/abs/1611.02109) — differentiable
  interpreter + a *reusable library of learned neural functions* that transfers
  across tasks. This is TCN's "recursive module abstraction inside a
  differentiable program," published in 2016, by the TerpreT authors.
- **DeepProbLog** (NeurIPS 2018 / AIJ) — exact inference is **#P-hard**;
  documented exponential BDD blowup in the number of neural ground atoms.
- **Scallop**, PLDI 2023 **[PR]**, [arXiv 2304.04812](https://arxiv.org/abs/2304.04812)
  — the engineering answer that works: differentiable reasoning via
  **provenance semirings** with *tunable* top-k provenance, so you buy
  scalability by approximating the provenance rather than the logic. This is
  the right template for "differentiate through a symbolic layer at
  controllable, chosen cost."

### 4.4 Neurosymbolic program induction — HOUDINI is TCN's closest published ancestor

> **HOUDINI: Lifelong Learning as Program Synthesis.** Valkov, Chaudhari,
> Srivastava, Sutton, Chaudhuri. **NeurIPS 2018 [PR]**.
> [arXiv 1804.00218](https://arxiv.org/abs/1804.00218)

HOUDINI is, in 2018, TCN's typed operator algebra plus TCN's recursive module
abstraction:

- Networks are **strongly typed differentiable functional programs**; the type
  system tracks tensor dimensions and distinguishes `bool`, `real`, and
  structured types (lists, graphs).
- A **typed higher-order combinator library**: `map`, `fold`, `conv` over lists
  *and* graph neighbourhoods — the same idea as TCN's aggregation / set /
  temporal operator groups.
- **Type-directed top-down search with holes**, where type inference
  automatically rejects type-unsafe completions.
- **Modules trained on earlier tasks are registered in the library and become
  candidates for later synthesis.**

**The number TCN should cite in its own favour, and also the reason its typing
claim is not novel:** on task CS1, type constraints reduced the candidate set at
program size 6 from **~1.3M (untyped) to 175 (typed)** — a **>7400×**
reduction.

Results (RMSE): CS1 ≈0.38 vs baseline ≳1.0; CS2 ≈0.33 (high-level transfer,
reusing a counting net) vs ≈0.5 for low-level transfer; summing task **2.15 vs
RNN 5.58**; shortest path discovers a Bellman–Ford-like relaxation. Beats
progressive neural networks and standard transfer.

HOUDINI's stated limits, all of which apply to TCN: synthesis up to **22
minutes** (1337 s for shortest path); the **evolutionary search strategy "has
high variance; in many runs… it times out without finding a solution"**;
long-sequence experiments **required capping evaluated programs at 20**;
requires properly typed data with **known target types**.

> **NEAR: Learning Differentiable Programs with Admissible Neural Heuristics.**
> Shah, Zhan, Sun, Verma, Yue, Chaudhuri. **NeurIPS 2020 [PR]**.
> [arXiv 2007.12101](https://arxiv.org/abs/2007.12101)

NEAR is the *correct architectural answer to TerpreT*: DSL program learning as
search in a graph of top-down syntactic derivations, where a neural network
relaxes any partial program, and the relaxed program's **training loss serves as
an approximately admissible heuristic** for A*/branch-and-bound over program
structures. **The relaxation guides discrete search; it never produces the
program.** (Honest caveat in the paper: admissibility is only approximate, so
optimality guarantees do not strictly hold.)

Also directly on TCN's node semantics:
**Differentiable Synthesis of Program Architectures**, Guofeng Cui & He Zhu,
**NeurIPS 2021 [PR]**
([proceedings](https://proceedings.neurips.cc/paper/2021/hash/5c5a93a042235058b1ef7b0ac1e11b67-Abstract.html))
— encodes architecture search as learning a probability distribution over all
program derivations induced by a **context-free grammar**, optimized by gradient
descent in a continuous relaxation of the discrete space of grammar rules.
Evaluated on four sequence-classification tasks. TCN's "softmax over legal
typed operator-and-binding candidates" is this construction with a type system
in place of a CFG.

And on the *heterogeneous analytic operator* claim specifically:

> **Extrapolation and Learning Equations** (EQL), Martius & Lampert, 2016-10-10,
> [arXiv 1610.02995](https://arxiv.org/abs/1610.02995) **[PP/WS ICLR 2017 workshop]**;
> **Learning Equations for Extrapolation and Control** (EQL÷), Sahoo, Lampert,
> Martius, **ICML 2018 [PR]**, [arXiv 1806.07259](https://arxiv.org/html/1806.07259v1)

EQL layers contain heterogeneous units — identity, sine, cosine, multiplication,
and (in EQL÷) division — combined by learned weights with sparsity
regularization, then read out as a symbolic expression. EQL÷ was applied to
**control** (cart-pendulum, robot arm). Read against TCN's §5, EQL÷'s training
procedure is already a crystallization schedule:

1. **Phase 1** (`t < t₁`): no L1 penalty, weights move freely.
2. **Phase 2** (`t₁ ≤ t < t₂`): L1 regularization drives many weights to zero.
3. **Phase 3** (`t ≥ t₂`): weights with `|w| < 0.001` are **frozen at 0** (fixed
   L0), and the survivors are retrained to fit optimally.

Plus a **curriculum-annealed division threshold** `θ(t) = 1/√(t+1)` that starts
conservative and relaxes — because "any division creates a pole at b→0 with an
abrupt change in convexity and diverging function value and its derivative."
This is precisely TCN's §2 requirement that "every evaluated candidate must be
defined over its relaxed inputs," solved by curriculum in 2018.

EQL÷'s honest limits, which are TCN's limits: networks are **L ∈ {2,3,4}
layers** (cart-pendulum needed L=4); the problem is strongly non-convex so
optimization "may get stuck in a local minimum or not select the correct
formula," requiring **10 independent random restarts**; **model selection out of
multiple plausible candidates was the *named* failure of the original EQL** and
required two purpose-built criteria to fix; and even then two target formulas
could not be learned to satisfactory precision and 1 of 10 runs failed on
cart-pendulum.

### 4.5 Learned discretization, QAT, and the DARTS discretization gap

**INQ — Incremental Network Quantization**, Zhou et al., **ICLR 2017 [PR]**,
[arXiv 1702.03044](https://arxiv.org/abs/1702.03044). Partition each layer's
weights by a magnitude/importance measure; **quantize and freeze group 1**;
**retrain group 2 at full precision to compensate**; repeat until all weights
are quantized. Result: improved or comparable accuracy vs full precision at
5/4/3/2-bit on ImageNet. **This is TCN's crystallization loop, in a different
domain, from 2017, and it works.** Transferable lessons: partition by a
confidence/importance measure rather than arbitrarily; always leave a
compensating free population; the schedule is the main hyperparameter.

**Straight-through estimator facts TCN needs.** The STE is a **biased estimator
by construction**. Post-QAT gradients often approach zero, amplifying that bias.
**Oscillation** is a distinct named pathology: latent weights near a
quantization boundary flip between bins and corrupt the final model
(Nagel et al., **ICML 2022 [PR]**, [arXiv 2203.11086](https://arxiv.org/pdf/2203.11086)).
A 2024/25 result argues most "custom" gradient estimators are
[STEs in disguise](https://openreview.net/forum?id=3j72egd8q1) **[PP]**, so a
bespoke surrogate is unlikely to buy much. Recent nuance: evaluating gradients
at the *deployed quantized* weights biases updates toward the low-loss basin
([arXiv 2606.09012](https://arxiv.org/abs/2606.09012) **[PP]**); FOGZO adds
zeroth-order finite-difference directions
([arXiv 2510.23926](https://arxiv.org/pdf/2510.23926) **[PP]**).

**Gumbel-Softmax / Concrete** (both **ICLR 2017 [PR]**) has a two-sided
annealing failure: aggressive annealing causes **premature collapse to a
suboptimal discrete solution**; too-slow annealing leaves an indecisive soft
model whose hardening does not match its soft behavior. Standard mitigations:
entropy regularization and an explicit hardening step.

**The DARTS failure literature is the closest analogue to TCN's core mechanism
and should be read as a checklist of things that will go wrong.**

| Failure | Source |
|---|---|
| **Discretization gap** — supernet validation performance deviates substantially from the discretized child's | DARTS follow-ups, general |
| **Collapse to parameter-free operations** — DARTS chronically collapses to skip-connections; on **NAS-Bench-201 CIFAR-10 it converges to 100% skip-connections**. Fair DARTS diagnoses the cause: under **exclusive softmax competition**, skip has an unfair advantage because it forms a residual that makes *supernet* training easy — a benefit that evaporates on discretization. Fix: independent per-operation **sigmoid** weights + a zero-one loss | Fair DARTS, **ECCV 2020 [PR]**, [arXiv 1911.12126](https://arxiv.org/abs/1911.12126) |
| **Hessian eigenvalue signature** — the dominant eigenvalue of the Hessian of validation loss w.r.t. architecture parameters correlates strongly with the discretized architecture's generalization error; DARTS-ES stops when it rises sharply. Fixes DARTS' failure modes across all 12 constructed benchmarks | R-DARTS, **ICLR 2020 [PR]**, [arXiv 1909.09656](https://arxiv.org/pdf/1909.09656) |
| **Argmax over learned weights is the wrong selection rule** | DARTS-PT, **ICLR 2021 Outstanding Paper [PR]**, [arXiv 2108.04392](https://arxiv.org/abs/2108.04392) |
| **The whole search may be no better than random** — on average SOTA NAS performs similarly to a random policy, and weight sharing degrades candidate ranking to the point of not reflecting true performance | Yu et al., **ICLR 2020 [PR]**; Li & Talwalkar, [arXiv 1902.07638](https://arxiv.org/pdf/1902.07638) |

**DARTS-PT deserves its own paragraph, because it is the most damaging single
paper for TCN's crystallization design.** Wang, Cheng, Chen, Tang, Hsieh
(ICLR 2021, Outstanding Paper Award) show **empirically and theoretically that
"the magnitude of architecture parameters does not necessarily indicate how much
the operation contributes to the supernet's performance."** Their replacement is
**perturbation-based selection**: score each operation by the *actual* drop in
supernet validation accuracy when it is perturbed/removed, discretize one edge
at a time, and let the supernet recover in between. This took DARTS from
**3.00% → 2.61%** test error at 0.8 GPU-days and "greatly alleviated" several
DARTS failure modes. Their conclusion is that **much of DARTS' poor
generalization is a failure of the selection rule, not of supernet
optimization.**

TCN's §5 currently measures "candidate entropy, selection stability" and then
"gradually concentrates candidate choices" — i.e. it selects by confidence in
the learned distribution, which is exactly the rule DARTS-PT falsified. TCN's
step 3 ("temporarily harden a candidate node/block and measure degradation") is
*already* perturbation-based and is the right idea; the fix is to make that the
**selection** criterion, not merely the **acceptance** criterion.

**And the sequential crystallization schedule already exists**: **SGAS —
Sequential Greedy Architecture Search**, Li et al., **CVPR 2020 [PR]**,
[arXiv 1912.00195](https://arxiv.org/abs/1912.00195), greedily selects and
prunes candidate operations sequentially, explicitly motivated by the
search/evaluation gap, and validates the mechanism with **Kendall-τ correlation
between search-phase and evaluation-phase rankings**. **P-DARTS** (ICCV 2019)
is the depth-gap precedent. Neither eliminates the gap.

### 4.6 One more: probes

TCN §7 makes probe readouts a first-class supervision interface and already
warns that probe accuracy must be checked against causal use. The canonical
citation is **Hewitt & Liang, "Designing and Interpreting Probes with Control
Tasks", EMNLP 2019 [PR]**
([ACL](https://aclanthology.org/D19-1275/), [PDF](https://nlp.stanford.edu/pubs/hewitt2019control.pdf)):
high probe accuracy may mean the *probe* learned the task, not that the
representation encodes it. Their remedy is **selectivity** — task accuracy minus
accuracy on a control task with randomized labels. See also
[Probing Classifiers: Promises, Shortcomings, and Advances](https://direct.mit.edu/coli/article/48/1/207/107571/)
(Belinkov, *Computational Linguistics* 48(1), 2022 **[PR]**). Report selectivity
for every TCN probe, not raw probe loss.


---

## 5. The honest delta

TCN's ARCHITECTURE.md makes four substantive claims. Against the literature
above, here is what is genuinely novel, what is already done, and what is
already known to fail.

### 5.1 Typed heterogeneous operator synthesis

**Already done, twice, and the pruning benefit is already published.**

- **HOUDINI** (NeurIPS 2018) is strongly typed differentiable functional
  programs with a typed higher-order combinator library, type-directed search
  that automatically rejects type-unsafe completions, and neural module reuse
  across tasks. Its **1.3M → 175 candidates at program size 6** is the number
  TCN would want to cite — and it is HOUDINI's number.
- **EQL / EQL÷** (2016 / ICML 2018) is heterogeneous analytic operator selection
  (identity, sin, cos, multiply, divide) with sparsity-driven selection, applied
  to control, with a curriculum-annealed guard on division's pole — i.e. TCN's
  §2 requirement that "every evaluated candidate must be defined over its
  relaxed inputs," solved in 2018.
- **Differentiable Synthesis of Program Architectures** (NeurIPS 2021) is
  "learn a probability distribution over derivations of a grammar and optimize
  it by gradient descent in a continuous relaxation." TCN's node semantics is
  that construction with a type system in place of a CFG.

**What is actually new in TCN's version:** (a) the *combination* of a typed
heterogeneous library with a **jointly learned operator-and-input-binding
distribution per node** — HOUDINI enumerates symbolically, EQL has fixed
layer-local wiring, dPads relaxes grammar rules rather than bindings; (b) the
insistence on **representation** as a first-class axis separate from semantic
type (ARCHITECTURE §1), which nothing in this literature does; (c) the
constitutional rule that **no domain gets a private primitive**, which — as §2.1
shows — the DLGN state of the art violates to obtain its headline number.

**What is already known to fail, and will bite:**
- **Exclusive softmax over operators of unequal optimization difficulty
  collapses to the cheap parameter-free ones.** Fair DARTS' diagnosis is exact
  and transfers directly: identity, copy, and aggregation operators are TCN's
  skip-connections. In deep IALGN, **88.4% of gates chose pass-through.**
  Instrument for this explicitly; the fix is independent sigmoids + a zero-one
  loss, not a better softmax.
- **The 16-way softmax is tractable only because 2^(2²) = 16.** Ternary logic is
  3^(3²) = 19,683 and the field abandoned softmax for it. TCN's candidate set is
  `|legal operators| × |legal bindings|`, a larger and more heterogeneous
  object. **Typing is what keeps this finite** — which is TCN's best argument,
  and also means TCN's Track 3 must report candidate-set sizes, not just
  accuracy.
- **Bounded candidate pools are correct; unbounded is a trap.** BitLogic:
  full-layer learnable connectivity **mode-collapses 7–8 pp below every
  bounded-candidate variant**, with **k = 8–16** best. TCN's "bounded
  predecessor pools" is on the right side of this. Use k in that range as the
  starting point rather than rediscovering it.
- **Fan-in may matter more than the relaxation.** BitLogic: once arity is
  matched, "the relaxation family matters much less than the literature
  suggests." A TCN ablation on operator arity is cheap and may be more
  informative than one on relaxation choice.

### 5.2 Hierarchical supervision interfaces

**The weakest novelty claim, and the one most likely to be waved away.**

Deep supervision (DSN, 2015), auxiliary tasks in RL (UNREAL, 2017), and
goal-conditioned value approximation (UVFA, 2015 — already cited in LESSONS.md)
cover the mechanism. The known failure modes are well documented: **negative
transfer** when an auxiliary task is misaligned, **auxiliary tasks slowing
learning if weighted too strongly**, and **gradient conflict** in any weighted
sum of objectives (PCGrad, CAGrad). ARCHITECTURE §8's instruction to "track
actual return and gradient conflicts because a weighted sum does not guarantee
alignment" is correct and should cite that literature.

**What is genuinely different:** the *typed, explicitly-scoped* supervision
record (`source, target_or_probe, allowed_graph_region, semantic_type,
representation, time_alignment, visibility, loss`) is more disciplined than
anything in the auxiliary-task literature, and the separation of probe from
target from input is a real engineering contribution. But it is an interface
contribution, not a learning result.

**Non-negotiable:** report **probe selectivity** (Hewitt & Liang, EMNLP 2019),
not raw probe loss. A probe that reads a value the program never causally uses
is exactly the failure ARCHITECTURE §7 warns about, and selectivity is the
standard instrument for it.

### 5.3 Progressive crystallization with rollback

**The strongest novelty claim — but narrower than the document implies, and it
now has a direct competitor from four months ago.**

Already published:
- **INQ** (ICLR 2017): quantize-and-freeze a confident group, retrain the rest
  to compensate, iterate. Improved or comparable to full precision at 2–5 bits
  on ImageNet.
- **SGAS** (CVPR 2020): sequential greedy selection and pruning of candidate
  operations, motivated explicitly by the search/evaluation gap, validated by
  **Kendall-τ between search and evaluation rankings**.
- **DARTS-PT** (ICLR 2021, **Outstanding Paper**): discretize one edge at a
  time, let the supernet recover in between, and — critically — **select by
  measured perturbation impact, not by learned weight magnitude**, because "the
  magnitude of architecture parameters does not necessarily indicate how much
  the operation contributes to the supernet's performance." 3.00% → 2.61%.
- **EQL÷** (ICML 2018): three-phase schedule — free training, then L1, then
  **freeze near-zero weights at exactly 0** and retrain survivors.
- ⭐ **CompactLogic** (2026, ETH SRI): **progressive layer-wise discretization
  inside a DLGN** — freeze each layer as it converges, shallow to deep.
  **+2.2 pp on CIFAR-10 and 1.8× faster training.**
- ⭐ **DiPRL** (2026-05, preprint): names TCN's problem verbatim — post-hoc
  discretization "can discard optimized branches and parameters… results in a
  collapse of policy expressivity" and after another 50% of training
  "performance is never recovered" — and solves it **without any hardening
  schedule**, by annealed architecture-entropy regularization that leaves the
  policy already near-discrete (entropy 0.000–0.004 vs π-PRL's 0.284–0.861).
- ⭐ **Mind the Gap** (NeurIPS 2025): solves the DLGN discretization gap with
  **Gumbel noise + straight-through**, achieving a **98% gap reduction while
  making training 4.5× cheaper** — not more expensive.

**What is left that is genuinely TCN's:**
1. **Transactional rollback.** Snapshot, trial-harden, measure degradation
   against declared thresholds across *multiple* objectives and interfaces,
   restore and defer on failure. INQ, SGAS and CompactLogic all freeze
   monotonically; DARTS-PT selects by perturbation but does not roll back a
   committed decision. **I located no work that snapshots and reverts a
   discretization decision.**
2. **The gradient-connectivity guard** — deferring a freeze that would
   disconnect learning signal to the interior. `docs/VALIDATION.md` records this
   firing four times in the joint experiment. This is a real, specific
   mechanism I found nothing equivalent to.
3. **Per-node rather than per-layer granularity**, over a *heterogeneous* typed
   library rather than a homogeneous gate set.

**But note the shape of the competition.** The field's two best answers to the
discretization gap — Gumbel+STE and entropy regularization — both make training
**cheaper** and require **no scheduler at all**. TCN's crystallization is a
scheduler with snapshots, trial hardening, and rollback: it is strictly more
expensive. **Track 1 must therefore beat not just naive argmax but also (a)
Gumbel-noise + straight-through, and (b) an entropy-regularization schedule, on
matched compute.** Beating argmax alone will not survive review.

### 5.4 Recursive module abstraction

**Already done, and the ways it fails are already measured.**

- **HOUDINI** (2018): modules trained on earlier tasks are registered in the
  library and become candidates for later type-directed synthesis. Measured
  transfer gains.
- **Differentiable Programs with Neural Libraries / NTPT** (Gaunt et al., 2016):
  a differentiable interpreter with a reusable library of learned neural
  functions transferring across tasks — by the TerpreT authors.
- **DreamCoder** (PLDI 2021): the ablation TCN wants — **no library learning
  never exceeds 60% of held-out tasks where full DreamCoder solves nearly
  100%.** That is the strongest existing evidence for TCN's Track 5 hypothesis.
- **NPI** (ICLR 2016): key-value program memory; learning to compose lower-level
  learned programs into higher-level ones.

**What is genuinely TCN's:** the abstracted module is **frozen, discrete, and
has no internal gradients**, yet is registered as a typed candidate inside a
*differentiable* search. HOUDINI's modules stay neural and differentiable;
DreamCoder's library is symbolic inside a symbolic search. **A non-differentiable
callable competing in a softmax over relaxed candidates is, as far as I can
tell, unclaimed** — and ARCHITECTURE §5's "frozen blocks can supply forward
values and targets to later training without becoming differentiable again" is
the interesting technical content.

**Already known to fail — take these seriously:**
- **A newly registered abstraction is a net loss by default.** LILO: raw STITCH
  abstractions handed to the solver scored **−30.60 REGEX, −11.11 LOGO, −2.91
  CLEVR**. Only after auto-generated names and docstrings did they become
  net-positive. **TCN's typed signature is not enough**; a module needs a
  semantic description the downstream searcher can use.
- **Library-learning gains can vanish entirely under compute-matching.** The
  LEGO-Prover case study found the gains disappear once computational cost is
  accounted for, with **no evidence of direct reuse and evidence contradicting
  indirect reuse.** Track 5 must report compute-matched baselines *and*
  behavioral evidence that abstracted modules are actually selected.
- **Symbolic compression beats learned abstraction by 3–4 orders of magnitude.**
  STITCH is 1000–10000× faster than DreamCoder's compression with equal or
  better libraries. **Do not build a differentiable abstraction-discovery
  mechanism.**
- **Abstraction bloat is worse for TCN than for enumerative systems.** Every
  registered operator widens the per-node candidate set. In DreamCoder an unused
  abstraction is dead weight; **in TCN it is a permanent tax on every node's
  softmax.**

### 5.5 Where a skeptical reviewer pushes hardest

Ordered by how much damage each does.

1. **"TerpreT already answered this. Where is your constraint-solver
   baseline?"** — *Our key empirical finding is that constraint solvers dominate
   the gradient descent and LP-based formulations.* Built by the differentiable-
   interpreter authors, on exactly this task. The DLGN line has never answered
   the equivalent objection either (*"circuit synthesis is a really
   well-researched field… have you tried the IWLS competition data sets?"*).
   **A SAT/SMT or enumerative synthesis baseline over the identical typed
   operator library is not optional.** Without it, TCN's central claim is
   untested against the known-superior alternative.
2. **"Your selection rule was falsified at ICLR 2021."** ARCHITECTURE §5 selects
   by candidate entropy and choice concentration. DARTS-PT (Outstanding Paper)
   shows empirically and theoretically that the magnitude of a learned
   architecture distribution does not indicate an operation's contribution.
   TCN's step 3 already *measures* degradation — **make that the selection rule,
   not just the acceptance test.** This is a small change with a large defensive
   payoff.
3. **"Depth does not pay off in this substrate, and your whole thesis is
   depth."** An et al.: a standard DLGN **loses 2.65 pp going from 48k to 1.8M
   gates via depth**, with 88.4% of deep gates doing nothing. Four independent
   groups report shallow-and-wide beating deep. TCN's answer must be that
   **typed structured wiring fixes the topology-induced credit degradation** —
   and that has to be *shown*, on a depth sweep, not asserted.
4. **"Is crystallization necessary, or is Gumbel+STE enough?"** Mind the Gap
   reduces the DLGN discretization gap by 98% while making training 4.5×
   *cheaper*; DiPRL removes post-hoc discretization entirely with entropy
   regularization. Both are simpler than a scheduler with snapshots and
   rollback. **Track 1's baseline set must include both.**
5. **"Your evidence is two nodes with real operator choice."**
   `research/AGENDA.md` records that in the only successful joint experiment,
   exactly two nodes carry real choice (16 truth tables each), every other node
   is single-candidate, the value function is a trainable constant, and held-out
   "episodes" vary input bits rather than structure. `docs/VALIDATION.md`
   records a **four-operation** learned program. **Against a literature whose
   working configurations are 4–6 layers of 8k–1M gates, and whose failure
   analyses concern 61M-gate models, four operations is not yet on the chart.**
   The honest framing is "mechanism verified, capability unestablished" — which
   VALIDATION.md already says, and which every public document should keep
   saying.
6. **"The arithmetic scaffold stayed at chance."** 2.03/4 after 640 episodes on
   a depth-1 truth table, in a scaffold with almost no operator choice. As
   AGENDA.md notes, that is more likely a training-dynamics fault than evidence
   about crystallization. The literature says exactly where to look: **learning
   rate** (Mommen et al.: 98% → 15% across depths at the default lr), **τ**
   (25–40 point swings), and **initialization cancellation** in a uniform
   mixture (the 16-gate softmax nullspace pathology). Track 2 should test those
   three before anything architectural.
7. **"Programs win on transfer and cost, not on score — so why is your headline
   capability?"** PIRL is slower than its own teacher; DSP needs 2M episodes for
   CartPole; DWC loses 34% on HalfCheetah; a purely logical Atari policy is
   0.5–9% of DQN. The durable wins in this literature are **crash ratios
   (0.04 vs 0.24–0.92), verified properties (39/50 vs 0/50), energy (nJ/action),
   and OOD transfer.** ARCHITECTURE's framing — "low cost and latency must be
   demonstrated at useful end-to-end capability" — sets a bar the field has not
   cleared. Consider leading with cost and verifiability.
8. **"Your generalization advantage may be a baseline artifact."** *Common
   Benchmarks Undervalue the Generalization Power of Programmatic Policies*
   shows TORCS DRL going from 0% to 69–100% OOD success from a reward change
   alone. Track 6's matched-information baseline must be tuned as hard as TCN
   is.
9. **"Show me seed variance."** NLM graduated 40% of 10 seeds; EQL÷ needed 10
   random restarts and still failed 1 in 10; Neural DNF-MT fails 6 of 32 runs.
   Single-seed results in this field are not evidence.
10. **"Is the substrate patented?"** WO 2023143707 A1 / US 18/883,354 /
    TW 112101045. A plain relaxed-gate-choice layer plausibly reads on it.


---

## What to steal and what to avoid

Concrete and actionable for this codebase. Ordered by expected value.

### Steal

**S1. Perturbation-based selection, not confidence-based selection.** *(Track 1,
`tcn/` crystallization scheduler — highest value item in this report.)*
ARCHITECTURE §5 step 1 measures "candidate entropy, selection stability" and
step 2 "gradually concentrates candidate choices." DARTS-PT (ICLR 2021
Outstanding Paper) shows that is the wrong criterion: the magnitude of a learned
architecture distribution does not indicate an operation's contribution. Step 3
already trial-hardens and measures degradation — **promote that from acceptance
test to selection rule.** Crystallize the candidate whose ablation most degrades
the block, not the one with the highest probability. Small change, large
defensive payoff, and it is directly testable in Track 1.

**S2. Add Gumbel noise + a straight-through estimator to the soft phase.**
*(Track 1 baseline set, and probably the default trainer.)* Mind the Gap
(NeurIPS 2025) reports a **98% discretization-gap reduction and 4.5× faster
wall-clock training** on DLGNs. CAGE's decomposition suggests the operative
ingredient is the **hard forward pass**, not the noise (Hard-ST 0% selection gap
vs Soft-Mix 24%). Try hard-forward/soft-backward first; it is a few lines.
Note the field disagrees on the mechanism, so ablate both.

**S3. Report a rank correlation between soft-phase scores and post-hardening
performance.** *(Track 1.)* SGAS validated sequential greedy discretization with
**Kendall-τ between search-phase and evaluation-phase rankings.** If TCN's τ is
near zero, crystallization is not working no matter how good the final number
looks. This is the single most informative diagnostic available and it is cheap.

**S4. Use `k = 8–16` bounded predecessor pools, and say why.** BitLogic
(TMLR 2026) finds full-layer learnable connectivity **mode-collapses 7–8 pp
below every bounded-candidate variant**, with k = 8–16 best. ARCHITECTURE
already specifies "bounded predecessor pools" — set the bound in that range as
the informed default rather than rediscovering it in Track 3.

**S5. Ablate operator *arity* before ablating the relaxation.** BitLogic: once
fan-in is matched, "the relaxation family matters much less than the literature
suggests," and the n=2 → n=4 step explains most of DiffLogic's gap. TCN has
heterogeneous arity by construction. A cheap arity sweep may be more informative
than anything about relaxation choice.

**S6. Fix Track 2 (the arithmetic scaffold at chance) with the three knobs the
literature identifies, before touching the architecture.** In order:
(a) **learning rate** — Mommen et al. report MNIST going 98% / 82% / 45% / 32% /
16% / 15% across depths 1–6 at the default lr, fixed by lr=0.1, and call it "by
far the biggest impact"; (b) **temperature** — 25–40 point swings from τ alone,
with the optimum moving by dataset and output count; (c) **initialization
cancellation** — the 16-gate softmax has a proven nullspace pathology where the
backward signal vanishes exactly at uniform init.
`examples/joint.py` already copies a predecessor to dodge this; the arithmetic
scaffold may not.

**S7. Do abstraction discovery symbolically.** *(Track 5.)* STITCH (POPL 2023) is
**1000–10000× faster than DreamCoder's compression with equal or better
libraries**. Do not build a differentiable abstraction-discovery mechanism.
Compress over the corpus of already-crystallized programs.

**S8. Auto-document every registered module.** *(Track 5.)* LILO measured raw
abstractions handed to a downstream solver at **−30.60 REGEX, −11.11 LOGO** —
they only became net-positive with generated names and docstrings. A typed
signature is not a semantic description. If TCN's module registry does not carry
a behavioral summary the search can condition on, expect registration to hurt.

**S9. Cite HOUDINI's 1.3M → 175 pruning result as the justification for typing.**
It is the strongest published number in TCN's favour, it is free, and citing it
prominently is also the honest way to handle the fact that HOUDINI got there
first.

**S10. Reframe the headline from capability to cost, verifiability and
transfer.** PIRL is slower than its own teacher; DWC loses 34% on HalfCheetah;
purely logical Atari policies reach 0.5–9% of DQN. What survives in this
literature is **crash ratio 0.04 vs 0.24–0.92 (PROPEL)**, **39/50 vs 0/50
properties verified (binary policies)**, **1–2 nJ per action (DWC)**, and OOD
transfer. TCN's existing frozen-program artifacts (`program.pyz`, exact
conformance, replay determinism) are already aimed at exactly these.

### Avoid

**A1. Do not claim TCN synthesizes programs by gradient descent without a
constraint-solver baseline.** TerpreT: *"constraint solvers dominate the gradient
descent and LP-based formulations."* The DLGN line has never answered the same
objection in its own field (*"circuit synthesis is a really well-researched
field… have you tried the IWLS data sets?"*). **A SAT/SMT or enumerative
baseline over the identical typed operator library is the single most important
missing experiment in `research/AGENDA.md`.** Consider adding it as Track 8.

**A2. Do not let Track 1's baseline be naive argmax alone.** The field's two best
answers to the discretization gap — Gumbel+STE and annealed architecture-entropy
regularization (DiPRL) — both make training **cheaper** and need **no scheduler
at all**. Crystallization with snapshots, trial hardening and rollback is
strictly more expensive. It must beat both, on matched compute, or the mechanism
is not justified.

**A3. Do not assume depth will pay off — measure it early.** A standard DLGN
**loses 2.65 pp going from 48k to 1.8M gates via depth**, with 88.4% of gates in
the deep circuit doing nothing, and four independent groups report
shallow-and-wide beating deep. TCN's whole thesis is depth via composition. The
plausible TCN answer is that **typed structured wiring fixes the
topology-induced credit degradation that random wiring causes** — that is a good
hypothesis and a publishable one, but it has to be shown on a depth sweep with
seeds, not assumed.

**A4. Expect and instrument for collapse to cheap parameter-free operators.**
Fair DARTS: exclusive softmax competition gives skip-connections an unfair
advantage because they make the *soft* network easy to optimize, a benefit that
evaporates on discretization — on NAS-Bench-201 CIFAR-10, DARTS converges to
**100% skip-connections**. TCN's identity, copy, projection and aggregation
operators are structurally the same hazard. Log the selected-operator
distribution every run. If it collapses, the fix is independent sigmoids plus a
zero-one loss, not a better softmax.

**A5. Do not report single-seed results.** NLM graduated **40% of 10 seeds**;
EQL÷ needed 10 random restarts and still failed 1 in 10; Neural DNF-MT fails 6
of 32 runs; SCoBots fails 1 of 3. `docs/VALIDATION.md`'s joint result should
carry a seed count and variance before it is cited anywhere.

**A6. Do not report probe accuracy without selectivity.** Hewitt & Liang
(EMNLP 2019). ARCHITECTURE §7 already states the concern; selectivity is the
instrument.

**A7. Do not let the module library grow unbounded.** Every registered operator
widens the per-node candidate set permanently. In DreamCoder an unused
abstraction is dead weight; **in TCN it is a tax on every node's softmax
forever.** Gate registration on a measured downstream-utility test, and report
the candidate-set size alongside accuracy in Track 3 and Track 5.

**A8. Do not claim the recursive-abstraction win without compute-matched
baselines and behavioral evidence of reuse.** The LEGO-Prover case study found
apparent library-learning gains **vanish entirely once compute is accounted
for**, with no evidence of direct reuse. Track 5 must show that crystallized
modules are *actually selected* by later synthesis and that removing them
degrades results.

**A9. Do not repeat the Mario claim.** Update `docs/LESSONS.md` from "not
identified in the initial search" to a positive negative finding, and name the
conflation candidates so the question stays closed. The motivation is still
sound — it just points at DWC on MuJoCo, not at Mario.

**A10. Do not build on `difflogic` as a dependency.** Last pushed
**2024-03-19**; the NeurIPS 2024 convolutional code has never been released
despite four open requests; chronic CUDA/torch build breakage; the method is
patent-pending (WO 2023143707 A1). TCN's decision to own its implementation
locally is, on this evidence, correct.

### Two experiments that would make TCN publishable

1. **Typed wiring converts depth into accuracy.** The field's clearest open
   problem is that randomly-wired logic networks gain nothing from depth
   (An et al., 2026) and that the diagnosed cause is topology-induced credit
   degradation. TCN's typed, structured candidate bindings are a principled
   answer. A depth sweep showing TCN's typed graphs scale where random wiring
   does not would be a genuine contribution to an active question — and it is
   measurable at small scale.
2. **Rollback earns its cost.** No work I located snapshots and *reverts* a
   discretization decision, and no work implements a gradient-connectivity guard
   on freezing. Both are TCN's. Show them beating Gumbel+STE and
   entropy-regularization on matched compute, with Kendall-τ and seed variance
   reported, and Track 1 answers a question the field has not asked.
