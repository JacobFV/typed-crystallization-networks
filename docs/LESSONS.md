# Lessons from the local review

This is supporting evidence for the [architecture](../ARCHITECTURE.md), not
another specification. The initial review was read-only. Code observations below were
inspected directly; historical experimental findings are reports from the
existing notes and were not reproduced in this session. The implementation now copies and refactors language, computer, rendering, and
physics mechanisms locally; these review lessons remain the rationale for its
contracts and checks.

## Failure modes to carry forward

| Evidence inspected | Implication for TCN |
|---|---|
| The agent architecture notes describe separate computer and robot action schemas/loops that later required consolidation. They also record that apparently duplicate UI components carried different behavior. [Agent architecture notes](../../commandagi/code-agent-memories/unified-agent-architecture.md) | One typed action/transition contract from the start. Preserve behavior while extracting useful code; superficial similarity is not evidence that two mechanisms are interchangeable. |
| A multimodal event interface uses `payload: Any`, with labels potentially in metadata and token status inferred from iterability and array dimensionality. [Event definition](../../synthEX/synthetic_worlds/experience/core.py) | A uniform envelope alone does not establish uniform semantics. Require explicit typed representations and visibility; do not guess semantic type from storage shape. |
| Language generation already constructs structured content before realization, keeps hidden state separate from the prompt, and separates lesson definitions from curriculum order. [Generation contract](../../symbolic-ai-models/symbolic_ai_data/lesson.py) | Extract these mechanisms. Replace the text-specific external record with the common generator record while retaining exact symbolic construction and optional supervision. |
| Generator rename notes identify a package-looking string that actually domain-separates a seeded permutation. Renaming it would change samples. [Addressing note](../../symbolic-ai-models/symbolic_ai_data/_vendor.py) | Refactoring copied code must distinguish cosmetic names from semantic constants. Establish local fingerprints and explicitly version intended distribution changes. This does not require preserving the old package architecture. |
| The symbolic project's approach documents boilerplate-dominated metrics, teacher-forcing gaps, and misleading component-only throughput claims. [Measurement lessons](../../symbolic-ai-models/APPROACH.md) | Score actual answers/actions, held-out structures, and free-running behavior. Measure full-path capability and cost; do not count a fast symbolic kernel as a complete language system. |
| An audit counted declared node types that never appeared in its sampled curriculum. [Type coverage note](../../symbolic-ai-models/notes/questions/2026-08-14-five-of-seventeen-node-types-are-dead.md) | Every supported semantic claim needs exercising examples. Separate declared, exercised, and measured support. A growing catalog is not growing capability. |
| A type-based gate treated failure to infer a type as evidence that a candidate was impossible. [Unknown-versus-false note](../../symbolic-ai-models/notes/lessons/2026-08-14-no-information-is-not-negative-evidence.md) | Hard exclusion requires a known incompatibility. Inference failure is a separate diagnostic; it cannot manufacture a semantic rejection or an observable label. |
| A task audit confused observations being different with a learnable observation-to-label mapping. [Identifiability note](../../symbolic-ai-models/notes/lessons/2026-08-19-distinguishable-is-not-identifiable.md) | Exact hidden truth does not imply that an actor can recover it. Probe targets must account for ambiguity and partial observability. |
| The computer fidelity ledger records actions that looked supported but lacked the claimed state effects, along with metadata fields that lacked runtime consumers. [Fidelity ledger](../../synthetic-computer-environment/docs/fidelity-acceptance-spec.md) | Verify that UI input, stored state, rendering, and reward inspect one causal transition. Interface/schema coverage does not prove execution semantics. |
| Computer runtime initialization, installation timestamps, and snapshots read host time; trajectory records also timestamp with `new Date()`. [Runtime](../../synthetic-computer-environment/packages/kernel/src/simulation.ts), [trajectory recorder](../../synthetic-computer-environment/packages/kernel/src/trajectory.ts) | Seeded network randomness does not make the entire episode replayable. Copying this implementation requires explicit logical clocks at all semantic consumers and complete state restoration. Stripping timestamps from a digest is insufficient. |
| The tensor-computer status reports a previously inverted straight-through estimator and divergences found by comparing independent executors. The current helper uses the corrected expression. [Status](../../tensor-computer/STATUS.md), [gradient helper](../../tensor-computer/packages/tc-common/src/tc_common/tensor.py) | Check hard-forward values and surrogate derivatives separately during trials; compare exact and exported execution. These checks establish correctness of the mechanism, not successful synthesis. |
| The symbolic project's notes report an apparently general reader failing on natural prose containing constructions absent from its own generated text. [Transfer lesson](../../symbolic-ai-models/notes/questions/2026-08-19-every-evaluation-here-runs-on-text-this-repo-wrote.md) | Keep this project's initial training fully synthetic as requested, while limiting conclusions to its distributions. Natural transfer requires its own later evidence. |

## Extraction decisions for implementation

Copy the language construction, grammar realization, seeded sampling, and
useful invariance checks into the appropriate local generator. Refactor external
records into the shared typed schema. Keep generated symbols/content synthetic
and inspect any real-corpus or knowledge-source path before enabling it. Domain
labels remain generator data and supervision, not agent opcodes.

Copy the computer state-transition mechanisms needed for the chosen curriculum:
files, processes, input effects, application state, and rendering state bindings.
Make local core contracts authoritative; replace conflicting lifecycle, clock,
and record semantics during extraction. Preserve the causal checks that expose
actual effects. A private cross-language transport is acceptable if needed for
the copied implementation, but it must carry this repository's contract rather
than perpetuate a second public architecture.

Use the other projects' mechanisms only where they serve those contracts. There
is no need to reproduce their service graphs, model registries, UI shells, or
ontology hierarchies. The end state is code owned and maintained in this repo,
with no runtime dependency on the old checkouts. These links document review
evidence; they are not proposed dependencies or an attribution system.

## Primary research context

The original differentiable logic-gate work learns relaxed gate choices and
discretizes the resulting network. It supports the Boolean starting mechanism;
TCN's typed heterogeneous synthesis, hierarchical interfaces, and recursive
module crystallization are proposed extensions, not results established by that
paper. [Deep Differentiable Logic Gate Networks](https://arxiv.org/abs/2210.08277).

Straight-through estimation is a training estimator for hard operations, with
its own limitations. It does not supply an ordinary derivative of a discrete
program, and TCN's final frozen modules deliberately have no internal gradients.
[Estimating or Propagating Gradients Through Stochastic Neurons](https://arxiv.org/abs/1308.3432).

Goal-conditioned value approximation provides relevant precedent for training
across explicit goals. World-model learning provides precedent for coupling
prediction with behavior optimization. Neither establishes that the proposed
TCN architecture will learn or transfer successfully.
[Universal Value Function Approximators](https://proceedings.mlr.press/v37/schaul15.html),
[Mastering Diverse Domains through World Models](https://arxiv.org/abs/2301.04104).

The particular Mario demonstration mentioned in the brief was not identified
by literature search. The project lead reports it was posted to Twitter/X on
2026-09-07, which is after this repository's search methods can reach: those
queries covered indexed papers, GitHub, blogs, and author publication lists,
none of which index a day-old social media post. The null finding recorded in
`research/literature/RESULTS.md` should be read as "not present in indexed
sources", not as "does not exist".

Two things still need capturing when the source is to hand: the architecture
and discretization method, and the training procedure. Both bear directly on
whether this repository's substrate is the right one for sequential control.
The nearest result that search did locate is Differentiable Weightless
Controllers, which trains Boolean lookup-table circuits with SAC to
competitive performance on four of five MuJoCo tasks.
