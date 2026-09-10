# tcn demo

```
demo         measured                                                           baseline                                                                                     verdict
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
synthesis    held-out max |err| 5.9e-08 on 584 unfitted points                  enumeration: unique of 96 in 2.7 ms (gradient 1 s, same program); matched MLP 1.8e-02        PASS
joint        frozen program 4.00/4, prediction loss 0.24884 -> 0.00223          best constant 2.38/4 over 64 episodes (3.00/4 over the 16 the record scores), uniform rando… PASS
structure    unseen pool 4.00/4 (seen 4.00/4), interpreter candidate chosen fr… best constant 2.56/4; the recorded scaffold measures 1.95-2.03 and cannot exceed 2.00        PASS
depth        exact frozen program d1 4.00, d2 4.00, d3 4.00, d4 4.00, d6 4.00,… best constant d1 2.12, d2 2.12, d3 2.38, d4 2.50, d6 2.06, d8 2.44                           PASS
abstraction  arm B conformant at step 20, module on output path True, 3 live n… arm A (no module) and arm C (distractor) both fail; the flat space of 230,400 programs is e… PASS
positional   2,304 positions over a 6,912-value observation with 3 caller node… 17 structural symbols shared against 109 per-position and 257 inlined at 32 positions, at 0… PASS
segmentation held-out max |err| 0.0 on 48 unseen episodes                       unique among 65,536 programs in 2.2 s; constant predictor 0.854 accuracy                     PASS
edge         held-out max |err| 0.0, accuracy 1.000                             unique among 48 staged programs in 0.5 s; undecomposed 4.9e10 programs (7.6 years projected… PASS
control      replay and restore identical, max |state delta| 0 after 8 further… scripted controller reaches upright 0.9994 where zero torque never exceeds -1.0000           PASS
language     post-audit stream (hardening='context_free_language', the shipped… post-audit majority constant 0.5262, random 0.500, best fitted feature 0.4738, training-str… PASS
```

- **synthesis** — exact typed program from 16 examples, exported and checked
  - measured: held-out max |err| 5.9e-08 on 584 unfitted points
  - baseline: enumeration: unique of 96 in 2.7 ms (gradient 1 s, same program); matched MLP 1.8e-02
  - evidence: research/baselines/RESULTS.md §3; research/enumerative-baseline/RESULTS.md §0
  - reproduce: `tcn demo --only synthesis`

- **joint** — goal-conditioned control and latent prediction, exact frozen agent
  - measured: frozen program 4.00/4, prediction loss 0.24884 -> 0.00223
  - baseline: best constant 2.38/4 over 64 episodes (3.00/4 over the 16 the record scores), uniform random 1.91/4, oracle 4.00/4
  - evidence: research/baselines/RESULTS.md section 4; research/crystallization-ablation/RESULTS.md
  - reproduce: `tcn demo --only joint`

- **structure** — held-out Boolean gate families, interpreter candidate not supplied
  - measured: unseen pool 4.00/4 (seen 4.00/4), interpreter candidate chosen from 17: True
  - baseline: best constant 2.56/4; the recorded scaffold measures 1.95-2.03 and cannot exceed 2.00
  - evidence: research/nondegenerate-generalization/RESULTS.md
  - reproduce: `tcn demo --only structure`

- **depth** — trained at depths 1-2 only, scored at depths never trained on
  - measured: exact frozen program d1 4.00, d2 4.00, d3 4.00, d4 4.00, d6 4.00, d8 4.00
  - baseline: best constant d1 2.12, d2 2.12, d3 2.38, d4 2.50, d6 2.06, d8 2.44
  - evidence: research/depth-generalization/RESULTS.md
  - reproduce: `tcn demo --only depth`

- **abstraction** — a learned sub-program reused, on a scaffold below the flat minimum
  - measured: arm B conformant at step 20, module on output path True, 3 live nodes
  - baseline: arm A (no module) and arm C (distractor) both fail; the flat space of 230,400 programs is exhausted with no solution, and 144 of the 144 solutions in 2,709,504 use the module
  - evidence: research/recursive-abstraction-retest/RESULTS.md
  - reproduce: `tcn demo --only abstraction`

- **positional** — one frozen module at every position of a wide observation
  - measured: 2,304 positions over a 6,912-value observation with 3 caller nodes, exact apply correct (38 s)
  - baseline: 17 structural symbols shared against 109 per-position and 257 inlined at 32 positions, at 0.91x the execution cost
  - evidence: research/positional-reuse/RESULTS.md
  - reproduce: `tcn demo --only positional`

- **segmentation** — background colour recovered from raw pixels over the full byte alphabet
  - measured: held-out max |err| 0.0 on 48 unseen episodes
  - baseline: unique among 65,536 programs in 2.2 s; constant predictor 0.854 accuracy
  - evidence: research/perception-ladder/RESULTS.md §4
  - reproduce: `tcn demo --only segmentation`

- **edge** — a learned two-position operator over raw pixels, offset searched
  - measured: held-out max |err| 0.0, accuracy 1.000
  - baseline: unique among 48 staged programs in 0.5 s; undecomposed 4.9e10 programs (7.6 years projected); constant predictor 0.850
  - evidence: research/discrete-perception/RESULTS.md §5
  - reproduce: `tcn demo --only edge`

- **control** — MuJoCo behind the generator contract, replay verified not assumed
  - measured: replay and restore identical, max |state delta| 0 after 8 further steps
  - baseline: scripted controller reaches upright 0.9994 where zero torque never exceeds -1.0000
  - evidence: research/external-environments/RESULTS.md §3
  - reproduce: `tcn demo --only control`

- **language** — balancedness and its lexical unit learned from raw prompt bytes
  - measured: post-audit stream (hardening='context_free_language', the shipped default): stage A unique among 10,496 (base 14, open byte 40), held-out position error 0.0; stage B 1.000 on 859 held-out episodes at lengths 16-22 never trained on. Pre-audit stream (hardening='none', section 19's): 0.9986187845 on 724
  - baseline: post-audit majority constant 0.5262, random 0.500, best fitted feature 0.4738, training-string lookup 0.4738, and section 19's own counting program 0.5262 on this stream; pre-audit majority constant 0.5483, best fitted feature 0.6478; gradient descent conforms 0 of 64 runs on the post-audit space and 0 of 44 pre-audit
  - evidence: research/language-post-audit/RESULTS.md, research/language-capability/RESULTS.md
  - reproduce: `tcn demo --only language`

