# Idea 2: Harness Transportability

## When Does a Harness Survive a Model Upgrade?

**Verdict:** primary sprint bet and best PhD spine, but only in the narrowed
prospective-prediction form.

**One-sentence claim:** harness improvements are neither universally portable nor
universally model-specific; their transfer is predictable from intervention
type, model-interface compatibility, task horizon, and operating point.

## Why this is open

The literature gives incompatible high-level answers:

- AHE and Life-Harness report broad cross-model transfer.
- Self-Harness frames useful changes as model-specific.
- AdaCoM transfers best between similarly capable agents.
- Harness Updating Is Not Harness Benefit finds non-monotone benefit by model
  tier.
- Stop Comparing finds harness variance dominates on one controlled slice.
- General Agent Evaluation finds backbone choice dominates overall across six
  benchmarks.

SEAGym now provides a direct descriptive collision: a DeepSeek-evolved harness
improves GLM ID performance but hurts GPT-5.4, and it attributes asymmetry to the
failure surfaces exposed by each rollout backend. This occupies the claim that
"harnesses sometimes fail to transfer." The remaining slot is to isolate atomic
interventions and predict transfer on prospectively sealed model-task cells.

AHE explicitly acknowledges that its step budget and timeout were fitted to one
model, confounding portability with operating-point coupling. No paper in the
screened set provides a general treatment-effect transport protocol across
atomic harness interventions, unseen models, and unseen task domains.

## Formal object

For model `m`, task distribution `d`, atomic harness intervention `h`, and fixed
resource policy `b`, define:

`Delta(m,d,h,b) = E[Y(h) - Y(h0) | m,d,b]`.

For source setting `s` and target setting `t`, define sign transfer and scaled
transfer:

- `SignTransfer = 1[Delta_s * Delta_t > 0]`;
- `TransferRatio = Delta_t / max(abs(Delta_s), epsilon)`.

The paper should estimate a hierarchical response surface rather than claim one
universal ratio. Candidate moderators include model capability, instruction
following, tool-schema compliance, task horizon, interface entropy, and whether
the intervention changes information, control, or only prose.

## Contributions needed for a main paper

1. A causal transport protocol with source, validation, and prospectively sealed
   target model-task cells.
2. A transfer matrix for atomic interventions: verification, retry/recovery,
   context refresh, tool filtering, state checkpointing, and instruction-only
   guidance.
3. A hierarchical model that predicts sign and magnitude on unseen cells.
4. A practical rule: which layers should be reused after a model upgrade and
   which require retuning.
5. A negative result showing where “universal harness” claims fail.

## Minimal experiment

Pilot:

- 2 models;
- 3 atomic interventions plus baseline;
- 12 deterministic tasks;
- 2 repeats.

Full minimum:

- 3 model tiers or families;
- 4 atomic interventions plus baseline;
- 2 task domains;
- 20 tasks per domain;
- 2 repeats for stochastic cells.

This is about 800-960 episodes. Use a fractional design only after the atomic
effects are measured cleanly. Keep tool permissions, task environment, timeout,
and expected resource budget fixed.

## Killer figure

A source-to-target transfer heatmap whose rows are atomic interventions and
columns are model/task shifts. The central result is that environment-contract
and deterministic-verification interventions transfer, while prose-level
strategy and aggressively tuned retry/context policies show sign flips.

The second panel compares predicted and observed target effects on sealed cells.

## Nearest collisions and required distinction

- Life-Harness: broad transfer, but one deterministic interface family and no
  transport estimand.
- AHE: transfer evidence, but acknowledged operating-point coupling.
- Meta-Harness: some held-out-model evidence, not a systematic transport study.
- SEAGym: explicit cross-model transfer matrices and sign reversals for evolved
  whole harnesses; this idea must add atomicity, a transport estimand, and
  out-of-sample prediction rather than another matrix.
- Agent Psychometrics: predicts unseen LLM-scaffold combinations when both parts
  were observed; it does not estimate atomic intervention transport.
- Harness Updating Is Not Harness Benefit: establishes non-monotonicity, not a
  predictive transport law.

## 18-day execution path

Days 1-3: atomic intervention runner and 72-96 episode kill test.

Days 4-8: full source matrix; freeze the target cells before inspecting results.

Days 9-12: target runs, hierarchical model, and leave-one-model/task-out tests.

Days 13-18: sensitivity to budget, paper, and artifact release.

## Kill criterion

Stop if all interventions have the same sign and similar magnitude across the
pilot cells, or if interventions cannot be made atomic without changing budgets,
tools, and information simultaneously.

## Main risk

The paper can look like “more ablations” unless the transport estimand,
prospective target split, and predictive result are central. The scope should be
four clean interventions, not ten loosely specified harnesses.
