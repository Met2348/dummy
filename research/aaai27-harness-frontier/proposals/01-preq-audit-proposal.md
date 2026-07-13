# Proposal 1: PREQ-Audit

## Auditing Anytime-Valid Certificates Under Endogenous Harness Evolution

**Status:** supporting-paper candidate, not the recommended AAAI-27 primary.

**Core claim:** anytime-valid statistical primitives can remain valid under
optional stopping while still giving misleading harness-promotion decisions when
the proposal process, task distribution, and evaluation stream are changed by the
evolving agent. A prequential audit can measure this gap and identify when a
certificate controls false promotion versus merely certifying its local statistic.

## 1. Abstract

Self-evolving language-model agents repeatedly modify prompts, tools, memory,
budgets, and orchestration code using evidence produced by their own behavior.
Recent work has begun to gate these modifications with confidence sequences and
e-processes. However, time-uniform validity of a statistical primitive does not by
itself establish that an accepted harness edit improves future performance: the
candidate proposer is adaptive, deployed policies alter the data distribution,
and a repeatedly inspected validation surface may cease to represent future
tasks. We propose PREQ-Audit, a controlled evaluation framework for testing
promotion procedures under endogenous harness evolution. PREQ-Audit separates
transductive gain, reused-gate gain, next-block prequential gain, and final sealed
audit gain; measures familywise false-promotion risk; and stress-tests promotion
rules under proposal adaptivity, optional stopping, task drift, and performative
feedback. The study combines a synthetic environment with known ground-truth
effects and two executable harness-evolution loops on deterministic agent tasks.
The intended contribution is not another gate, but the first empirical
falsification protocol for claims made by anytime-valid harness-evolution systems.

## 2. Motivation and Problem Statement

An evolving harness creates an evaluation loop:

1. run the current harness;
2. inspect traces and failures;
3. propose one or many edits;
4. evaluate candidates on a gate set;
5. promote a candidate;
6. repeat until the score or budget is satisfactory.

This loop violates the clean separation between a fixed hypothesis and an
exogenous evaluation sample. Fixed-sample confidence intervals are clearly
unsafe under repeated peeking. Anytime-valid methods address optional stopping,
but three harder questions remain:

- Does the confidence sequence cover the estimand that matters after deployment?
- Does the proposal mechanism induce a distribution shift not represented by the
  paired gate comparison?
- Does repeated promotion on one surface translate into forward improvement on
  the next unseen tasks?

[SEA](../reading-cards/2607.00871.md) directly implements anytime-valid harness
edit gates, so simply adding confidence sequences is occupied. SEA also states
that the composition of its statistical ingredients inside the endogenous loop
is conjectural and reports single-run expensive evaluations. [TTHE](../reading-cards/2607.08124.md)
explicitly distinguishes transductive evaluation from prequential scoring. The
open problem is therefore an audit of the complete decision process.

## 3. Research Questions and Hypotheses

### RQ1: False promotion

When candidate edits have zero or negative future effect, how often does each
promotion rule accept at least one harmful edit over an adaptive run?

**H1:** fixed-sample and repeatedly reused holdout gates substantially exceed
their nominal familywise error rate. A correctly specified anytime-valid gate
controls optional stopping, but can still exceed the desired false-promotion rate
when the gate estimand differs from the future deployment estimand.

### RQ2: Forward generalization

How do transductive, gate-set, prequential, and sealed-audit gains diverge as the
harness adapts?

**H2:** optimization increases same-batch and reused-gate scores faster than
next-block and sealed-audit scores; the gap grows with proposal count and task
drift.

### RQ3: Endogeneity

Which forms of endogeneity break the link between a valid local certificate and a
valid deployment claim?

**H3:** policy-induced task selection and proposal selection from many candidates
are more damaging than optional stopping alone. Pairing and common random numbers
reduce variance but do not repair estimand shift.

### RQ4: Practical promotion policy

Can a prequential-plus-sealed protocol reduce harmful promotions without making
the evolution loop inert?

**H4:** a two-stage rule, consisting of an anytime-valid paired gate followed by
delayed prequential confirmation, reduces false promotion at a tolerable cost in
time-to-improvement.

## 4. Formal Setup

Let `H_t` be the deployed harness before round `t`. The system observes feedback
history `F_1:t` and proposes candidates

`C_t = A(H_t, F_1:t, U_t)`,

where `A` is an adaptive proposer and `U_t` is proposer randomness. Tasks at round
`t` are drawn from a policy-dependent distribution

`X_t ~ D_t(H_t)`.

For candidate `c`, define four effects relative to the incumbent:

- `Delta_trans(c,t)`: effect on the same tasks used to construct the edit;
- `Delta_gate(c,t)`: effect on the repeatedly queried promotion set;
- `Delta_pre(c,t)`: effect on the next unseen block before adaptation to it;
- `Delta_audit(c)`: effect on a prospectively sealed final audit distribution.

The primary safety estimand is the familywise harmful-promotion probability:

`FHP = P(exists t: Promote(C_t) and Delta_audit(C_t) <= -epsilon_harm)`.

Secondary estimands are:

- false-promotion count per run;
- time to first true promotion;
- cumulative forward regret;
- `Delta_gate - Delta_pre` optimism gap;
- confidence-sequence coverage for its declared local estimand;
- sealed-audit coverage and calibration.

The central distinction is between **statistical coverage** and **decision
validity**. A gate can cover `Delta_gate` while promotion remains unsafe for
`Delta_audit`.

## 5. Proposed Audit Framework

### 5.1 Four promotion procedures

1. **Fixed-N Gate:** paired bootstrap or Wald interval checked after each round.
2. **Reused Holdout Gate:** fixed validation set with a conventional significance
   or margin threshold.
3. **Anytime Gate:** paired confidence sequence or e-process with a global error
   budget across proposals.
4. **PREQ-Confirm:** Anytime Gate plus confirmation on the next unseen block;
   accepted candidates remain provisional until forward confirmation.

All procedures receive identical candidate proposals. This isolates the gate from
the quality of the evolver.

### 5.2 Endogeneity stressors

The audit varies one stressor at a time:

- **optional stopping:** stop when an edit passes;
- **multiple proposals:** choose the best of `K` candidates before gating;
- **reused tasks:** revisit the same gate set over rounds;
- **adaptive task selection:** preferentially sample recent failures;
- **performative shift:** accepted edits change which task states are encountered;
- **block drift:** task-family proportions change over time;
- **proxy mismatch:** gate uses process reward while audit uses task success.

### 5.3 Synthetic ground-truth environment

Construct a low-cost simulator in which each harness edit has a known effect on
task strata and changes future stratum exposure. This permits repeated Monte Carlo
runs and direct measurement of FHP, coverage, and regret.

The simulator should include:

- 20 task strata with heterogeneous baseline difficulty;
- sparse positive, null, and harmful edit effects;
- candidate selection from noisy trace-derived signals;
- controllable proposal multiplicity `K`;
- controllable policy-induced distribution shift;
- at least 500 independent evolution streams per condition.

This synthetic layer is essential. A real benchmark cannot reveal whether a
nominal null was truly null.

### 5.4 Executable agent layer

Use two small evolution mechanisms:

- a TTHE-style trace-to-edit proposer operating on prompt and retry parameters;
- a Self-Harness-style proposer with regression gating.

Apply them to deterministic workspace/tool tasks where success is checked by
scripts, not an LLM judge. Restrict mutable harness state to three auditable knobs:

- verification instruction or check insertion;
- retry/recovery policy;
- context refresh template.

The real layer demonstrates that the synthetic failure modes occur in agent
engineering, but it is not expected to estimate FHP with the same precision.

## 6. Experimental Design

### 6.1 Synthetic factorial

| Factor | Levels |
|---|---|
| Promotion procedure | Fixed-N, Reused Holdout, Anytime, PREQ-Confirm |
| Proposal multiplicity | 1, 4, 16 |
| Performative shift | none, moderate, strong |
| Block drift | stationary, gradual, abrupt |
| Stream length | 10, 25 promotion rounds |

Use at least 500 Monte Carlo streams per cell for the main subset. A fractional
design can be used for interactions after all one-factor effects are verified.

### 6.2 Agent experiment

| Dimension | Minimum design |
|---|---|
| Base models | 2 model families |
| Evolvers | 2 |
| Gate procedures | Reused Holdout, Anytime, PREQ-Confirm |
| Task surfaces | local workspace suite plus one public benchmark slice |
| Stream blocks | 6-8 |
| Stream orderings | 2 predeclared orderings |
| Repeats | 2 when model stochasticity is non-negligible |

The sealed audit block is generated or sampled before any evolution run and its
labels remain inaccessible to the proposer and gate.

### 6.3 Pilot

The 72-hour pilot uses one model, one evolver, three gates, four blocks, and 10
tasks per block. Continue only if at least one of the following is observed:

- a measurable gate-to-prequential optimism gap;
- at least one harmful or non-forward promotion under the reused gate;
- a meaningful reduction in false promotion under PREQ-Confirm;
- synthetic FHP inflation under a realistic endogeneity stressor.

## 7. Metrics and Statistical Analysis

### Primary metrics

- empirical FHP with Clopper-Pearson or Wilson intervals across simulated streams;
- prequential paired gain and its confidence sequence;
- sealed-audit harmful-promotion rate;
- cumulative forward regret.

### Secondary metrics

- number of proposed and accepted edits;
- time and tasks required per true promotion;
- abstention rate;
- token and wall-clock cost;
- interval width and stopping time;
- transductive-to-forward optimism ratio.

### Analysis plan

1. Verify confidence-sequence coverage for the exact local estimand it claims.
2. Test whether coverage transfers to the future estimand; report the difference,
   not only rejection rates.
3. Estimate stressor effects on FHP with a binomial generalized linear model.
4. Cluster real-agent uncertainty by task and stream ordering.
5. Report all promotion rounds, including no-op candidates and rejected edits.
6. Keep the final audit untouched until all thresholds and plots are frozen.

## 8. Baselines and Nearest Work

- [SEA](../reading-cards/2607.00871.md): direct anytime-valid gate baseline and
  main collision.
- [TTHE](../reading-cards/2607.08124.md): transductive evolution baseline.
- [Self-Harness](../reading-cards/2606.09498.md): reused regression-gate baseline.
- [AHE](../reading-cards/2604.25850.md): trace-driven harness evolution baseline.
- [Generalization in Adaptive Data Analysis and Holdout Reuse](../reading-cards/1506.02629.md):
  methodological ancestor for adaptive reuse.
- [Safe Anytime-Valid Inference](../reading-cards/2210.01948.md): statistical
  primitive reference.
- [Performative Prediction](../reading-cards/2002.06673.md): distribution-shift
  lens for policy-dependent data.
- [AI Agents That Matter](../reading-cards/2407.01502.md): benchmark-overfitting
  motivation.

## 9. Required Ablations

- same candidate sequence across all gates;
- optional stopping only versus proposal adaptivity only;
- paired versus unpaired evaluation;
- common random numbers on versus off;
- gate reward equal to versus different from audit reward;
- sealed block stationary versus shifted;
- one candidate versus best-of-`K` candidates;
- provisional promotion with and without rollback after failed confirmation.

## 10. Expected Contributions

1. A taxonomy separating time-uniform statistical validity from promotion and
   forward-deployment validity.
2. An open simulator with known edit effects and performative feedback.
3. A reproducible prequential audit protocol for evolving harnesses.
4. Empirical conditions under which anytime-valid gates do and do not control
   harmful promotions.
5. A practical delayed-confirmation rule with quantified safety-cost tradeoff.

The main-track bar requires a non-obvious result. Merely confirming that fixed-N
tests fail under peeking is insufficient.

## 11. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Looks like a critique of SEA | Contribute a general simulator, formal estimand split, and two independent evolvers. |
| No gap appears in real tasks | Make synthetic ground-truth evidence primary and use real tasks for external validity. |
| Confidence sequence is misimplemented | Validate against a known Bernoulli stream and reproduce published coverage before agent runs. |
| Proposal and gate cannot be separated | Pre-generate and replay candidate sequences across gate conditions. |
| Too many factors | Freeze one main stressor interaction after the 72-hour pilot. |

## 12. Kill Criteria

Stop the AAAI version if any of the following holds:

- the synthetic audit does not reveal a deployment-validity gap beyond standard
  optional-stopping failures;
- PREQ-Confirm offers no safety improvement at a usable promotion rate;
- a second evolver cannot be reproduced by day 4;
- the contribution reduces to "use an untouched test set";
- the sealed audit must be inspected to tune the method.

## 13. Eighteen-Day Execution Plan

| Days | Deliverable |
|---|---|
| 1-2 | Simulator, estimands, and unit-tested confidence sequence |
| 3 | 72-hour synthetic and one-evolver pilot; go/no-go |
| 4-6 | Full synthetic experiment and stressor analysis |
| 7-9 | Second evolver and deterministic task-stream runner |
| 10-12 | Agent experiments and sealed audit |
| 13-14 | Ablations, sensitivity, and final figures |
| 15-16 | Seven-page paper draft |
| 17-18 | Reproducibility pass, artifact, and claim audit |

## 14. Resource Budget

- Synthetic experiments: CPU only; target under 6 hours total.
- Agent pilot: 240-400 episodes.
- Full agent study: approximately 1,000-1,500 episodes depending on repeats.
- Store every candidate, task ID, randomization seed, gate statistic, decision,
  and future outcome.
- Set a hard API/token ceiling before day 1; the synthetic layer must remain
  publishable if the expensive layer is reduced.

## 15. Reproducibility Checklist

- immutable task-block manifest and sealed-audit hash;
- versioned harness candidates and diffs;
- deterministic simulator seeds;
- declared gate estimand and error budget;
- no unlogged candidate filtering;
- complete promotion and rejection ledger;
- environment, model, prompt, and budget versions;
- script-generated tables and figures.

## 16. Proposed Paper Structure

1. Introduction: certificates are not deployment claims.
2. Related work: self-evolution, adaptive evaluation, anytime-valid inference.
3. Formal distinction among local, prequential, and sealed-audit effects.
4. PREQ-Audit benchmark and promotion procedures.
5. Synthetic ground-truth results.
6. Executable agent results and case studies.
7. Limitations and implications for evolving-agent evaluation.

## 17. Longer-Term PhD Extension

The sprint version studies harness edits. A broader program could cover adaptive
benchmark construction, learned reward models, model-weight updates, and human
feedback loops, eventually developing validity conditions for closed-loop AI
systems whose policies, data, and evaluators co-evolve.
