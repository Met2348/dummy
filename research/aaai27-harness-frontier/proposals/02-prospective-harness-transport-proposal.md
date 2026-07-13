# Proposal 2: Prospective Harness Transport

## Predicting Which Atomic Harness Interventions Survive Model and Task Shifts

**Status:** recommended AAAI-27 primary and strongest four-year research spine.

**Core claim:** harness interventions are not universally portable or universally
model-specific. Their target effects can be predicted from intervention mechanism,
model-interface behavior, task horizon, and operating point, provided the study
uses atomic interventions and prospectively sealed target cells.

## 1. Abstract

Agent performance is produced by a model-harness-environment system, yet harness
improvements are commonly reported on the model and task distribution used to
develop them. Existing papers provide conflicting evidence: some harness changes
transfer across many models, others are model-specific, and recent work reports
cross-model sign reversals. These studies establish that transfer can succeed or
fail, but do not provide a prospective rule for deciding which intervention should
be reused after a model upgrade. We propose Prospective Harness Transport, a causal
evaluation protocol and predictive model for atomic harness interventions. The
protocol estimates paired intervention effects under fixed tools and budgets,
separates source, validation, and sealed target model-task cells, and predicts the
sign and magnitude of unseen target effects using intervention descriptors and
measured operating-point features. The empirical study covers verification,
retry/recovery, context refresh, and tool-schema normalization across multiple
model families and deterministic task domains. The intended result is a transport
law: environment-contract interventions should transfer broadly, while
prose-level and aggressively tuned control policies should depend on model and
task operating points. The work turns a descriptive portability question into a
prospective prediction problem.

## 2. Motivation and Novelty Boundary

The practical question is simple: when a laboratory or company upgrades the base
model, which harness improvements can be retained without rerunning a full search?

Current evidence is fragmented:

- [Life-Harness](../reading-cards/2605.22166.md) reports broad transfer from one
  source model to 17 others in deterministic environments.
- [AHE](../reading-cards/2604.25850.md) reports cross-model transfer but notes that
  step budget and timeout were tuned to a source model.
- [Self-Harness](../reading-cards/2606.09498.md) emphasizes model-specific harness
  adaptation.
- [HarnessBridge](../reading-cards/2606.12882.md) transfers a learned interface
  controller to unseen generators.
- [SEAGym](../reading-cards/2606.17546.md) shows that a harness evolved with one
  backend can improve one target and harm another.
- [Probe-and-Refine](../reading-cards/2606.20512.md) shows a concrete cross-model
  collapse in repository guidance.

These works occupy the statement "transfer is heterogeneous." A main-conference
paper must add three things:

1. **atomicity:** isolate what changed;
2. **prospection:** seal target cells before model fitting;
3. **prediction:** forecast target effects, not merely describe them afterward.

## 3. Research Questions and Hypotheses

### RQ1: How heterogeneous are atomic harness effects?

**H1:** between-cell heterogeneity is substantial relative to rollout noise, and
at least one intervention changes sign across model-task cells.

### RQ2: Which intervention mechanisms transfer?

**H2:** deterministic environment-contract interventions, such as schema
normalization and executable verification, preserve effect sign more often than
prose guidance, context policies, and retry schedules.

### RQ3: Can target effects be predicted prospectively?

**H3:** a hierarchical response model using intervention descriptors and measured
operating-point features predicts sealed target effects better than source-mean,
global-effect, and nearest-model baselines.

### RQ4: What must be retuned after a model upgrade?

**H4:** the need for retuning is predictable from interface-error rate, clean
success rate, average trajectory horizon, and intervention resource coupling.

### RQ5: Does budget matching change the conclusion?

**H5:** some apparent transfer gains disappear when expected token, step, and
wall-clock budgets are equalized, especially for retry and context interventions.

## 4. Formal Problem Definition

Let a cell be `c = (m, d, b)`, where:

- `m` is a model backend;
- `d` is a task distribution;
- `b` is a resource policy fixing tool access, step limit, timeout, and expected
  token budget.

Let `h_0` be the baseline harness and `h_j` an atomic intervention. For task `i`,
define potential outcomes `Y_i(h_j, c)` and `Y_i(h_0, c)`. The cell-specific
average treatment effect is

`Delta(c,j) = E[Y_i(h_j,c) - Y_i(h_0,c)]`.

For a source cell `s` and target cell `t`, define:

- `SignTransfer(s,t,j) = 1[Delta(s,j) * Delta(t,j) > 0]`;
- `TransferError = |Delta_hat(t,j) - Delta(t,j)|`;
- `TransferRatio = Delta(t,j) / max(|Delta(s,j)|, epsilon)`;
- `NegativeTransfer = 1[Delta(s,j) > epsilon and Delta(t,j) < -epsilon]`.

The primary prediction task is:

`Delta_hat(t,j) = f(z_j, x_t, r_t; source data)`,

where `z_j` describes the intervention, `x_t` describes the model-task cell, and
`r_t` contains low-cost clean-run operating features measured before target
intervention outcomes are observed.

## 5. Atomic Intervention Library

Each intervention must have a written contract specifying changed information,
control flow, resource use, and expected mechanism.

### H1: Executable verification

After a candidate final action, run a deterministic checker and return a bounded
failure message. The model, tools, and task state are unchanged.

**Mechanism class:** environment contract / feedback.

### H2: Tool-schema normalization

Canonicalize tool names, argument order, error fields, and machine-readable error
codes without adding task knowledge.

**Mechanism class:** interface alignment.

### H3: Retry and rollback

On a declared tool-error class, restore the previous valid state and permit one
additional attempt. Match expected step cost with the baseline through a fixed
budget policy or report both unconstrained and cost-matched effects.

**Mechanism class:** control / recovery.

### H4: Context refresh

At a predeclared context-load threshold, replace stale trajectory detail with a
deterministic state summary assembled from logged facts. Do not use a model
summary in the primary experiment.

**Mechanism class:** information exposure.

### Optional H5: Prose-only strategy guidance

Add a short strategy instruction without changing executable behavior. This is a
useful negative-transfer comparator but should be excluded if schedule is tight.

## 6. Atomicity Tests

Before experiments, each intervention must pass:

- identical tool permission set;
- identical initial observations and task data;
- no hidden task-specific examples;
- declared token and step-budget effect;
- deterministic unit test on a fixed trace;
- one-change diff review;
- no other intervention triggered implicitly;
- logged intervention activation count.

If atomicity fails, the corresponding effect cannot support the causal transport
claim.

## 7. Cell Features for Prediction

Only features available before target intervention outcomes may be used.

### Model-interface features

- malformed tool-call rate on clean runs;
- tool-error recovery rate;
- instruction-compliance rate on synthetic probes;
- valid finalization rate;
- repeated-action or loop rate;
- average tokens per successful transition.

### Task features

- median required horizon;
- number of tool types;
- state observability;
- verifier determinism;
- branching factor;
- fraction of tasks requiring rollback or external state mutation.

### Operating-point features

- clean success rate;
- mean and variance of trajectory length;
- fraction of budget consumed before finalization;
- interface-error composition;
- timeout and truncation rate.

### Intervention descriptors

- changes information versus control versus executable contract;
- deterministic versus model-generated;
- stateful versus stateless;
- expected cost increment;
- activation frequency;
- dependence on model prose compliance.

## 8. Predictive Models

### 8.1 Primary hierarchical model

For binary task success:

`logit P(Y_i=1) = alpha_i + mu_c + tau_j + gamma^T(z_j x x_c) + u_mj + v_dj`,

with partial pooling over model, task domain, and intervention. The prediction for
a sealed cell excludes its intervention outcomes.

Use a Bayesian hierarchical logistic model if implementation and diagnostics are
stable by day 6. Otherwise use a penalized mixed-effects or regularized logistic
model with cluster bootstrap intervals. The paper's contribution is the protocol
and prospective prediction, not a needlessly complex estimator.

### 8.2 Baselines

1. **Zero-effect:** predict no transfer.
2. **Global mean:** one effect per intervention across source cells.
3. **Source copy:** use the originating cell's effect.
4. **Nearest model:** choose the source cell closest in clean success rate.
5. **Task-only model:** ignore model-interface features.
6. **Matrix factorization:** low-rank model-intervention response surface.

### 8.3 Prediction evaluation

- sign accuracy and balanced accuracy;
- Brier score for probability of positive transfer;
- magnitude MAE and RMSE;
- 80% and 95% interval coverage;
- regret of the induced reuse/retune decision;
- calibration plot for negative-transfer risk.

## 9. Experimental Design

### 9.1 Task domains

Use two deterministic domains with distinct failure surfaces:

1. **Workspace/tool tasks:** file editing, structured search, command execution,
   and artifact verification with local graders.
2. **Stateful service tasks:** a bounded offline subset of `tau`-style tool tasks
   or another deterministic environment with explicit state transitions.

Select 20 tasks per domain for the main experiment and reserve additional tasks
for the sealed target. Avoid LLM-as-judge primary outcomes.

### 9.2 Models

Minimum full design:

- one lower-cost open or locally served model;
- one stronger model from a different family;
- one prospectively sealed target backend.

The target model is named and its access configuration is frozen before source
results are analyzed. A fourth model is useful only if the first three finish by
day 11.

### 9.3 Main matrix

| Dimension | Levels |
|---|---|
| Models | 3 |
| Domains | 2 |
| Harness conditions | baseline + 4 atomic interventions |
| Tasks | 20 per domain |
| Repeats | 2 for stochastic cells |

The upper bound is `3 x 2 x 5 x 20 x 2 = 1,200` episodes. Paired task seeds and a
sequential stopping rule for clearly null cells can reduce cost, but target cells
must follow the predeclared rule.

### 9.4 Prospective split

- **Source cells:** two models across both domains.
- **Model-validation cells:** leave one source model-domain pair out in rotation.
- **Sealed target:** third model, with at least one task subset not used for model
  selection.
- **Final audit:** fresh tasks from both domains after prediction thresholds are
  frozen.

The target effect table remains encrypted, access-controlled, or at minimum
script-hidden until the prediction file and commit hash are recorded.

## 10. Seventy-Two-Hour Kill Test

Use:

- 2 models;
- 1 task domain;
- baseline plus verification, schema normalization, and retry;
- 12 tasks;
- 1 repeat, followed by repeats only on discordant tasks.

This is 72-96 episodes. Estimate paired effects and inspect activation logs.

Continue only if:

1. at least one intervention has materially different effects across models;
2. at least one measured operating feature plausibly explains the difference;
3. atomicity and budget matching are intact;
4. runtime projects to completion by day 11.

If every intervention has the same sign and similar magnitude, stop the transport
claim or broaden to a task shift only if that split was predeclared.

## 11. Statistical Analysis

### Effect estimation

- use paired task outcomes and report risk differences;
- use exact McNemar tests only as supporting evidence;
- obtain cluster bootstrap intervals over tasks;
- fit a hierarchical model for the joint response surface;
- report both raw and cost-matched effects.

### Heterogeneity

Estimate variance components for intervention-by-model and
intervention-by-domain terms. Report the fraction of total harness-effect variance
attributable to each moderator.

### Prospective prediction

Freeze model specification, feature transformations, and hyperparameters before
unsealing targets. Compare all baselines on identical targets. Use paired bootstrap
differences in prediction loss.

### Multiple outcomes

Primary outcome: deterministic task success.

Secondary outcomes: tokens, wall-clock time, steps, recovery rate, and verifier
activation. Do not promote a secondary metric to primary after seeing results.

## 12. Required Baselines and Collisions

- [SEAGym](../reading-cards/2606.17546.md): descriptive cross-model sign reversal.
- [Life-Harness](../reading-cards/2605.22166.md): broad transfer in deterministic
  interface families.
- [AHE](../reading-cards/2604.25850.md): evolved harness transfer with
  operating-point coupling.
- [Meta-Harness](../reading-cards/2603.28052.md): held-out-model harness evidence.
- [HarnessBridge](../reading-cards/2606.12882.md): learned controller transfer.
- [Probe-and-Refine](../reading-cards/2606.20512.md): negative cross-model guidance
  result.
- [AgentTether](../reading-cards/2607.06273.md): cross-model repair-layer transfer.
- [Stop Comparing LLM Agents](../reading-cards/2605.23950.md): harness variance and
  ranking-reversal motivation.
- [Agent Psychometrics](../reading-cards/2604.00594.md): prediction for unseen
  model-scaffold combinations, but not atomic treatment effects.

## 13. Required Ablations

- atomic interventions versus bundled whole harness;
- predictor without intervention descriptors;
- predictor without model-interface features;
- clean capability only versus full operating profile;
- raw budget versus expected-cost-matched budget;
- source-copy versus hierarchical partial pooling;
- random target split versus prospectively sealed target;
- deterministic verification versus prose-only guidance;
- target calibration with 0, 4, and 8 anchor tasks.

## 14. Expected Contributions

1. A prospective causal-transport protocol for model-harness systems.
2. A versioned library of atomic harness interventions with enforceable contracts.
3. The first out-of-sample prediction of harness treatment effects on sealed
   model-task cells.
4. A mechanistic taxonomy of portable versus model-coupled interventions.
5. A practical reuse/retune rule for model upgrades.
6. A negative-transfer dataset and complete paired trajectories.

The main-conference contribution is item 3. Without prospective predictive lift,
the work is a careful benchmark but not the intended paper.

## 15. Killer Figures and Tables

### Figure 1: transfer response surface

Rows are atomic interventions; columns are source-to-target shifts. Cells show
paired effect and uncertainty. Sign reversals are explicitly marked.

### Figure 2: predicted versus observed target effects

Plot sealed target predictions with intervals against observed effects. Include
the global-effect and nearest-model baselines.

### Figure 3: reuse/retune decision curve

Show utility or regret as the tolerated negative-transfer risk changes.

### Table 1: intervention contracts

State exactly what information, control, and budget each intervention changes.

### Table 2: prospective prediction metrics

Report sign accuracy, MAE, interval coverage, and decision regret.

## 16. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Looks like more ablations | Make sealed prediction and decision regret the central result. |
| Effects are too sparse | Choose tasks with observable interface failures and verify activation frequency in pilot. |
| Interventions are not atomic | Enforce contracts, unit tests, and activation logs before any main run. |
| Model access changes | Freeze exact model identifiers and cache all requests and responses. |
| Third-model target is too expensive | Reduce task count, not intervention identity or target integrity. |
| Predictor overfits few cells | Use small predeclared features, partial pooling, and strong simple baselines. |
| Budget confounds transfer | Report raw and cost-matched estimands separately. |

## 17. Kill Criteria

Stop or downgrade if:

- all pilot effects agree within narrow intervals;
- atomicity requires changing tools, information, and budget together;
- target prediction does not beat a constant-effect baseline;
- fewer than two intervention mechanisms activate often enough to estimate;
- the third model or second domain cannot be completed by day 11;
- the final story is only "some harnesses transfer and some do not."

## 18. Eighteen-Day Execution Plan

| Days | Deliverable |
|---|---|
| 1 | Freeze cells, tasks, interventions, target, and outcome schema |
| 2 | Atomicity tests and paired runner |
| 3 | 72-96 episode kill test and go/no-go |
| 4-6 | Complete source cells and operating profiles |
| 7-8 | Fit and freeze predictor; leave-cell-out validation |
| 9-11 | Run sealed target cells |
| 12 | Unseal and score target predictions |
| 13-14 | Cost matching, ablations, and final audit tasks |
| 15-16 | Paper draft and figures |
| 17-18 | Reproducibility, artifact, and claim audit |

## 19. Resource Budget

- Pilot: 72-96 episodes.
- Full main matrix: up to 1,200 episodes.
- Recommended API reduction: one repeat for deterministic/temperature-zero cells,
  two repeats only where outcome discordance or model stochasticity is observed.
- Storage: complete traces, tool calls, state hashes, intervention activations,
  token counts, and grader outputs.
- Compute: local deterministic environments; no model training is required.

## 20. Reproducibility and Leakage Controls

- hash source, validation, target, and final-audit task manifests;
- commit target prediction file before unsealing;
- log model version and serving parameters;
- version intervention contracts and code diffs;
- use isolated task workspaces and paired seeds;
- prevent target outcomes from entering prompts or feature construction;
- preserve all failed, timed-out, and no-op runs;
- release analysis scripts that regenerate effect and prediction tables.

## 21. Proposed Paper Structure

1. Introduction: the model-upgrade portability problem.
2. Related work and the gap between evidence and prospective prediction.
3. Atomic intervention and causal transport setup.
4. Prospective protocol and predictive response model.
5. Source-cell effects and heterogeneity.
6. Sealed-target prediction results.
7. Mechanisms, cost matching, and practical reuse rules.
8. Limitations and broader implications.

## 22. Four-Year PhD Program

This direction scales naturally beyond one sprint:

1. atomic harness-effect transport across model upgrades;
2. transfer across task domains and deployment budgets;
3. transport-aware harness optimization and warm starts;
4. online monitoring for transport failure after deployment;
5. safety and governance rules for retaining harness components across rapidly
   changing model ecosystems.
