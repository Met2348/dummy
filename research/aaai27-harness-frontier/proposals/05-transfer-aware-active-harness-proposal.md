# Proposal 5: ActiveHarness-T

## Transfer-Aware Warm-Start Optimization for Harnesses After Model Upgrades

**Status:** longer-term algorithmic proposal. The original ActiveHarness claim is
retired because HARBOR directly covers cost-aware Bayesian harness optimization.
This proposal survives only by making cross-model transfer and negative-transfer
control central.

**Core claim:** a harness optimizer should not restart from zero after every model
upgrade, nor should it blindly reuse the old response surface. A transfer-aware
warm start can reuse stable component effects, detect model-specific interactions,
and reach a safe near-optimal target configuration with fewer rollouts than both
cold-start optimization and naive pooling.

## 1. Abstract

Modern agent harnesses expose many configurable components, including context
compression, caching, verification, retry, memory, and tool-interface policies.
HARBOR formalizes configuration search as constrained noisy Bayesian optimization
with mixed variables, multi-fidelity acquisition, and posterior safety checks.
However, its response surface is conditional on a particular agent-task-model
triple, and a model change requires a new optimization run. This is costly in a
deployment environment where models are upgraded frequently. We propose
ActiveHarness-T, a transfer-aware harness optimizer for model upgrades. The method
uses source-model archives to construct a hierarchical response prior, probes a
small target anchor set to estimate source-target compatibility, and adaptively
mixes shared and target-specific surrogates. A negative-transfer gate falls back
to target-only optimization when source rankings or effects disagree, while a
safe acquisition rule protects a baseline quality constraint. We evaluate the
method first on complete offline harness-response tables and then prospectively on
an eight-component agent harness across multiple model families. The main result
is evaluations-to-safe-near-optimality on an unseen target model. The intended
contribution is not another Bayesian optimizer, but a statistically auditable
mechanism for deciding what optimization evidence survives a model upgrade.

## 2. Motivation and Collision With HARBOR

The original ActiveHarness proposal asked how to find a strong harness
configuration under expensive stochastic evaluations. [HARBOR](../reading-cards/2604.20938.md)
already addresses that problem with:

- constrained noisy Bayesian optimization;
- a mixed, cost-heterogeneous flag space;
- a block-additive sparse surrogate;
- multi-fidelity cost-aware acquisition;
- TuRBO trust regions;
- posterior chance-constrained safety;
- an end-to-end production-harness run.

A paper that replaces its acquisition function or adds a best-arm confidence
bound would likely look incremental. HARBOR also identifies the surviving gap: its
posterior is conditional on the agent-task-model triple, it claims no transfer
across agent families, and each model epoch requires a new run.

This proposal therefore studies **optimization evidence transport**. It is related
to the Prospective Harness Transport proposal, but asks a distinct algorithmic
question:

> Given source response surfaces and a limited target rollout budget, how should
> the optimizer reuse, discount, or reject source evidence while maintaining a
> target safety constraint?

## 3. Research Questions and Hypotheses

### RQ1: Is a warm start useful after a model upgrade?

**H1:** source harness evaluations reduce target simple regret when the source and
target share stable main effects, even when some interactions change.

### RQ2: When does naive pooling cause negative transfer?

**H2:** pooling all source observations is harmful when clean anchor rankings are
weakly correlated or when interface-error composition changes substantially.

### RQ3: Can negative transfer be detected cheaply?

**H3:** a small, structured anchor set containing the baseline, single-component
toggles, and two interaction probes predicts whether source warm starts should be
trusted.

### RQ4: Does transfer help under a safety constraint?

**H4:** transfer-aware acquisition reaches a configuration whose lower confidence
bound exceeds the deployment baseline with fewer target rollouts than cold-start
HARBOR, random search, and standard multi-task BO.

### RQ5: Which harness effects are reusable?

**H5:** deterministic environment-contract and interface components have stable
main effects, while model-facing prose, context thresholds, and component
interactions require target adaptation.

## 4. Formal Problem

Let `x in X` be a binary/categorical harness configuration. For model-task epoch
`e`, evaluation returns quality and cost:

`Y_e(x) = f_e(x) + epsilon`,

`C_e(x) = c_e(x) + eta`.

Source archives are

`D_S = {(e, x, y, c): e in source epochs}`.

For unseen target epoch `t`, the goal is to select `x_hat` under target evaluation
budget `B` such that:

- simple regret `f_t(x*) - f_t(x_hat)` is small;
- cost satisfies `C_t(x_hat) <= C_max`;
- safety satisfies `P(f_t(x_hat) >= f_t(x_0)-delta) >= 1-alpha`.

The transfer structure is

`f_e(x) = f_shared(x) + r_e(x)`,

where `f_shared` captures portable effects and `r_e` captures epoch-specific
deviation. The method must learn how much source evidence to borrow without
assuming `r_t=0`.

## 5. Harness Search Space

Use eight components grouped by mechanism:

### Interface and contract

1. tool-schema normalization;
2. executable final verification;

### Context and memory

3. deterministic context refresh;
4. retrieval cache;
5. cross-episode memory;

### Control and recovery

6. bounded retry/rollback;
7. loop detector;

### Model-facing guidance

8. strategy instruction.

For the sprint, thresholds are fixed and each component is binary. A later paper
can introduce categorical thresholds. This yields `2^8 = 256` configurations,
large enough to require search but small enough to obtain a complete oracle on one
cheap source setting if needed.

## 6. Source-Target Compatibility Probes

Before transfer, evaluate a target anchor set:

- all-off baseline;
- all-on configuration;
- eight single-component toggles or a balanced fractional subset;
- two predeclared interaction probes based on source ANOVA;
- one source-optimal configuration.

With 8-12 anchor configurations, compute:

- rank correlation between source predictions and target anchor outcomes;
- sign agreement for estimated main effects;
- discrepancy in clean model-interface features;
- source-optimum degradation;
- residual likelihood under shared versus target-only surrogates.

These diagnostics feed a transfer weight `rho_t in [0,1]`.

## 7. Proposed Method

### 7.1 Hierarchical response surrogate

Represent the configuration response as a sparse ANOVA:

`f_e(x) = beta_0,e + sum_j beta_j,e x_j + sum_(j,k in E) beta_jk,e x_j x_k`.

Use hierarchical shrinkage:

`beta_j,e ~ N(mu_j, sigma_j^2)`,

with stronger sharing for intervention classes expected to transfer. Keep the
interaction set sparse and source-defined before target outcomes are inspected.

A Gaussian-process implementation is possible, but a hierarchical ANOVA is more
interpretable and easier to audit in an 18-day study.

### 7.2 Transfer gate

Estimate `rho_t` from anchor evidence. Three regimes:

1. **positive transfer:** borrow shared main and interaction effects;
2. **partial transfer:** borrow main effects but relearn interactions;
3. **negative/unknown transfer:** fall back to target-only prior.

The regime decision is logged and cannot be changed after each disappointing
target observation without spending a declared adaptation budget.

### 7.3 Robust mixture prediction

For candidate `x`, combine source-informed and target-only predictions:

`mu_mix(x) = rho_t mu_shared(x) + (1-rho_t) mu_target(x)`.

Inflate uncertainty by source-target discrepancy:

`sigma_mix^2(x) = sigma_target^2(x) + rho_t^2 d_t(x)^2`.

This prevents a confident source optimum from dominating when anchor evidence
shows mismatch.

### 7.4 Safe cost-aware acquisition

Select the next configuration by expected improvement or knowledge gradient per
unit cost, subject to:

`LCB_quality(x) >= f_t(x_0)-delta`,

and

`UCB_cost(x) <= C_max`.

If no candidate satisfies the quality constraint, evaluate a target-only
uncertainty-reducing probe or return the baseline.

### 7.5 Finalist confirmation

When the surrogate identifies 3-5 finalists, use paired tasks and fresh target
rollouts for successive elimination. The final selection certificate reports:

- target observations used;
- source borrowing weight;
- predicted and observed target effect;
- quality lower bound;
- cost upper bound;
- simple-regret estimate relative to the evaluated frontier.

## 8. Possible Theoretical Statement

Do not promise a deep theorem before the empirical method works. A realistic
finite-sample result can assume:

- bounded sub-Gaussian rollout noise;
- a finite configuration set;
- source-target discrepancy bounded by `Gamma(x)` estimated conservatively from
  anchors;
- valid lower/upper confidence bounds after a fixed acquisition schedule or a
  time-uniform correction.

Then show that the selected configuration is epsilon-optimal among safe candidates
with probability at least `1-alpha`, where epsilon depends on target sample size
and the discrepancy bound. If adaptive BO invalidates a clean proof, state a
fixed-confidence finalist guarantee rather than overstating global BO regret.

## 9. Offline Oracle Experiments

Before expensive prospective runs, use complete or near-complete response tables.

### Dataset A: full `2^5` component table

Use [Cross-Component Interference](../reading-cards/2605.05716.md) or a reproduced
equivalent as a small oracle. Construct source-target pairs across model/task
cells and replay optimizers under controlled budgets.

### Dataset B: new cheap `2^8` table

On one locally served model and 12 deterministic tasks, evaluate all 256
configurations once with paired seeds. Repeat a subset to estimate noise. This
table becomes the primary offline oracle.

### Offline questions

- probability of selecting an epsilon-optimal target configuration;
- simple regret versus target evaluation count;
- safety violations;
- failure under deliberately mismatched source epochs;
- calibration of the transfer gate;
- benefit of main-effect versus interaction transfer.

## 10. Prospective Experiment

### Source epochs

- two model families on one deterministic task domain;
- complete or 50%-coverage response archives.

### Target epoch

- a third model family or a clearly versioned model upgrade;
- target outcomes hidden from algorithm tuning;
- target budget limited to 10-20% of the 256 configurations;
- fresh finalist tasks not used by acquisition.

### Replication

Run at least 20 optimizer simulations on offline tables. For the expensive
prospective target, use 3-5 independent acquisition seeds if budget permits and
report each trajectory.

## 11. Baselines

1. random search;
2. greedy component addition;
3. cold-start HARBOR-style Bayesian optimization;
4. BOHB or Hyperband-style multi-fidelity search;
5. naive pooled multi-task BO;
6. source optimum deployed directly;
7. target-only successive halving;
8. transfer-aware method without negative-transfer gate;
9. oracle source-selection baseline on offline tables.

## 12. Evaluation Metrics

### Optimization quality

- simple regret versus target rollouts;
- probability of epsilon-optimal selection;
- evaluations to beat baseline by a fixed margin;
- hypervolume of quality-cost frontier;
- final fresh-task success.

### Safety and transfer

- number of unsafe candidate evaluations;
- probability final selection violates baseline margin;
- negative-transfer detection AUROC or balanced accuracy;
- calibration of `rho_t`;
- regret relative to cold start under compatible and incompatible sources.

### Cost

- target episodes;
- full-suite-equivalent evaluation units;
- tokens and wall-clock time;
- source archive amortization across upgrades.

## 13. Seventy-Two-Hour Kill Test

Use the existing `2^5` table or reproduce a 32-configuration local table.

1. Define two source-target splits with positive transfer and one with negative
   transfer.
2. Implement cold target-only, naive pooling, and transfer-gated hierarchical
   ANOVA.
3. Simulate target budgets from 4 to 16 configurations.
4. Repeat 200 subsampling seeds.

Continue only if:

- transfer gating improves simple regret over cold start in compatible splits;
- it avoids most naive-pooling failures in incompatible splits;
- the result survives multiple source-target split definitions;
- improvement is not solely due to seeing the source optimum;
- a prospective eight-component run is feasible by day 10.

## 14. Required Baselines and Nearest Work

- [HARBOR](../reading-cards/2604.20938.md): direct cold-start optimization
  collision and principal baseline.
- [Meta-Harness](../reading-cards/2603.28052.md): search over complete harness
  programs.
- [AHE](../reading-cards/2604.25850.md): trace-driven harness evolution and
  transfer evidence.
- [Bayesian-Agent](../reading-cards/2606.08348.md): posterior-guided skill
  lifecycle, not configuration-response transfer.
- [Cross-Component Interference](../reading-cards/2605.05716.md): interaction and
  non-submodularity evidence.
- [SEAGym](../reading-cards/2606.17546.md): cross-model sign reversals motivating
  negative-transfer control.
- [Life-Harness](../reading-cards/2605.22166.md): broad reusable environment-side
  structure.

## 15. Required Ablations

- cold start versus naive source pooling versus transfer gate;
- main-effect transfer only versus main-plus-interaction transfer;
- anchor set size 4, 8, 12;
- structured anchors versus random anchors;
- transfer weight fixed versus evidence-adaptive;
- discrepancy inflation on versus off;
- safe acquisition on versus unconstrained acquisition;
- paired tasks on versus independent tasks;
- one versus multiple source epochs;
- compatible versus deliberately incompatible sources.

## 16. Killer Figures

### Figure 1: target regret versus target rollout budget

Show cold HARBOR, naive pooling, random search, and ActiveHarness-T under positive
and negative-transfer splits.

### Figure 2: transfer gate calibration

Plot estimated source compatibility against realized warm-start benefit.

### Figure 3: component effect transport

Compare source and target main/interaction effects; highlight components retained
or relearned.

### Figure 4: prospective acquisition trace

Show every target configuration evaluated, source borrowing weight, safety bound,
and final fresh-task confirmation.

## 17. Expected Contributions

1. The first harness optimizer explicitly designed for model upgrades.
2. A transfer gate that distinguishes reusable source evidence from negative
   transfer using a small target anchor set.
3. A hierarchical, interaction-aware response model with safe cost-aware target
   acquisition.
4. Offline oracle and prospective evidence of target sample savings.
5. An auditable reuse/discount/reject decision for source harness archives.

The main-conference bar requires a prospective target result and robust negative-
transfer evidence. Offline simulations alone are insufficient.

## 18. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Too close to HARBOR | Make source-target transfer and negative-transfer control the formal problem and main experiment. |
| Too close to Proposal 2 | Proposal 2 predicts atomic effects; this proposal optimizes a combinatorial target configuration under a rollout budget. |
| Hierarchical model is generic | Use harness mechanism groups, sparse interactions, paired tasks, and a model-upgrade prospective protocol. |
| No positive transfer | A useful negative result requires showing when the gate correctly falls back, but the optimization benefit claim must be narrowed. |
| Full table is expensive | Build one local oracle, then limit the prospective target to 10-20% coverage. |
| Safety guarantee is weak under adaptive BO | Move the formal certificate to a fresh paired finalist stage. |

## 19. Kill Criteria

Stop the standalone paper if:

- transfer-aware search does not beat cold start on compatible offline splits;
- the gate cannot avoid naive-pooling harm on incompatible splits;
- all component effects are either perfectly stable or entirely unrelated across
  models;
- the prospective target cannot be completed with at least three baselines;
- the only novelty is a new kernel or acquisition function;
- no defensible fresh-task selection certificate can be produced.

## 20. Eighteen-Day Execution Plan

This is the least sprint-friendly proposal.

| Days | Deliverable |
|---|---|
| 1 | Freeze search space, archives, target, and safety constraint |
| 2 | Offline oracle loader and cold/pooled baselines |
| 3 | Transfer-gate kill test and go/no-go |
| 4-6 | Hierarchical ANOVA, anchors, and discrepancy calibration |
| 7-9 | Build or complete local `2^8` response table |
| 10-12 | Prospective target optimization |
| 13 | Fresh-task finalist confirmation |
| 14 | Ablations and negative-transfer stress tests |
| 15-16 | Paper draft and figures |
| 17-18 | Reproducibility, theory scope, and artifact audit |

If the local oracle is not available by day 6, merge the work into the Prospective
Harness Transport paper instead of forcing a standalone submission.

## 21. Resource Budget

- Small offline oracle: 32 configurations across existing task/model cells.
- New local oracle: 256 configurations x 12 tasks = 3,072 episodes, preferably on
  a locally served low-cost model.
- Prospective target: 26-52 configurations x 20 tasks, plus 3-5 finalists on fresh
  tasks.
- CPU for hierarchical model and optimizer simulation; no model training needed.
- Store full response archive, cost, model version, task seed, and activation
  counts for every component.

## 22. Reproducibility and Leakage Controls

- source archives versioned and immutable;
- target model and task manifest declared before anchor evaluation;
- target outcomes unavailable during transfer-gate design;
- every acquisition decision logged with posterior and constraint state;
- fresh finalist tasks isolated from optimization;
- all unsafe, failed, and timed-out evaluations retained;
- offline simulation seeds and source-target splits released;
- exact baseline HARBOR-style implementation documented.

## 23. Proposed Paper Structure

1. Introduction: model upgrades make cold harness retuning unsustainable.
2. Related work: HARBOR, harness evolution, multi-task optimization, transport.
3. Transfer-aware safe optimization problem.
4. Anchor compatibility, hierarchical response, and gated acquisition.
5. Offline oracle experiments.
6. Prospective model-upgrade experiment.
7. Negative transfer, limitations, and deployment economics.

## 24. Longer-Term PhD Extension

This proposal naturally follows the Prospective Harness Transport program. Later
work can study continual model upgrades, source archive selection, task-domain
transfer, nonstationary costs, online drift alarms, and joint optimization of
model routing and harness configuration. It is a strong second- or third-year
algorithmic paper after atomic transport has established which effects are stable.
