# Proposal 3: MRT-Harness

## Micro-Randomized Trials for State-Conditional Agent-Harness Interventions

**Status:** highest-novelty surviving direction; recommended pivot, with high
execution and statistical-design risk.

**Core claim:** verification, retry, and context refresh do not have one global
effect. Their causal value depends on the decision state and remaining horizon.
Prospective micro-randomization identifies when an intervention helps, does
nothing, or causes harm, and can produce a policy that dominates static harness
rules at equal cost.

## 1. Abstract

Agent harness components are usually evaluated as episode-level switches: a
verifier, memory module, or retry policy is enabled for an entire run and average
task success is compared. This design cannot answer the operational question of
when the harness should intervene. We propose MRT-Harness, a micro-randomized
experimental protocol for language-model agents. At predeclared decision points,
the harness records an availability indicator and state features, randomizes a
binary intervention with known probability, and measures a deterministic proximal
outcome over the next few transitions. Weighted and centered least squares
estimates marginal and state-conditional causal excursion effects while accounting
for repeated decisions within episodes. The first study focuses on executable
verification after tool actions in deterministic environments. It tests whether
verification helps after ambiguous or failed actions, wastes budget in clean
states, and becomes harmful near the horizon. A simple effect-informed policy is
then prospectively evaluated against never-verify, always-verify, and heuristic
trigger baselines. The contribution is a causal experimental design and open trace
standard for sequential harness decisions, rather than another learned controller.

## 2. Motivation and Novelty Boundary

The same intervention can have opposing effects:

- verification can detect a silent failure, but can also consume the last useful
  tool step;
- retry can recover a transient failure, but can amplify a bad action loop;
- context refresh can remove stale information, but can drop a crucial constraint;
- memory retrieval can restore a fact, but can distract a model in a clean state.

Episode-level comparisons average over these states and cannot identify the
decision rule. Existing work leaves a clear gap:

- [Cross-Component Interference](../reading-cards/2605.05716.md) estimates static
  whole-episode component effects and interactions.
- [AgentSpec](../reading-cards/2606.14674.md) studies controlled scaffold
  composition, not repeated within-trajectory interventions.
- [Causal Agent Replay](../reading-cards/2606.08275.md) intervenes retrospectively
  on failed traces.
- [Offline Harness Control](../reading-cards/2607.05458.md) learns a policy from
  logged trajectories but does not prospectively randomize decision points.

Micro-randomized trials provide the missing identification strategy. They were
developed for just-in-time adaptive interventions, where treatment availability,
past treatment, and proximal outcomes evolve over time. The statistical machinery
exists, but its application to agent harnesses requires new definitions of
availability, state, proximal progress, carryover, and cost.

## 3. Research Questions and Hypotheses

### RQ1: Is the intervention effect state-dependent?

**H1:** executable verification has a positive proximal effect after tool errors,
state inconsistencies, or repeated actions, near-zero effect in clean progress
states, and negative net value under a low remaining budget.

### RQ2: Does a global component effect hide opposing local effects?

**H2:** the marginal verification effect is modest, while conditional effects have
opposite signs across predeclared state strata.

### RQ3: Can identified effects improve control?

**H3:** an effect-informed trigger policy improves task success or recovery at
equal expected intervention cost relative to always, never, and heuristic rules.

### RQ4: Do effects transfer across models?

**H4:** qualitative state moderation transfers, while effect magnitude depends on
baseline interface-error and self-recovery rates.

### RQ5: What carryover horizon is defensible?

**H5:** most verification benefit appears within three environment transitions;
longer windows add noise and treatment overlap.

## 4. Formal Setup

Index episodes by `i` and decision points by `t`. Let:

- `I_it` be an availability indicator;
- `S_it` be state features observed before randomization;
- `A_it in {0,1}` be the randomized intervention;
- `p_it = P(A_it=1 | H_it, I_it=1)` be the known randomization probability;
- `Y_i,t+k` be a proximal outcome over horizon `k`;
- `C_it` be intervention cost;
- `H_it` be observed trajectory history.

The primary marginal causal excursion effect is

`beta_k = E[Y_i,t+k(A_it=1) - Y_i,t+k(A_it=0) | I_it=1]`.

For moderator vector `f(S_it)`, define

`beta_k(S_it) = f(S_it)^T beta`.

The primary utility effect includes cost:

`U_i,t+k = Y_i,t+k - lambda_step * Steps - lambda_token * Tokens`,

where cost weights are fixed before the main analysis. Raw outcome and cost are
also reported separately.

## 5. Primary Intervention: Executable Verification

The sprint should study one intervention well.

### Treatment `A=1`

Run a deterministic, task-agnostic verifier after an eligible tool action. Return
one bounded structured observation:

- `PASS`;
- `FAIL(error_class, evidence_pointer)`;
- `INCONCLUSIVE`.

No extra model call is used in the primary treatment. If `FAIL`, the model can act
on the feedback in the next normal turn.

### Control `A=0`

Do not run or reveal verifier output at that point. The trajectory continues under
the same remaining resource policy.

### Availability `I=1`

A decision point is available only when:

- an executable tool action has changed or queried environment state;
- no verification has occurred within the last `k` transitions;
- at least `k+1` steps remain;
- the verifier can run deterministically;
- the episode is not already terminal.

Availability is logged before treatment assignment.

## 6. State Features and Predeclared Moderators

Use a small set of automatically measured features:

1. tool outcome class: success, explicit error, ambiguous/empty output;
2. repeated-action indicator within the last three turns;
3. state-change indicator from environment hash;
4. remaining step-budget fraction;
5. trajectory horizon so far;
6. unresolved-constraint count from deterministic task state;
7. previous verification result, if outside the cooldown window;
8. model-family indicator for the secondary transport analysis.

Do not use post-treatment trace information or an LLM-generated state label in the
primary estimator.

## 7. Proximal Outcomes

The primary outcome must be dense and deterministic.

### Primary proximal outcome

`Progress_k = 1` if, within the next `k=3` transitions, the agent reaches a new
task-valid state, resolves the triggering error, or completes the task without
introducing a new contract violation.

The exact rule is environment-specific but encoded by deterministic state and
grader functions.

### Secondary proximal outcomes

- recurrence of the same tool error within three transitions;
- repeated-action loop within three transitions;
- number of valid state transitions;
- constraint-retention score;
- token and step cost;
- eventual task success;
- time to recovery.

Final success is important for policy evaluation but too sparse to be the only MRT
outcome.

## 8. Randomization Design

### Primary design

At each available decision point:

`A_it ~ Bernoulli(0.5)`.

Use a cooldown of `k=3` transitions after assignment. No new primary treatment is
randomized during cooldown, preventing overlapping proximal windows.

### Stratification

To maintain balance in important states, randomize within predeclared strata:

- explicit error versus no explicit error;
- remaining budget above versus below 40%;
- model family.

Randomization probabilities remain known and bounded in `[0.35, 0.65]` if adaptive
balancing is used. A simple fixed 0.5 probability is preferred for the first paper.

### Safety override

If a task has a deterministic safety-critical invariant, verification may be
mandatory. Such points have `I=0` for the trial and are analyzed separately. Never
randomize an intervention whose absence could cause irreversible external harm.

## 9. Estimation

Use weighted and centered least squares (WCLS):

`sum_i,t I_it w_it [Y_i,t+k - g(S_it)^T alpha - (A_it-p_it) f(S_it)^T beta]^2`,

where `w_it` is the inverse randomization weight if probabilities vary. Use
episode-clustered sandwich standard errors or an episode bootstrap.

### Estimands

- marginal effect `beta_0`;
- error-state interaction;
- remaining-budget interaction;
- repeated-action interaction;
- model-family interaction;
- utility effect after intervention cost.

### Sensitivity analyses

- proximal horizons `k=1,3,5`;
- cooldown lengths `2,3,5`;
- alternative progress definitions;
- exclusion of inconclusive verifier outputs;
- stabilized versus unstabilized weights;
- per-domain versus pooled estimates;
- randomization-balance and positivity checks.

## 10. Sample Size and Power Workflow

Do not guess sample size from episode count alone. The unit of randomization is a
decision point, but outcomes are correlated within episodes.

### Pilot

- 40-60 episodes;
- target 200 available decision points;
- estimate availability rate, proximal-outcome prevalence, within-episode
  correlation, treatment activation cost, and effect variance.

### Main target

- 1,500-2,000 randomized decision points;
- at least 200 episodes;
- at least 50 treated and 50 control observations in each primary moderator
  stratum;
- simulation-based power using pilot-derived cluster size and outcome rates.

If the pilot yields fewer than five available points per episode, increase task
horizon or change the environment. Do not manufacture repeated randomizations by
shrinking the cooldown below the causal horizon.

## 11. Experimental Environments

Use two deterministic multi-step domains:

### Domain A: local workspace operations

Tasks require file discovery, structured edits, command execution, and artifact
verification. State hashes and test results provide deterministic progress labels.

### Domain B: stateful tool/service simulator

Tasks require multi-step transactions with explicit preconditions and state
changes. An offline `tau`-style environment is suitable if all external effects
are sandboxed.

Select 20-30 tasks per domain, emphasizing trajectories long enough to expose
multiple eligible decision points. Generate parameterized variants before the
trial and freeze them.

## 12. Models and Harness Conditions

### Models

- one low-cost model with higher interface-error rate;
- one stronger model from a different family.

Temperature, tool schema, and maximum steps are fixed. A third model is an
extension only after the primary effect is established.

### Policy-evaluation conditions

After estimating effects, compare on fresh episodes:

1. never verify;
2. always verify when available;
3. heuristic verify after explicit errors;
4. effect-informed policy;
5. oracle retrospective policy as an unattainable upper reference.

Match expected verification counts or include cost in utility so the learned
policy cannot win by simply intervening more often.

## 13. Effect-Informed Policy

Fit a deliberately simple policy from the MRT estimates:

`pi(S)=1` if `E[beta(S)] - lambda_cost * Cost(S) > tau`.

Use either:

- a sparse linear threshold using the predeclared moderators; or
- a shallow decision tree with depth at most three.

Freeze the policy before fresh evaluation. The paper should not use an expressive
black-box policy that obscures the identified effects.

## 14. Evaluation Metrics

### Identification quality

- randomization balance by state stratum;
- effective sample size;
- positivity violations;
- marginal and conditional effect estimates;
- cluster-robust interval width;
- sensitivity across proximal horizons.

### Policy value

- deterministic task success;
- progress transitions per episode;
- recovery probability;
- verification count;
- token, step, and wall-clock cost;
- utility at fixed intervention budget;
- policy regret relative to always/never/heuristic baselines.

## 15. Seventy-Two-Hour Kill Test

1. Instrument one model and one deterministic domain.
2. Define availability and `k=3` progress outcome.
3. Run until 200 decision points are logged.
4. Verify treatment probability, cooldown, missingness, and state balance.
5. Plot treatment-control differences by explicit-error and remaining-budget
   strata.

Continue only if:

- at least five eligible points occur per episode on average;
- proximal outcome prevalence is between roughly 10% and 90%;
- deterministic labels agree with manual trace inspection on at least 95% of a
  50-point audit;
- randomization and logging are correct;
- at least one predeclared state stratum shows plausible effect heterogeneity.

## 16. Required Baselines and References

- [The Micro-Randomized Trial](../reading-cards/2107.03544.md): primary design and
  analysis foundation.
- [Stratified MRT](../reading-cards/1711.03587.md): availability and
  history-dependent randomization foundation.
- [Cross-Component Interference](../reading-cards/2605.05716.md): static component
  factorial collision.
- [AgentSpec](../reading-cards/2606.14674.md): controlled scaffold composition.
- [Causal Agent Replay](../reading-cards/2606.08275.md): retrospective causal
  intervention collision.
- [Offline Harness Control](../reading-cards/2607.05458.md): policy-optimization
  collision without prospective randomization.
- [Towards a Science of AI Agent Reliability](../reading-cards/2602.16666.md):
  reliability outcome context.

## 17. Required Ablations

- proximal horizon `k`;
- cooldown length;
- deterministic verifier versus model-based verifier as a secondary experiment;
- randomization at all eligible states versus error-only randomization;
- marginal-only versus moderator model;
- effect-informed policy with and without cost term;
- one domain versus pooled domains;
- one model versus cross-model moderator;
- always/never/heuristic at matched intervention count.

## 18. Killer Figures

### Figure 1: state-conditional effect surface

Show verification effect by tool-outcome class and remaining-budget fraction with
cluster-robust intervals.

### Figure 2: global average hides opposing effects

Decompose the marginal effect into positive, null, and negative state strata.

### Figure 3: policy frontier

Plot task success or progress against verification count/cost for never, always,
heuristic, and effect-informed policies.

### Figure 4: causal trajectory case study

Show paired examples where verification interrupts an error cascade and where it
wastes the final useful step.

## 19. Expected Contributions

1. The first micro-randomized protocol for agent-harness interventions.
2. Definitions of availability, proximal progress, carryover, and intervention
   cost for tool-using agents.
3. State-conditional causal excursion effects with valid clustered uncertainty.
4. An effect-informed policy prospectively validated on fresh episodes.
5. An open sequential trace schema and randomization implementation.

## 20. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Too few decision points | Use longer deterministic tasks and one clearly available intervention. |
| Proximal outcome is subjective | Use state hashes, test results, and explicit error classes; no primary LLM judge. |
| Carryover invalidates analysis | Predeclare cooldown and conduct horizon sensitivity. |
| Intervention changes later availability | Treat availability as part of the observed history and use standard MRT estimands, not naive per-point differences. |
| Model cost is excessive | Decision points are dense within episodes; pilot one model and add the second after logging is validated. |
| Policy overfits effect estimates | Use a sparse threshold or shallow tree and a fresh evaluation set. |

## 21. Kill Criteria

Stop or downgrade if:

- availability is below five points per episode;
- progress labels require an LLM judge;
- randomization or cooldown cannot be implemented without changing normal agent
  behavior;
- positivity fails in primary state strata;
- conditional effects are flat with narrow intervals;
- the effect-informed policy cannot beat a matched-cost heuristic on fresh tasks;
- sequential interference cannot be bounded or explained.

## 22. Eighteen-Day Execution Plan

| Days | Deliverable |
|---|---|
| 1 | Availability, treatment, and proximal-outcome specification |
| 2 | Instrumented runner, randomization tests, and trace schema |
| 3 | 200-point pilot and go/no-go |
| 4 | Simulation-based power and frozen analysis plan |
| 5-8 | Main MRT data collection |
| 9-10 | WCLS analysis and sensitivity checks |
| 11 | Fit and freeze effect-informed policy |
| 12-13 | Fresh policy evaluation |
| 14 | Ablations and trace case studies |
| 15-16 | Paper draft and figures |
| 17-18 | Reproducibility, artifact, and statistical audit |

## 23. Resource Budget

- Pilot: 40-60 episodes and 200 decision points.
- Main MRT: 200-350 episodes and 1,500-2,000 decision points.
- Policy evaluation: 200-300 fresh episodes across four policies.
- No training GPUs are required unless a later learned policy is added.
- Store pre-randomization state, availability, probability, treatment, verifier
  output, proximal outcome, cost, cooldown, and episode cluster ID.

## 24. Reproducibility and Safety

- deterministic randomization seed stream separate from model seeds;
- immutable task variants and environment version;
- randomization balance report generated before outcome modeling;
- complete log of unavailable and safety-overridden points;
- no irreversible external tools;
- no post hoc moderator creation in the main table;
- analysis script run from raw event logs;
- fresh policy-evaluation task manifest and commit hash.

## 25. Proposed Paper Structure

1. Introduction: components have states, not one global effect.
2. Related work: static scaffold studies, causal replay, control policies, MRTs.
3. Sequential intervention formalization.
4. MRT-Harness design and trace schema.
5. State-conditional causal effects.
6. Effect-informed policy evaluation.
7. Carryover, limitations, and future interventions.

## 26. Longer-Term PhD Extension

Future papers can study multi-arm interventions, adaptive randomization, dynamic
treatment regimes, cross-model transport of excursion effects, safety-constrained
intervention policies, and offline-policy learning initialized from randomized
agent trajectories. This direction can become a causal methodology program for
agent runtime systems rather than a one-off benchmark.
