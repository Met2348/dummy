# Proposal 4: Harness-Curves

## Cross-Layer Stability Envelopes and Rank-Reversal Laws for LLM Agents

**Status:** feasible but crowded; suitable only if the contribution is a severity
response law and predictive stability analysis, not another fault catalog.

**Core claim:** nominal task success is an incomplete measure of agent capability.
Model-harness pairs have distinct degradation curves under small, semantics-
preserving controller corruptions, and these curves expose critical severities and
ranking reversals that can be predicted from clean-run trajectory features.

## 1. Abstract

LLM agents are evaluated primarily under clean execution conditions, even though
their harnesses repeatedly transform context, tool interfaces, state, feedback,
and memory. Recent work has introduced fault injection, error-injection metrics,
and domain-specific perturbations, but most studies report performance at a small
number of fault settings and do not estimate a common degradation response across
harness layers. We propose Harness-Curves, a controlled framework for measuring
stability envelopes of model-harness pairs. Four semantics-preserving corruption
families are parameterized by severity: context delay/drop, tool-error format
variation, state staleness, and verification noise. For each pair, we estimate
success-versus-severity curves, stability area, critical severity, recovery
probability, and model-harness ranking reversal. A monotone hierarchical response
model separates clean capability from degradation rate and predicts collapse from
clean-run trajectory features. A compact subset is then selected and validated on
held-out tasks. The intended scientific result is that the nominally strongest
agent is not necessarily the most stable, and that instability thresholds are
mechanistically linked to feedback dependence and recovery behavior. The work is
positioned as a science of harness stability rather than a larger fault benchmark.

## 2. Motivation and Novelty Boundary

Harnesses mediate every agent-environment interaction. Small controller defects
can therefore persist across turns:

- a delayed observation causes an action based on stale state;
- an unfamiliar tool error is treated as success;
- a context refresh drops an unresolved requirement;
- a false-positive verifier allows a wrong artifact to become the new baseline;
- an irrelevant memory item triggers a repeated detour.

The area is already active:

- [MAS-FIRE](../reading-cards/2602.19843.md) injects 15 fault types into
  multi-agent systems.
- [Claw-Eval](../reading-cards/2604.06132.md) combines error injection with
  completion, safety, and robustness metrics.
- [RepoMirage](../reading-cards/2605.26177.md) applies semantics-preserving
  repository-context perturbations.
- [Towards a Science of AI Agent Reliability](../reading-cards/2602.16666.md)
  defines a broad reliability profile.
- [Self-Healing Orchestrators](../reading-cards/2606.01416.md) evaluates recovery
  under controlled faults.

These papers occupy "inject faults and measure robustness." The remaining slot is
to estimate a shared **severity response** across controller layers, quantify
uncertainty in critical thresholds, and explain or predict rank reversal.

## 3. Research Questions and Hypotheses

### RQ1: Do model-harness pairs have distinct stability envelopes?

**H1:** clean success and stability area are only moderately correlated; some
high-performing pairs degrade more sharply than lower-performing pairs.

### RQ2: Are there reproducible critical severities?

**H2:** each corruption family exhibits pair-specific thresholds beyond which
success and recovery deteriorate nonlinearly.

### RQ3: Do rankings reverse under mild corruption?

**H3:** at least one clean model-harness ranking reverses at a severity below the
median critical severity.

### RQ4: Which trajectory mechanisms predict instability?

**H4:** repeated-action rate, dependence on unstructured error text, low clean
self-recovery, and high context-refresh frequency predict steeper degradation.

### RQ5: Can a compact subset preserve stability conclusions?

**H5:** a task-corruption subset selected on source pairs preserves pairwise
stability ordering on held-out model-harness pairs better than a random subset of
equal size.

## 4. Formal Objects

For model `m`, harness `h`, corruption family `c`, and severity `q`, let

`p_mhc(q) = P(Y=1 | m,h,c,q)`.

Define clean performance `p_mh(0)` and:

### Corruption error

`CE_mhc(q) = p_mh(0) - p_mhc(q)`.

### Normalized stability area

`SAUC_mhc = integral_0^1 p_mhc(q) dq / max(p_mh(0), epsilon)`.

### Critical severity

For tolerance `delta`,

`q*_mhc(delta) = inf{q: p_mhc(q) <= p_mh(0) - delta}`.

Use `delta=0.10` as the primary absolute success drop, with sensitivity at 0.05
and 0.20.

### Recovery curve

`R_mhc(q,k) = P(return to a valid progressing state within k transitions after corruption)`.

### Ranking reversal

Pairs `a` and `b` reverse under family `c` if

`p_a(0) > p_b(0)` but `p_ac(q) < p_bc(q)`

with a simultaneous interval excluding zero for both comparisons.

## 5. Corruption Design Principles

Every corruption must be:

- non-adversarial and representative of ordinary runtime variation;
- semantics-preserving at the task level;
- localized to one declared harness layer;
- deterministic conditional on seed and severity;
- monotone in a measurable physical parameter where possible;
- reversible and sandboxed;
- independent of the model's identity.

A corruption is rejected if it changes the correct task answer, removes all
possible solution paths, or quietly changes resource budgets.

## 6. Primary Corruption Families

Restrict the sprint to four families.

### C1: Context availability

Delay or omit a fraction of nonterminal observation fields while preserving the
underlying environment state.

Severity parameter:

- `q=0`: complete current observation;
- `q=1`: one-turn delay of one redundant field;
- `q=2`: one-turn delay of one task-relevant but recoverable field;
- `q=3`: two-turn delay or omission requiring active re-query.

Do not drop irrecoverable task instructions in the primary suite.

### C2: Tool-error representation

Keep the same error semantics while varying surface format:

- canonical structured code;
- paraphrased message with code retained;
- code omitted but meaning preserved;
- reordered/noisy wrapper with the same recoverable content.

This tests dependence on error syntax rather than tool correctness.

### C3: State staleness

Return a valid observation from `s` transitions earlier after a state-changing
action, while preserving a fresh-state query path.

Severity `q` is the staleness lag: 0, 1, 2, or 3 transitions.

### C4: Verification noise

Inject bounded false-positive or false-negative noise into a non-safety-critical
intermediate verifier. Final grading remains authoritative.

Severity is the flip probability: 0, 0.05, 0.15, 0.30. Use deterministic seeded
flips for paired comparisons.

## 7. Secondary Corruptions

Only after the primary study completes:

- irrelevant-tool addition;
- retry jitter;
- delayed corrective feedback;
- stale memory retrieval;
- duplicated context chunks;
- partial checkpoint fields.

These should not enter the main claim if added after inspecting primary results.

## 8. Semantic-Preservation Validation

Each transformed task passes four checks:

1. the original oracle solution still succeeds;
2. a corruption-aware scripted policy can recover;
3. the final grader and correct answer are unchanged;
4. a human audit of at least 20 examples per family confirms ordinary runtime
   plausibility and recoverability.

Report preservation failures and exclude the entire transformation rule if its
failure rate exceeds a predeclared threshold, such as 2%.

## 9. Response Model

Use a hierarchical monotone model:

`logit p_mhc(q) = alpha_task + mu_mh - g_c(q; theta_mhc)`,

where `g_c(0)=0` and `g_c` is nondecreasing. Candidate implementations:

- monotone I-splines with partial pooling;
- ordinal severity coefficients constrained to be nondecreasing;
- a simpler hierarchical logistic model with severity as ordered categorical.

The ordered-categorical model is the sprint-safe default. A smooth curve is useful
only if diagnostics and threshold intervals are stable.

### Variance components

Estimate:

- clean model effect;
- clean harness effect;
- model-harness interaction;
- corruption-family effect;
- model-harness-corruption interaction;
- task-level variance.

This quantifies whether the harness changes degradation rate beyond clean
capability.

## 10. Stability Predictor

Predict `SAUC` and critical severity from clean-run features only:

- malformed tool-call rate;
- explicit-error recovery rate;
- repeated-action rate;
- number of observation re-queries;
- average trajectory length;
- verifier dependence;
- context-compaction count;
- clean success;
- token and step budget utilization.

Use a regularized linear or shallow tree model with leave-one-pair-out validation.
Baselines are clean success alone and model identity alone. The predictor is a
secondary contribution but helps move the paper beyond benchmark construction.

## 11. Experimental Design

### Models

- 3 model families spanning different clean capability levels.

### Harnesses

- a minimal ReAct-style harness;
- a richer harness with context management, recovery, and verification.

Keep model prompts and tool semantics aligned as closely as possible across
harnesses.

### Tasks

- 20 deterministic tasks from two domains;
- parameterized variants frozen before evaluation;
- tasks must expose context, tool errors, state updates, and verification.

### Main matrix

`3 models x 2 harnesses x 4 families x 4 severities x 20 tasks = 1,920 episodes`.

Use one paired seeded run at temperature zero for the primary curve and repeat a
predeclared 25% subset to estimate stochasticity. If models remain stochastic at
temperature zero, use two repeats throughout and reduce tasks to 15 rather than
dropping a severity level.

## 12. Fractional Sprint Design

The 72-hour pilot:

- 2 models;
- 2 harnesses;
- C1 and C2;
- clean plus two nonzero severities;
- 12 tasks.

This is 144 episodes. Continue if:

- semantic preservation is at least 98%;
- at least one pair loses 10 absolute success points at mild/moderate severity;
- degradation differs across pairs beyond obvious clean-score floor effects;
- traces expose a plausible recovery mechanism;
- full runtime fits the remaining schedule.

If effects appear only at catastrophic severity, the stability-envelope claim is
weak and should be stopped.

## 13. Statistical Analysis

### Primary analysis

- hierarchical ordered-severity logistic model;
- task-clustered bootstrap for curve summaries;
- simultaneous bootstrap intervals for pairwise ranking differences;
- uncertainty interval for each critical severity;
- familywise rank-reversal declaration using max-statistic bootstrap.

### Floor and ceiling correction

Report raw curves and normalized degradation. Exclude no pair, but explicitly
model clean success so weak models are not labeled stable merely because they
cannot decline much further.

### Missing and failed runs

Timeouts and harness crashes count as failures for task success and receive a
separate process label. Do not silently rerun only failed corrupted conditions.

## 14. Compact Stability Subset

After the full source-pair matrix:

1. score each task-corruption item by discrimination among pair stability curves;
2. select a fixed-size subset using source pairs only;
3. freeze the subset;
4. test stability-rank correlation on a held-out model-harness pair;
5. compare with 1,000 random subsets of equal size.

Metrics:

- Spearman rank correlation of `SAUC`;
- preservation of detected rank reversals;
- mean absolute critical-severity error;
- evaluation-cost reduction.

This is secondary. It must not replace the main severity-law result.

## 15. Required Baselines and Collisions

- [MAS-FIRE](../reading-cards/2602.19843.md): fault taxonomy and process-level
  reliability baseline.
- [Claw-Eval](../reading-cards/2604.06132.md): repeated reliability and
  error-injection baseline.
- [RepoMirage](../reading-cards/2605.26177.md): semantics-preserving
  context-perturbation baseline.
- [Towards a Science of AI Agent Reliability](../reading-cards/2602.16666.md):
  consistency, robustness, predictability, and safety metrics.
- [Self-Healing Orchestrators](../reading-cards/2606.01416.md): controlled recovery
  under fault injection.
- [Harness-Bench](../reading-cards/2605.27922.md): complete-configuration harness
  evaluation rather than controlled severity.
- [Stop Comparing LLM Agents](../reading-cards/2605.23950.md): ranking-reversal and
  perturbation motivation.

## 16. Required Ablations

- raw success curve versus normalized degradation;
- ordered categorical versus monotone spline response;
- clean-success-only versus trajectory-feature stability predictor;
- one corruption family at a time versus pooled stability score;
- model-only, harness-only, and model-harness interaction terms;
- exact structured error versus paraphrased error;
- fresh-state re-query available versus unavailable;
- compact subset selected on source pairs versus random subset.

## 17. Killer Figures and Tables

### Figure 1: stability envelopes

Show success versus severity for all six model-harness pairs. Highlight critical
severity and uncertainty.

### Figure 2: clean capability versus stability

Scatter clean success against normalized `SAUC`; label ranking reversals.

### Figure 3: mechanism-to-collapse trace

Show how a stale observation or changed error format creates repeated invalid
actions in one harness but triggers recovery in another.

### Figure 4: compact subset preservation

Compare full and compact stability rankings on the held-out pair.

### Table 1: corruption contracts

For every family and severity, state what changes, what remains invariant, and how
recoverability is checked.

## 18. Expected Contributions

1. A versioned, semantics-preserving cross-layer corruption specification.
2. Stability envelopes, critical-severity estimates, and simultaneous rank-
   reversal uncertainty for model-harness pairs.
3. Evidence that clean capability and harness stability are distinct.
4. A trajectory-based predictor of degradation using clean runs.
5. A validated compact stability subset.

The main-track bar requires contributions 2 and 3. A released corruption suite
without a scientific stability result is not enough.

## 19. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Another benchmark paper | Center the paper on severity curves, thresholds, and predictive rank reversal. |
| Corruption changes semantics | Oracle replay, scripted recovery, and human preservation audit. |
| Floor effects make weak models look stable | Model clean performance jointly and report normalized plus raw curves. |
| Matrix is too large | Keep four families and four severities; reduce task count only after pilot. |
| Curves are non-monotone | Report ordered empirical response and investigate recovery activation rather than forcing smoothness. |
| No mild-severity effect | Stop; catastrophic-only failure is not the proposed result. |

## 20. Kill Criteria

Stop or downgrade if:

- semantic preservation falls below 98%;
- clean-to-mild degradation is smaller than rollout noise for all pairs;
- no ranking reversal or meaningful degradation-rate difference appears;
- critical severity cannot be estimated within useful uncertainty;
- the stability predictor does not beat clean success alone;
- the work's only novelty is adding fault types to MAS-FIRE or Claw-Eval.

## 21. Eighteen-Day Execution Plan

| Days | Deliverable |
|---|---|
| 1 | Corruption contracts and semantic-preservation tests |
| 2 | Injector implementation and deterministic replay tests |
| 3 | 144-episode pilot and go/no-go |
| 4-7 | Full context and tool-error curves |
| 8-10 | State-staleness and verification-noise curves |
| 11 | Hierarchical response model and threshold intervals |
| 12 | Rank-reversal analysis and trace mechanisms |
| 13 | Compact-subset selection and held-out validation |
| 14 | Ablations and final figures |
| 15-16 | Paper draft |
| 17-18 | Artifact, reproducibility, and semantics audit |

## 22. Resource Budget

- Pilot: 144 episodes.
- Main grid: 1,920 episodes plus a 25% repeated subset.
- No training GPUs required.
- Use local deterministic environments and cache uncorrupted baseline traces only
  where replay does not alter agent interaction.
- Store corruption seed, layer, severity, original observation, transformed
  observation, recovery action, state hashes, and final outcome.

## 23. Reproducibility and Safety

- schema-versioned corruptions;
- deterministic transformation seeds;
- original and transformed event logs;
- no corruption of final safety checks or irreversible external systems;
- oracle solution replay for every transformed task;
- full timeout/crash accounting;
- held-out pair declared before compact-subset selection;
- script-generated response curves and rank tests.

## 24. Proposed Paper Structure

1. Introduction: clean capability is not stability.
2. Related work: agent reliability, fault injection, context perturbations.
3. Corruption contracts and stability estimands.
4. Experimental protocol and response model.
5. Stability envelopes and critical severities.
6. Rank reversals, mechanisms, and prediction.
7. Compact subset, limitations, and release.

## 25. Longer-Term PhD Extension

The benchmark can evolve into a reliability-science program: learned stability
predictors, online drift detection, repair policies targeted at specific envelope
failures, safety-critical harness certification, and cross-domain studies of how
runtime contracts fail as models and tools change.
