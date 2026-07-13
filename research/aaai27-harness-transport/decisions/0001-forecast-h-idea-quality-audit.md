# DR-0001: FORECAST-H Idea Quality and AAAI-27 Feasibility Audit

- **Date:** 2026-07-11
- **Owner:** project team
- **Status:** superseded by DR-0002 for the method center; deadline and evidence-gap audit retained
- **Review date:** after the 72-hour kill test, no later than 2026-07-14
- **Supersedes:** informal recommendation in the original Proposal 2

> **2026-07-11 更新：** 中文 [DR-0002](0002-trace-h-reassessment-zh.md) 已将通用 FORECAST-H predictor 重构为 TRACE-H。随后又完成 [SEAGym、LIFE-HARNESS 与九篇近邻全文复核](../foundations/notes/fulltext-collision-reassessment-zh.md)，进一步修正了碰撞边界和评分。本文保留为历史审计，不再是当前正式 Proposal、文献判断或评分入口。

> **当前系统级表述：** TRACE-H 已进一步从 patch effect prediction 转为 cross-executor executable harness policy transport。导师阅读请先看[约 600 字完整陈述](../notes/trace-h-idea-600zi-zh.md)与[术语说明](../notes/trace-h-terminology-guide-zh.md)，再看[Policy Transport 正式 Proposal](../proposals/trace-h-policy-transport-proposal-zh.md)和[DR-0003B](0003b-trace-h-policy-transport-pivot-zh.md)。本文后续英文收缩性评价仅作历史风险记录。

## Executive Decision

FORECAST-H is a good research idea but not yet a good paper. Its problem evidence, timeliness, and potential narrative are strong; its own empirical evidence is currently zero, its surviving novelty is a narrow conjunction, and the AAAI-27 schedule makes the full design high risk. Proceed for 72 hours to test whether active, budget-clean cross-model effect heterogeneity exists. Commit to a full AAAI-27 paper only if that pilot passes all gates. Otherwise pivot immediately rather than submit a descriptive transport study.

## Deadline Reality

The [AAAI-27 official timetable](https://aaai.org/conference/aaai/aaai-27/) gives:

- abstract deadline: 2026-07-21, 11:59 PM UTC-12;
- full paper deadline: 2026-07-28, 11:59 PM UTC-12;
- supplementary material and code: 2026-07-31.

From 2026-07-11, this leaves roughly 10 days to the abstract and 17 days to the paper. The original "three weeks" assumption is no longer accurate.

The most recent official main-track statistic is approximately 17.5% overall acceptance for AAAI-26, reported in the [AAAI-26 opening ceremony](https://aaai.org/wp-content/uploads/2026/01/AAAI-26-Opening-Ceremony.pdf). AAAI's stated criteria emphasize significance, novelty, empirical/theoretical soundness, community relevance, clarity, responsible practice, and reproducibility; the project is evaluated below against those dimensions rather than against topical fashion alone.

## Scorecard

Scores distinguish **current state** from **credible ceiling if the sealed-target study succeeds**.

| Dimension | Current | Successful-study ceiling | Confidence | Bottom line |
|---|---:|---:|---|---|
| Neatness | 7.2/10 | 9.0/10 | medium | Clean operational question, but current design contains too many moving parts |
| Excitement | 7.8/10 | 9.0/10 | medium | Fast-moving problem with visible positive and catastrophic negative cases |
| Problem evidence | 9.0/10 | 9.2/10 | high | Literature strongly establishes heterogeneous transfer and model-harness interaction |
| Evidence for proposed solution | 2.5/10 | 8.0/10 | high about current weakness | No project-owned experiment yet demonstrates predictability |
| Novelty | 7.0/10 | 8.5/10 | medium-low | Survives only as target-conditioned prospective effect prediction plus abstention |
| Technical depth | 6.3/10 | 8.2/10 | medium | Can be substantive if effect estimation, calibration, and decision regret are rigorous |
| Workload feasibility | 3.8/10 | 6.5/10 after scope cut | medium | Seventeen days is the largest risk |
| AAAI community fit | 7.5/10 | 8.5/10 | medium | Strong agent/evaluation relevance; must look like AI methodology, not harness bookkeeping |
| Reproducibility potential | 8.2/10 | 9.2/10 | high | Seal protocol, schemas, paired runs, and complete traces are natural strengths |
| Submission readiness | 2.0/10 | 8.0/10 | high | Literature and protocol are ready; results and paper are not |

**Overall idea quality now:** approximately 7.4/10.

**AAAI-ready paper quality now:** approximately 2/10. These should not be conflated.

## 1. Neatness

### Why It Is Neat

The practical question fits in one sentence:

> When the base model changes, which existing harness components should be reused, retuned, or rejected before running a full target evaluation?

It has a clean mathematical object, the target effect `Delta(target, intervention)`, and a clean prospective test: commit predictions, open the target table, and score them. The source-target structure produces one obvious killer figure: predicted versus observed target effects, colored by reuse/retune/reject decision.

The idea also joins three normally separate communities in a coherent way:

- agent harness engineering supplies the interventions;
- causal/experimental design supplies paired effects and atomicity;
- uncertainty-aware prediction supplies calibration and abstention.

### Why It Is Not Yet Fully Neat

The current proposal tries to carry too much:

- four or five intervention mechanisms;
- multiple models and domains;
- causal language;
- hierarchical modeling;
- budget matching;
- target fingerprints;
- mechanism decomposition;
- a benchmark/protocol contribution;
- a deployment decision rule.

In seven AAAI technical pages, this can read as an evaluation suite with many knobs rather than one sharp method.

### Required Simplification

The main paper should contain only:

1. three atomic intervention classes;
2. one baseline-only target fingerprint;
3. one regularized effect predictor;
4. one reuse/retune/reject rule;
5. one sealed target audit;
6. one activation-normalized mechanism analysis.

Move broad taxonomies, a fourth intervention, complex Bayesian variants, and the second-order benchmark story to the appendix. If the method cannot be explained in one diagram and one equation block, neatness falls below 7.

## 2. Excitement

### Sources of Excitement

The literature contains a genuine tension rather than a manufactured gap:

- Life-Harness and CoEvoSkills report broad positive transfer.
- SEAGym shows cross-backend sign reversals.
- Probe-and-Refine reports catastrophic Qwen-to-Nemotron guidance transfer.
- SkillCraft and the skill-lifecycle study show creator-consumer compatibility and negative efficiency effects.
- AHE explicitly identifies source-fitted operating-point coupling.

This makes the result space interesting in either direction. A successful predictor would convert scattered observations into a model-upgrade decision rule. A well-executed failure could still reveal that baseline-only fingerprints are insufficient, but that negative outcome is less likely to support the intended deadline submission.

The topic is also moving extremely quickly: TTHE was submitted on 2026-07-09. That increases attention and collision risk simultaneously.

### Limits on Excitement

Broad AAAI reviewers may see "harness" as engineering vocabulary around prompts and tools rather than a fundamental AI problem. The work becomes exciting to them only if it demonstrates a general phenomenon:

- effects are non-monotone across model capability;
- a low-cost behavioral fingerprint predicts that non-monotonicity;
- calibrated abstention prevents measurable deployment loss.

A paper whose result is only a heatmap of four prompts across three models has excitement below 6/10.

### Excitement Trigger

At least one of these is needed:

1. a clear sign reversal predicted correctly before unsealing;
2. a substantial reduction in decision regret over source-copy;
3. a surprising mechanism, such as stronger models needing less recovery but being harmed more by prose control;
4. a simple transport law that holds across intervention classes.

## 3. Evidence Strength

### Evidence That the Problem Exists: Strong

The [transport evidence matrix](../notes/transport-evidence-matrix.md) contains multiple direct positive, negative, atomic, and held-out precedents. The problem statement does not depend on one paper or one benchmark.

Particularly strong evidence:

- controlled 3 x 3 model-harness interaction and ranking reversals;
- 2^5 component factorial evidence with model-scale sign changes;
- one-factor memory comparisons with cross-model ranking reversal;
- catastrophic static guidance transfer with detailed trajectory mechanism;
- multiple broad positive artifact-transfer results that rule out a universal-negative story.

### Evidence That Target Effects Are Predictable: Moderate Indirect Evidence

PACE predicts absolute agent performance on held-out models. Agent Psychometrics predicts model-scaffold combinations when both components were seen. Cost-aware skill rewriting predicts a rewrite policy from task/skill features. Rosetta Memory adapts an interface using unseen target profiles.

Together these show structured predictability is plausible, but none establishes that the sign or magnitude of an atomic harness delta can be forecast for a sealed new model.

### Project-Owned Evidence: Absent

At present there are:

- no pilot trajectories;
- no measured activation rates;
- no source response surface;
- no fitted predictor;
- no sealed target;
- no uncertainty or regret result.

Therefore any claim about expected predictive success remains a hypothesis. This is the largest difference between a strong proposal and a strong submission.

### Evidence Grade

- motivation and problem existence: **A-**;
- novelty boundary: **B+**, because the search is deep but the frontier is unstable;
- feasibility of measurement: **B**;
- feasibility of prediction: **C**;
- evidence for the final claimed method: **F / not yet tested**.

## 4. Novelty

### Generic Version: Not Novel

The following claims should not appear as primary novelty:

- harnesses matter;
- harnesses can transfer;
- transfer can be negative;
- components interact with models;
- held-out models should be evaluated;
- a transfer matrix or component ablation is useful;
- a strategy selector can be frozen before evaluation.

All are occupied by recent work.

### Surviving Novel Conjunction

The [collision map](../notes/collision-map.md) supports a narrower novelty claim:

> Predict, before observing any target intervention outcome, the signed effect of a versioned atomic harness intervention on a fully unseen model-task cell using only source effects, intervention contracts, and baseline-only target features; then evaluate a calibrated reuse/retune/reject decision.

The most novel pieces are not the regression family. They are:

1. effect prediction instead of absolute performance prediction;
2. a genuinely unseen target model rather than a new combination of seen identities;
3. intervention outcome sealing;
4. explicit negative-transfer abstention;
5. matched-budget and activation-aware estimands.

### Novelty Fragility

The novelty score has medium-low confidence because:

- the field adds papers daily;
- several 2026 papers already combine two or three required ingredients;
- a contemporaneous preprint could independently reach the same conjunction;
- reviewers may treat the conjunction as an incremental composition of known ideas unless the empirical result is striking.

The novelty should be described as "the first prospective target-effect audit we find" only after a final search, not as an unconditional first-ever claim.

## 5. Technical Depth

### What Makes It Methodological

To avoid looking like benchmark bookkeeping, the paper needs a precise prediction and decision formulation:

- paired intervention effect and uncertainty;
- pre-intervention target covariates;
- regularized interaction model with leave-model-out validation;
- calibrated probability of negative transfer;
- risk-controlled decision rule;
- regret against an oracle and fixed-policy baselines.

Activation normalization can add a mechanistic decomposition:

`effect = activation opportunity x conditional recovery - interference/resource cost`.

This is a stronger technical center than a large collection of agent runs.

### Depth Risks

- Too few source models make a learned predictor statistically unconvincing.
- Too many features make leakage and overfitting almost certain.
- A complicated Bayesian model with six target cells will look decorative.
- Calling randomized paired comparisons "causal transport" may trigger avoidable demands for stronger identification theory.
- If target fingerprints are mostly clean success rate, reviewers may conclude that nearest-capability baselines explain everything.

Use a small, auditable model and win against strong simple baselines. Complexity is not a substitute for sample size.

## 6. Workload

### Full Desired Design

A credible minimum full design is approximately:

- 3 source/validation models plus 1 sealed target model;
- 2 deterministic task domains;
- baseline plus 3 interventions;
- 20 tasks per domain;
- temperature-zero primary runs, selective repeats.

Primary episodes: `4 x 2 x 4 x 20 = 640`, before smoke tests, reruns, and ablations. With one optional repeat on 20% of cells, practical volume is about 750-800 episodes.

### Work Breakdown Under Heavy AI Assistance

| Work package | Focused human hours | Wall-clock risk | AI assistance value |
|---|---:|---|---|
| Environment and paired runner | 12-24 | high | high for implementation, low for debugging external APIs |
| Three atomic interventions and tests | 8-16 | medium | high |
| Pilot and trajectory audit | 8-12 | medium | medium-high |
| Source matrix execution | 8-16 supervision | high | low for API latency and quota |
| Feature extraction and predictor | 10-18 | medium | high |
| Seal and target execution | 6-10 | high | medium |
| Statistics, figures, robustness | 12-20 | medium | high |
| Seven-page paper and appendix | 24-40 | high | high for drafting, medium for scientific compression |
| Final related-work refresh and claim audit | 6-10 | medium | high |

Estimated focused human effort: roughly **94-166 hours**, plus model execution wall time. Heavy AI assistance can reduce coding and drafting time, but it cannot remove experiment latency, quota failures, validity checks, or author judgment.

### Feasibility by Starting State

- Existing stable environment, model credentials, and runner: difficult but plausible.
- Environment available but runner incomplete: high risk.
- No benchmark environment or API access currently working: AAAI-quality completion is improbable.
- Attempting two domains before the pilot passes: poor resource allocation.

### Scope Discipline

The second domain should be dropped before reducing the number of models below four. The core claim is model replacement; model diversity is more valuable than domain breadth.

If the full 640-episode design cannot start by 2026-07-16, the honest strategy is to target a later venue or submit only if an already-working runner makes compressed execution possible.

## 7. Community Acceptance

### Why the Community May Accept It

- Model upgrades are common and harness retuning is expensive.
- Agent evaluation currently conflates model and harness effects.
- The work connects evaluation, uncertainty, causal-style experimentation, and agent systems.
- A precommitted sealed test is unusually credible in a fast-moving empirical area.
- Complete traces, contracts, and schemas create a useful reusable artifact.

### Why Reviewers May Reject It

1. **"This is only an empirical benchmark."** No clear algorithmic contribution beyond regression.
2. **"There are too few models."** Three source models cannot support a transport law.
3. **"The interventions are arbitrary."** Results may not generalize beyond hand-picked components.
4. **"The target is not truly unseen."** Public model knowledge or target probes may leak information.
5. **"The predictor is unnecessary."** Source-copy or clean capability may perform equally well.
6. **"The result is contemporaneous/incremental."** Recent skill and harness papers already cover most ingredients.
7. **"The scope exceeds the evidence."** Two domains and four intervention types with sparse cells invite overclaiming.
8. **"Agent infrastructure noise dominates."** API versions, timeouts, and benchmark execution may undermine reproducibility.

### Probability Ranges

These are subjective conditional forecasts, not measured frequencies. They are anchored to the approximately 17.5% AAAI-26 main-track base rate and assume competent writing and no policy violation.

| Outcome at submission | Estimated AAAI-27 acceptance probability | Reason |
|---|---:|---|
| Current state, forced submission with no project-owned results | 0-3% | Proposal, not a paper |
| Pilot only; descriptive heterogeneity, no sealed prediction | 5-12% | Relevant but collides with existing work |
| Full matrix; weak predictor tied with source-copy | 4-10% | Main claim fails |
| One sealed target; modest predictive lift; limited mechanism | 12-22% | Near conference base rate, vulnerable to sample-size objections |
| Sealed target; clear lift, calibrated abstention, mechanism evidence, complete artifact | 25-40% | Stronger than base rate, but still exposed to novelty and scope concerns |
| Same as above plus second independent target or prospective replication | 35-50% | Much stronger external validity, probably unrealistic by this deadline |

The probability of completing the strongest feasible tier by 2026-07-28 is itself limited. Conditional on an already working environment and model access, estimate **25-40%**. Without them, estimate **below 15%**. Therefore the unconditional chance of an AAAI acceptance from today's state is materially lower than the 25-40% best-case paper-quality range.

### Broader Community Reception

The likely qualitative response if executed well:

- agent-systems researchers: high interest;
- evaluation/benchmark researchers: high interest if sealing and uncertainty are rigorous;
- causal inference researchers: cautious unless causal terminology is disciplined;
- general AAAI reviewers: moderate interest, strongly dependent on simple framing and methodological depth;
- practitioners: high interest in reuse/retune/reject outputs.

## 8. Simulated Reviewer Score Profile

Using a rough 1-10 scale:

| Review dimension | Weak completed paper | Target paper |
|---|---:|---:|
| Significance | 6 | 8 |
| Novelty | 5 | 8 |
| Soundness | 5 | 8 |
| Empirical evaluation | 5 | 8-9 |
| Clarity | 6 | 8 |
| Relevance | 7 | 8 |
| Reproducibility | 7 | 9 |
| Overall | 5, reject | 7-8, accept-leaning |

The transition from reject to accept is almost entirely controlled by the sealed-target result and whether it beats simple baselines.

## 9. Go, Pivot, and Stop Gates

### Gate 1: 72-Hour Pilot

Go only if:

- at least two interventions activate on at least four tasks per model;
- at least one intervention has a sign disagreement or three cross-model discordant paired outcomes;
- the difference survives basic budget matching;
- one baseline-only feature or activation statistic has a plausible mechanistic link.

### Gate 2: Source-Surface Readiness by 2026-07-16

Go only if:

- four model endpoints and one deterministic domain run reliably;
- the episode schema is complete with no silent missing telemetry;
- projected 640-episode cost and wall time fit the remaining window;
- the abstract can state a specific method without promising unseen results.

### Gate 3: Predictor Freeze by 2026-07-21

Go only if leave-model-out validation beats global-mean and source-copy on at least two primary metrics. Otherwise pivot the paper to measurement or defer submission.

### Gate 4: Sealed Target by 2026-07-23

The target must remain unopened until the prediction file is committed. If the predictor fails after unsealing, do not relabel a retrospective analysis as prospective.

## 10. Recommended Decision

**Conditional go for 72 hours; no unconditional AAAI commitment yet.**

The idea is neat and potentially exciting enough for AAAI. The literature evidence for the problem is unusually strong. The novelty is real but narrow, and the work is much larger than it first appears. The deadline, not the intellectual quality, is the dominant negative factor.

The rational strategy is:

1. run the 96-episode kill test immediately;
2. make model and environment reliability the first engineering objective;
3. preserve the four-model axis even if the second domain must be dropped;
4. freeze a simple predictor by the abstract deadline;
5. defer to a later venue if the pilot or source-copy comparison fails.

## Change Log

- 2026-07-11: Initial audit created from the 68-paper isolated corpus and AAAI official timetable/review information.
