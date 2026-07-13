# 历史版本：FORECAST-H Novelty Collision Map

> **状态：历史版本。** 2026-07-11 已逐页复核 SEAGym、LIFE-HARNESS 与九篇近邻。当前边界以[全文碰撞复核](../foundations/notes/fulltext-collision-reassessment-zh.md)、[新增近邻精读](../foundations/notes/new-collisions-and-method-foundations-zh.md)和 [DR-0002](../decisions/0002-trace-h-reassessment-zh.md) 为准。下方“occupied”只表示宽泛 claim 已有先例，不表示应回避其尚未完成的 estimand。

## Proposed Claim Under Audit

> Before running interventions on a newly introduced model, predict the sign and magnitude of each atomic harness effect from intervention mechanism, clean target behavior, task structure, and operating point; then choose reuse, retune, or reject under explicit negative-transfer risk.

The claim survives only as a conjunction. Nearly every individual ingredient is already occupied.

## Occupied Territory

| Candidate claim | Closest collisions | Verdict |
|---|---|---|
| Harnesses transfer across models | Life-Harness, AHE, Meta-Harness, HarnessFix, CoEvoSkills, Trace2Skill | Fully occupied |
| Harness transfer can fail or reverse | SEAGym, Probe-and-Refine, MemDelta, SkillCraft | Fully occupied |
| Harness effects interact with model choice | Stop Comparing, More Is Not Always Better, MemDelta | Fully occupied |
| Harness components can be isolated and ablated | More Is Not Always Better, MemDelta, NLAH | Occupied |
| Evaluate on held-out models | Meta-Harness, HarnessBridge, Rosetta Memory, PACE | Occupied |
| Predict unseen model performance | PACE | Occupied |
| Predict unseen model-scaffold combinations | Agent Psychometrics | Occupied when both components were seen and interaction is assumed absent |
| Learn a policy on held-out tasks and freeze it across models | What Should a Skill Remember? | Occupied for cost-aware skill-rewrite selection |
| Use target model profiles to adapt an interface | Rosetta Memory | Occupied for learned memory presentation |
| Build a source-target skill utility matrix | SkillCraft, Raw Experience to Skill Consumption | Occupied |
| Use utility-derived descriptors to select better skills | Raw Experience to Skill Consumption | Partly occupied; its rubric is not prospectively validated on a sealed target model |
| Optimize a harness safely under uncertainty | HARBOR | Occupied within one agent-task-model triple; it explicitly reruns after a model swap |

## Surviving White Space

No checked paper jointly provides all of the following:

1. atomic, versioned executable harness interventions;
2. paired target effects under a clean baseline and a matched resource policy;
3. target-model features measured without observing target intervention outcomes;
4. a prediction file committed before a new model's intervention table is opened;
5. signed and numerical effect forecasts, not just absolute performance or a strategy label;
6. calibrated negative-transfer risk and an explicit abstain/retune action;
7. evaluation on target models and task slices absent from feature and hyperparameter selection.

This is the defensible contribution. Removing any one of items 3-6 makes the work collide with a recent paper.

## Dangerous Reviewer Comparisons

### Versus More Is Not Always Better

Reviewer attack: "They already study atomic scaffold effects and model-dependent sign changes."

Required answer: that paper maps a small retrospective prompt-component lattice. We forecast effects of predeclared executable interventions on sealed model-task cells, score calibration and decision regret, and include resource/activation controls. Reproducing its descriptive finding is not a contribution.

### Versus What Should a Skill Remember?

Reviewer attack: "They already learn an intervention selector and transfer it across models."

Required answer: its selector uses task/skill structure to choose a rewrite family and is not conditioned on the target model. It does not predict the target-specific treatment effect or negative-transfer probability, and its cross-stack evaluation reuses the full task pool after policy fitting. Our target model and audit tasks are sealed.

### Versus Raw Experience to Skill Consumption

Reviewer attack: "They already predict skill utility from validated dimensions."

Required answer: their utility rubric is mined from high-gap pairs and evaluated on the same pair collection before being used to guide extraction. We require a prospectively committed effect prediction on an unseen model, with no target intervention outcomes in descriptor construction.

### Versus PACE and Agent Psychometrics

Reviewer attack: "Agent outcome prediction is already solved."

Required answer: PACE predicts an absolute score under one common agent framework; Agent Psychometrics predicts combinations whose model and scaffold were individually seen using an additive model. Neither estimates the causal delta from switching an atomic harness component on a fully unseen model.

### Versus Rosetta Memory

Reviewer attack: "Profile-conditioned cross-model adaptation is already available."

Required answer: Rosetta learns to rewrite memory for a target profile. Our task is a pre-intervention decision: retain the unchanged component, retune it, or reject it, with the forecast audited against an unopened target effect table.

## Hard Stop Conditions

Downgrade or abandon the main novelty claim if a new paper is found that has all three:

1. atomic model-external interventions;
2. sealed or genuinely unseen target models;
3. precommitted prediction of target intervention deltas or negative-transfer risk.

Also stop if the pilot cannot produce estimable activation, heterogeneity, or budget-matched paired effects. A polished predictor over near-zero deltas is not a main-conference result.
