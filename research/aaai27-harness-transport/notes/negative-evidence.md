# Negative Evidence and Claim Discipline

## Evidence Against an Easy Paper

1. **"Some harnesses transfer and some do not" is already known.** SEAGym and Probe-and-Refine provide explicit sign/collapse evidence; multiple skill papers provide both broad gains and negative cells.
2. **Atomic ablation is no longer enough.** The CCI paper runs a full 32-condition component lattice, and MemDelta varies one factor at a time.
3. **A held-out model is no longer enough.** Meta-Harness, HarnessBridge, Rosetta Memory, and PACE all evaluate unseen or held-out models.
4. **A selector is no longer enough.** Cost-aware skill rewriting freezes a learned selector before held-out tasks and cross-stack evaluation.
5. **A transfer matrix is no longer enough.** SkillCraft and the skill-lifecycle study already cross creators/extractors with target consumers.

## Terms That Must Not Be Taken at Face Value

- Meta-Harness calls five model columns held-out, but GPT-OSS-20B was used to select the harness; four models are actually unseen during search.
- AgentTether's cross-model result reuses the repair architecture, while diagnoses and guidance are regenerated from each model's traces. It is method reuse, not artifact transfer.
- TTHE's cross-model check reruns test-time evolution per model. Its paper explicitly says the transductive protocol does not establish forward generalization.
- AHE reports positive transfer but acknowledges that the step budget and timeout were fitted to the source operating point.
- HarnessBridge reports single-run trends and notes that gains shrink when the target baseline is already efficient.
- The skill-lifecycle rubric is utility-informed and tested on the same high-gap pair collection; it is not a clean prospective target test.

## Evidence That Can Falsify Our Mechanism Story

- CoEvoSkills transfers Opus-evolved packages to six models with +35.4 to +44.1 pp gains. If our executable interventions also transfer uniformly, a heterogeneous-effect predictor may have no signal.
- Life-Harness improves 116/126 model-environment settings. Environment-contract interventions may be nearly universally positive in deterministic domains.
- More Is Not Always Better finds that a simple main-effects model can outperform an interaction model in its small factorial grid. A complicated predictor may overfit.
- Rosetta Memory adapts successfully to unseen models from lightweight profiles. Direct target-conditioned adaptation may dominate prediction-and-abstention for mutable textual components.
- PACE achieves strong absolute-score prediction with cheap atomic probes. Capability-only features may be hard to beat.

## Required Countermeasures

- Include universal-positive baselines and report when prediction adds no value.
- Separate deterministic contract components from prose/control components before fitting.
- Predeclare source-copy and global-effect baselines as the primary comparisons.
- Use target baseline features only; never derive features from target intervention trajectories.
- Report null and dormant interventions rather than filtering them after seeing outcomes.
- Treat single-seed or small-task literature findings as motivation, not settled effect sizes.
- Keep a live collision log until submission freeze.

## Current Verdict

The direction remains promising, but only after the reframing to prospective, target-conditioned effect forecasting with abstention. The original broad transport claim is no longer novel enough for a main-conference submission.

