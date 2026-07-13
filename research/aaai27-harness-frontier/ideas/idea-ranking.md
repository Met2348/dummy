# Five-Idea Ranking and Sprint Decision

Scores are 1-5. A high execution-risk score is bad; other high scores are good.

| Rank | Idea | Novelty | Importance | 18-day feasibility | Main-conference ceiling | Execution risk | Recommended role |
|---:|---|:---:|:---:|:---:|:---:|:---:|---|
| 1 | Prospective Harness Transport | 3 | 5 | 3 | 5 | 4 | AAAI-27 primary / PhD spine |
| 2 | MRT-Harness | 5 | 5 | 2 | 5 | 5 | high-upside pivot |
| 3 | PREQ certificate audit | 2 | 5 | 4 | 4 | 4 | supporting evaluation study |
| 4 | Harness-C | 2 | 4 | 4 | 3 | 4 | crowded benchmark fallback |
| 5 | ActiveHarness | 1 | 4 | 2 | 3 | 5 | retire or merge into Transport |

## Recommendation

For the current deadline, start with **Prospective Harness Transport**. It must be
narrower than a general transfer benchmark: atomic interventions, prospectively
sealed target cells, and an out-of-sample prediction of effect sign or magnitude.
SEAGym already shows cross-model sign reversals for evolved whole harnesses, so a
descriptive transfer matrix alone is no longer enough.

Keep **MRT-Harness** as the pivot because the full corpus contains no prospective
decision-point-randomized study of harness interventions. Its novelty is stronger,
but instrumentation, carryover, and sample size make it the riskier 18-day bet.

The updated collision audit changes three previous judgments:

- SEA directly implements anytime-valid admission gates for self-evolving harnesses;
  TTHE explicitly identifies prequential scoring as the next evaluation step.
- HARBOR already performs constrained, noisy, multi-fidelity Bayesian optimization
  over harness flags, so generic sample-efficient harness selection is occupied.
- MAS-FIRE, Claw-Eval, RepoMirage, and the agent-reliability literature make a
  general corruption suite too crowded without a new scientific estimand.

The transport kill test is positive only if effects vary materially and a model
trained on source cells predicts a sealed target better than global-effect and
nearest-model baselines. If that fails, do not relabel an ablation table as a
transport paper.

## First 72 Hours

1. Freeze two source models, two task families, and one sealed target slice.
2. Implement verification, context refresh, and retry as atomic interventions.
3. Run 72-96 paired episodes and estimate intervention effects by source cell.
4. Predict target effects from intervention type, task horizon, model tier, and
   observed interface-error rate.
5. Continue only with heterogeneity plus prospective predictive lift.
6. If flat, pivot to a 200-decision-point verification MRT pilot.

## AAAI Reality Check

As of 2026-07-10, the official AAAI-27 schedule gives:

- abstract deadline: 2026-07-21, 11:59 PM UTC-12;
- full paper deadline: 2026-07-28, 11:59 PM UTC-12;
- supplementary material and code: 2026-07-31, 11:59 PM UTC-12.

The main paper has 7 pages of content plus up to 2 pages of references. That makes
scope discipline non-negotiable: one estimand, one method, one central figure,
and one external-validity confirmation.

Official sources:

- [AAAI-27 Main Technical Track Call](https://aaai.org/conference/aaai/aaai-27/main-technical-track-call/)
- [AAAI-27 Submission Instructions](https://aaai.org/conference/aaai/aaai-27/submission-instructions/)

## What Would Fail the Main-Track Bar

- one model or one task family only;
- no uncertainty over stochastic outcomes;
- no distinction from the nearest 2026 paper;
- a new benchmark with no scientific finding;
- an algorithm with no fresh-task or fresh-model evaluation;
- a result selected after inspecting every possible split and metric;
- treating heavy AI assistance as a substitute for experimental ownership.
