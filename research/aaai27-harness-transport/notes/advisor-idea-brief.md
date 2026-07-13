# 历史版本：Advisor Brief: Is FORECAST-H Worth Pursuing?

> **状态：已被替代。** 当前中文导师简报见 [TRACE-H 导师一页简报](advisor-brief-trace-h-zh.md)，最新全文证据边界见[碰撞复核](../foundations/notes/fulltext-collision-reassessment-zh.md)。下方 FORECAST-H 评分、实验量和概率均为历史记录，不得用于当前汇报。

## One-Sentence Idea

When a base model is replaced, predict which atomic harness interventions should be reused, retuned, or rejected before observing target intervention outcomes.

## Scorecard

| Dimension | Current assessment |
|---|---:|
| Neatness | 7.2/10 now; 9/10 after scope compression |
| Excitement | 7.8/10; requires a sealed sign reversal or decision-regret win |
| Problem evidence | 9/10 |
| Evidence for our solution | 2.5/10; no experiment yet |
| Novelty | 7/10, narrow and time-sensitive |
| Technical depth | 6.3/10 now; 8.2/10 with calibrated effect prediction |
| Workload feasibility | 3.8/10 under the current deadline |
| AAAI fit | 7.5/10 |

## Why It Is Promising

- Strong recent evidence shows both broad positive transfer and catastrophic negative transfer.
- Existing predictors estimate absolute performance or select policies, not sealed target intervention effects.
- The outcome is operational and easy to explain: reuse, retune, or reject after a model upgrade.
- Prospective sealing and complete traces provide unusually strong empirical credibility.

## Why It Is Risky

- The surviving novelty requires several ingredients simultaneously; generic harness transport is occupied.
- A convincing predictor needs more model diversity than a normal three-week agent experiment.
- AAAI-27 abstracts are due July 21 and papers July 28.
- A null predictor leaves only a descriptive result already covered by recent work.

## Acceptance Forecast

AAAI-26's official overall main-track rate was approximately 17.5%. Subjective conditional estimates for this project:

- pilot-only descriptive paper: 5-12%;
- one sealed target with modest lift: 12-22%;
- clear sealed-target lift, calibrated abstention, mechanism evidence: 25-40%;
- strong independent replication: 35-50%, probably infeasible by this deadline.

## Recommendation

Run the 96-episode, 72-hour kill test. Continue only if atomic effects activate, differ across models after budget control, and show an interpretable signal. Preserve four model endpoints and sacrifice domain breadth first. Do not submit a transport heatmap as the main contribution.

Full rationale: [DR-0001](../decisions/0001-forecast-h-idea-quality-audit.md).
