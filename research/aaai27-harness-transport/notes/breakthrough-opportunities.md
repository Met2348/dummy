# 历史工作稿：FORECAST-H Breakthrough Opportunities

> **状态：已被替代。** 当前方法中心见 [TRACE-H 正式 Proposal](../proposals/trace-h-formal-proposal-zh.md) 和 [Idea 大调整记录](idea-iteration-and-selection-zh.md)。本文保留用于追踪旧版候选，不作为当前 novelty claim。

## Recommended Reframing

Rename the working contribution from generic "Prospective Harness Transport" to:

**FORECAST-H: Target-Conditioned Forecasting of Atomic Harness Effects Under Model Replacement**

The paper question becomes operational:

> A new model version arrives. For each existing harness component, should the team reuse it unchanged, retune it, or reject it before paying for a full target sweep?

## 1. Risk-Controlled Reuse / Retune / Reject

**Rank: 1. Main contribution.**

Predict both effect magnitude and `P(Delta_target < -epsilon)`. Convert the forecast into three actions:

- `reuse` when the lower uncertainty bound exceeds zero;
- `reject` when negative-transfer risk exceeds a frozen threshold;
- `retune` otherwise.

Primary metric: decision regret versus an oracle that sees the target effect table. Secondary metrics: sign balanced accuracy, effect MAE, Brier score, and interval coverage.

Why this is stronger: existing work reports positive/negative transfer or chooses a strategy. It does not make and audit a risk-bearing model-upgrade decision.

## 2. Baseline-Only Target Fingerprints

**Rank: 2. Enabling technical idea.**

Measure a cheap target fingerprint without exposing any intervention outcomes:

- malformed tool-call rate;
- valid finalization rate;
- loop/repetition rate;
- recovery after native tool errors;
- clean success and trajectory length;
- budget utilization;
- response to short synthetic interface probes.

Fit effects from source cells using intervention descriptors crossed with this fingerprint. Compare zero target anchors against 4- and 8-task calibration variants. The zero-anchor setting is the strongest claim; the anchor curve makes the result useful even if strict zero-shot prediction is weak.

## 3. Opportunity-Normalized Effects

**Rank: 3. Mechanistic breakthrough.**

A harness component cannot help if its trigger never occurs. Decompose the observed effect into:

`Delta = P(activation) x E[local recovery gain | activation] - resource/interference cost`.

Report:

- activation opportunity rate;
- success conversion conditional on activation;
- cost paid on non-activated runs;
- downstream interference after activation.

This directly explains why deterministic checks may transfer broadly while prose guidance or retry loops become model-specific. It also guards against false nulls caused by dormant interventions.

## 4. Mechanism Contracts Instead of Text Embeddings

**Rank: 4. Representation contribution.**

Each intervention gets a machine-readable contract:

- information added or removed;
- control flow changed;
- deterministic or model-mediated;
- stateful or stateless;
- trigger and stop condition;
- maximum extra steps/tokens;
- dependence on prose compliance;
- failure surface targeted.

Use these fields as the primary intervention representation. Text embeddings can be an ablation. This is more auditable and less likely to learn paper-specific wording than embedding a prompt or code diff.

## 5. Model-Replacement Challenge Protocol

**Rank: 5. Artifact contribution.**

Release a sealed-cell protocol with:

- source, validation, target, and final-audit manifests;
- intervention contracts and one-change tests;
- paired seeds and target prediction files;
- raw and cost-matched estimands;
- complete failures and no-op activations;
- a public script that verifies target outcomes were timestamped after predictions.

This can survive even if the best statistical predictor is simple. A benchmark alone is not enough for the intended AAAI paper, but it materially strengthens reproducibility.

## The Smallest Main-Conference Story

The minimum coherent claim is:

1. atomic executable interventions exhibit model-dependent effects under matched budgets;
2. baseline-only target fingerprints explain part of that heterogeneity;
3. a frozen predictor improves reuse/retune/reject regret on a sealed new model over source-copy, global-mean, and capability-only baselines;
4. opportunity normalization identifies at least one mechanism behind success and one behind negative transfer.

Do not attempt a broad harness optimizer, a new general agent, and a transport predictor in the same three-week sprint.

## Recommended Scope Cut

- Primary domain: deterministic stateful tool use.
- Secondary audit: a small local workspace/tool subset only after the primary matrix works.
- Primary interventions: schema normalization, deterministic finalization verification, bounded retry/rollback, and prose guidance as a model-coupled comparator.
- Primary target: one sealed model family; a fourth model is an optional replication, not a dependency.
- Primary estimator: regularized hierarchical regression with bootstrap intervals; avoid complex deep meta-learning.
