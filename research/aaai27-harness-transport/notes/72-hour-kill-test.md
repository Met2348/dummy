# 历史版本：FORECAST-H 72-Hour Kill Test

> **状态：已被替代。** 当前执行版本为 [TRACE-H 72 小时机制杀伤实验](trace-h-72-hour-kill-test-zh.md)。本文件保留用于追踪旧版 FORECAST-H 的实验范围，不再作为执行依据。

## Purpose

This pilot does not validate the full predictor. It answers whether there is enough estimable, model-dependent signal to justify spending the remaining sprint on FORECAST-H.

## Frozen Mini-Design

| Dimension | Choice |
|---|---|
| Domain | One deterministic, stateful tool-use environment |
| Tasks | 12 tasks selected before any intervention run |
| Models | Two different families: one source and one provisional target |
| Conditions | Baseline plus three atomic interventions |
| Repeats | One temperature-zero run; repeat only discordant or infrastructure-failed pairs |
| Episodes | `2 x 12 x 4 = 96` primary episodes |

### Intervention A: Tool-Schema Normalization

Canonicalize argument order, required-field errors, and machine-readable error codes. Do not add task-specific hints or new tools.

### Intervention B: Deterministic Finalization Gate

Before accepting the final answer/action, run an existing deterministic state checker and return one bounded error message. No model judge.

### Intervention C: Prose Workflow Guidance

Add a short predeclared workflow instruction without executable enforcement. This is the model-coupled comparator most likely to expose negative transfer.

## Atomicity Checks Before Running

- identical model identifier, decoding, task state, tool permissions, and grader;
- one-change diff for each condition;
- exact trigger and maximum added budget recorded;
- intervention activation logged per task;
- no target-specific examples or post-hoc prompt edits;
- target task order and paired seeds frozen in a manifest hash.

## Measurements

- paired task success delta;
- activation opportunity and activation count;
- malformed tool calls;
- loop/repeated-action rate;
- valid finalization rate;
- steps, tokens, and wall time;
- success conversion conditional on activation;
- regressions where baseline succeeds and intervention fails.

## Timeline

### Hours 0-12

Freeze task/model manifests, implement three interventions, write atomicity tests, and run four smoke tasks that are excluded from analysis.

### Hours 12-36

Run the 96 paired episodes. Preserve all failures. Infrastructure failures are rerun with the same condition and separately labeled.

### Hours 36-52

Compute paired deltas, activation-normalized effects, cost-matched summaries, and a two-model effect-difference table. Do not fit a flexible predictor.

### Hours 52-72

Audit 12-20 decisive trajectories, classify mechanisms, estimate full-run cost, and issue a signed go/pivot/stop decision.

## Go Criteria

Continue to the full study only if all are true:

1. at least two interventions activate on at least four tasks per model;
2. at least one intervention has a meaningful cross-model difference, defined before running as a sign disagreement or at least three discordant paired outcomes between models;
3. at least one clean-run feature or activation statistic has a plausible mechanism link to that difference;
4. raw and cost-matched conclusions do not contradict each other without an explainable budget mechanism;
5. the projected full matrix can finish by the experiment cutoff.

## Pivot Criteria

- If effects exist but are uniform across models, pivot to **opportunity-normalized harness effects** rather than transport prediction.
- If only prose guidance varies, narrow the paper to **behavioral calibration transfer** and use Probe-and-Refine as the direct baseline.
- If only task domains vary, pivot to task-shift transport and drop the model-upgrade claim.
- If interventions rarely activate, redesign tasks before collecting more episodes; do not inflate the matrix with dormant conditions.

## Stop Criteria

- no estimable paired effect after activation filtering;
- atomicity requires changing information, control, and budget together;
- all observed differences are infrastructure noise or task leakage;
- target access or task runtime makes a sealed audit infeasible;
- the only conclusion is a replication of Life-Harness, CCI, or SEAGym.

## Artifact Checklist

- `experiments/manifests/source_tasks.json`
- `experiments/manifests/target_tasks.json`
- `experiments/manifests/models.json`
- `experiments/interventions/*.json`
- `experiments/predictions/pilot-decision.md`
- raw JSONL trajectories and activation logs
- one command that regenerates the paired-effect table
