# Detailed Research Proposals

These proposals are the expanded, collision-aware versions of the five idea
notes. They are written as executable research plans rather than concept briefs.

## Proposal Index

1. [PREQ-Audit: Auditing Anytime-Valid Certificates](01-preq-audit-proposal.md)
2. [Prospective Harness Transport](02-prospective-harness-transport-proposal.md)
3. [MRT-Harness: Micro-Randomized Interventions](03-mrt-harness-proposal.md)
4. [Harness-Curves: Stability Envelopes](04-harness-stability-envelope-proposal.md)
5. [ActiveHarness-T: Transfer-Aware Warm Starts](05-transfer-aware-active-harness-proposal.md)

## Current Decision

| Proposal | Surviving novelty | 18-day feasibility | Expected main experiment | Current role |
|---|---:|---:|---:|---|
| Prospective Harness Transport | medium-high | medium | up to 1,200 paired episodes | **AAAI-27 primary** |
| MRT-Harness | very high | low-medium | 1,500-2,000 decision points plus policy test | **high-upside pivot** |
| PREQ-Audit | medium-low | medium-high | synthetic streams plus 1,000-1,500 agent episodes | supporting paper |
| Harness-Curves | medium-low | medium | about 1,920 episodes plus repeats | crowded fallback |
| ActiveHarness-T | high if transfer works | low | local response table plus target optimization | longer-term / merge |

## Why the Proposals Differ

### PREQ-Audit

The unit of analysis is a **promotion decision**. The paper asks whether a
statistical certificate controls harmful harness upgrades under an endogenous
evolution loop.

### Prospective Harness Transport

The unit of analysis is an **atomic intervention effect in a model-task cell**.
The paper asks whether target effect sign and magnitude can be predicted before
target outcomes are observed.

### MRT-Harness

The unit of analysis is a **randomized decision point inside a trajectory**. The
paper asks when an intervention has positive or negative proximal causal effect.

### Harness-Curves

The unit of analysis is a **model-harness degradation curve over corruption
severity**. The paper asks where stability collapses and rankings reverse.

### ActiveHarness-T

The unit of analysis is an **optimization trajectory after a model upgrade**. The
paper asks how much source response-surface evidence should be reused under a
target rollout and safety budget.

## Shared Infrastructure

All five proposals can reuse:

- deterministic workspace and stateful tool tasks;
- a versioned baseline harness;
- executable verification, retry, and context interventions;
- isolated task workspaces;
- paired seeds and complete trajectory logging;
- token, step, wall-clock, and success accounting;
- model-interface probes;
- script-generated analysis tables;
- sealed task manifests and commit hashes.

The shared event schema should include:

```text
run_id
episode_id
task_id
model_id
harness_version
intervention_id
decision_point
pre_state_hash
post_state_hash
tool_action
tool_result_class
availability
randomization_probability
treatment
verification_result
steps_used
tokens_used
wall_clock_ms
task_success
failure_class
```

Each proposal uses a subset of these fields, but one runner should preserve all of
them.

## Seventy-Two-Hour Branching Rule

Start with the shared atomic-intervention runner.

1. Run the 72-96 episode Transport pilot.
2. If intervention effects vary across source models and a pre-treatment operating
   feature explains the difference, continue Transport.
3. If transport effects are flat, instrument verification decision points and run
   the 200-point MRT pilot.
4. If MRT availability or proximal outcomes fail, run the two-family Harness-
   Curves pilot using the same tool/context surfaces.
5. PREQ-Audit proceeds only if a reproducible evolver exists and the gate-to-
   forward optimism gap appears quickly.
6. ActiveHarness-T should not start during this sprint unless a complete response
   table already exists.

## Main-Conference Gates

| Proposal | Evidence required to justify submission |
|---|---|
| PREQ-Audit | A nontrivial gap between local certificate validity and harmful-promotion control, plus a useful confirmation rule |
| Transport | Sealed-target prediction better than constant and nearest-model baselines |
| MRT-Harness | State-conditional excursion effects plus a prospectively better matched-cost policy |
| Harness-Curves | Mild-severity degradation differences, simultaneous rank reversal, and a mechanism or predictor |
| ActiveHarness-T | Prospective target sample savings and successful negative-transfer fallback |

No proposal should be submitted if its result is only a larger table, another
module, or a post hoc narrative around noisy runs.

## Recommended Order for a Four-Year PhD

1. **Atomic transport:** establish a stable scientific object and reusable runner.
2. **Sequential causal interventions:** identify when components help inside
   trajectories.
3. **Transfer-aware optimization:** use the first two papers to warm-start safe
   harness search after model upgrades.
4. **Reliability envelopes:** extend the intervention library into deployment
   stress and stability certification.
5. **Adaptive validity:** generalize evaluation guarantees to co-evolving models,
   harnesses, tasks, and judges.

This order is not a commitment to one narrow topic. It forms a coherent program:
**measurement, causal identification, transport, optimization, and reliability of
the execution layer around AI agents.**
