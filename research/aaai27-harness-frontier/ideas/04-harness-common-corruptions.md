# Idea 4: Harness-C

## Common Controller Corruptions and Stability Envelopes for LLM Agents

**Verdict:** crowded after full-corpus audit; viable only with cross-layer severity
laws or a compact stability predictor, not as a generic corruption benchmark.

**One-sentence claim:** agent capability should include a stability envelope over
small harness-input corruptions, because nominally similar model-harness pairs can
have different failure thresholds and ranking reversals.

## Why this is open

Stop Comparing proposes a perturbation stress test in its appendix but does not
run it. Harness-Bench compares native complete configurations rather than
controlled corruption severity. Existing robustness work usually attacks one
surface, such as tool descriptions, prompt injection, or context compaction.

There is no established common-corruption suite spanning the controller surfaces
that repeatedly mediate an agent trajectory.

The open space is narrower than this original statement. MAS-FIRE injects 15
intra- and inter-agent fault types into multi-agent systems; Claw-Eval combines
controlled error injection with completion, safety, and robustness metrics;
RepoMirage applies semantics-preserving repository-context perturbations; and
Towards a Science of AI Agent Reliability already defines robustness profiles.
Harness-C must therefore establish a shared severity response law across harness
layers and model-harness rank reversals, not merely collect more fault types.

## Corruption families

Use common, non-adversarial corruptions with 3-5 severity levels:

1. context: reorder, drop, duplicate, or delay retrieved chunks;
2. tool interface: paraphrase schema, add irrelevant tools, alter error format;
3. state: stale observation, missing checkpoint field, partial workspace view;
4. scheduling: delayed corrective feedback, retry jitter, reduced step budget;
5. verification: false-positive or false-negative check noise;
6. memory: stale item, irrelevant retrieval, delayed reminder.

The suite should preserve task semantics and label which harness layer is
corrupted.

## Formal objects

- **Corruption Error:** success drop at severity `q`.
- **Stability AUC:** area under success-versus-severity curve.
- **Critical Severity:** smallest `q` where success drops by a prespecified amount.
- **Recovery Curve:** probability of returning to a task-advancing state within
  `k` steps after a corrupted observation.
- **Ranking Instability:** count and confidence of model/harness rank reversals.

Fit a mixed-effects degradation model over task, model, harness, corruption, and
severity. The contribution is the response curve, not a single average score.

## Contributions needed for a main paper

1. A versioned corruption specification and deterministic injector.
2. Stability metrics with bootstrap or hierarchical uncertainty.
3. A multi-model, multi-harness study showing distinct failure envelopes.
4. At least one mechanistic trace finding that predicts a threshold, such as
   delayed feedback causing repeated stale actions.
5. A compact evaluation subset that preserves stability rankings.

## Minimal experiment

- 3 models;
- 2 harnesses;
- 4 corruption families;
- 3 nonzero severities plus clean control;
- 20 tasks;
- a fractional design for secondary interactions.

Begin with context order/drop and tool error-format corruptions because they are
cheap and directly motivated by prior work.

## Killer figure

Six degradation curves reveal that the nominal winner has the narrowest stability
margin, while a lower-scoring pair dominates after mild controller corruption.

## Nearest collisions and required distinction

- Stop Comparing: contains an unexecuted future-work protocol; Harness-C must be
  broader, released, and empirically validated.
- Harness-Bench: complete-configuration diagnostics, not severity-controlled
  common corruptions.
- MAS-FIRE: systematic fault injection for multi-agent systems with 15 fault types.
- Claw-Eval: controlled error injection and repeated reliability metrics.
- RepoMirage: semantics-preserving context perturbations for coding agents.
- tool-surface and prompt-injection work: narrow adversarial surfaces rather than
  cross-layer non-adversarial stability.
- Harness-Induced Belief Divergence: measures alternative harness views, not a
  standardized robustness envelope.

## 18-day execution path

Days 1-3: corruption schema, injectors, and semantic-preservation tests.

Days 4-6: two-family pilot and effect-size check.

Days 7-11: full run and mixed-effects analysis.

Days 12-14: compact subset selection and trace mechanism.

Days 15-18: paper and benchmark packaging.

## Kill criterion

Stop if corruption semantics cannot be held constant, clean-versus-mild effects
are below noise across all cells, or the task matrix cannot finish by day 6.

## Main risk

Without a reusable injector, severity model, and cross-layer result, reviewers
may see this as another benchmark. The paper must answer a scientific question
about stability, not only release tasks.
