# Frontier Map: Model x Harness x Task

## Executive conclusion

The first-order question, “does the harness matter?”, is already answered.
Multiple studies show large harness effects, model-harness interactions, and
ranking reversals. A paper that only reports a model-by-harness grid is now too
late.

**Full-corpus update:** SEA and HARBOR close much of the original adaptive-validity
and efficient-design space, while SEAGym occupies descriptive cross-model harness
transfer. The primary open claim is now prospective prediction of atomic
intervention transfer. Decision-point micro-randomization remains the clearest
high-novelty alternative.

The open frontier has moved to five second-order questions:

1. **Transport:** when does a harness effect survive a model or task change?
2. **Sequential causality:** at which trajectory state does an intervention help?
3. **Adaptive validity:** can a self-evolving harness be evaluated after repeatedly
   querying the same tasks or proxies?
4. **Stability:** how sensitive is a model-harness pair to small controller-input
   corruptions?
5. **Efficient design:** how can a good component set be identified without an
   exponential evaluation campaign?

## 1. System-level measurement is crowded

The core empirical finding is no longer novel:

- [Stop Comparing LLM Agents](https://arxiv.org/abs/2605.23950) formalizes a
  model-harness variance decomposition and reports ranking reversals.
- [Harness-Bench](https://arxiv.org/abs/2605.27922) evaluates complete harness
  configurations over 5,194 trajectories, explicitly framing its results as
  diagnostic rather than component-causal.
- [General Agent Evaluation](https://arxiv.org/abs/2602.22953) runs a factorial
  study over architectures, backbones, and six benchmarks.
- [Agent Psychometrics](https://arxiv.org/abs/2604.00594) decomposes agent ability
  into LLM and scaffold components with an IRT model.
- [HAL](https://arxiv.org/abs/2510.11977) supplies standardized evaluation and a
  large model-scaffold-benchmark trace collection.

Open space: causal effects of atomic mechanisms, transport across unseen models
and tasks, and valid population-level claims when the sampled set of harnesses
has no principled distribution.

## 2. Static component attribution is also occupied

- [Cross-Component Interference](https://arxiv.org/abs/2605.05716) evaluates all
  `2^5` component subsets, computes Shapley values, and shows greedy selection can
  fail.
- [AgentSpec](https://arxiv.org/abs/2606.14674) studies controlled composition and
  interaction effects in embodied agents.
- [Inside the Scaffold](https://arxiv.org/abs/2604.03515) supplies a source-code
  taxonomy of coding-agent architectures.

Open space: the CCI paper explicitly reports no confirmed mechanism. Existing
studies treat a component as present for the whole episode; they do not estimate
time-varying causal effects of verification, retry, memory injection, or context
refresh at individual decision points.

## 3. Harness optimization has become a dense subfield

The slot “automatically improve the harness” is full:

- [Meta-Harness](https://arxiv.org/abs/2603.28052) searches over harness code.
- [AHE](https://arxiv.org/abs/2604.25850) evolves coding-agent harnesses from
  trace evidence.
- [Self-Harness](https://arxiv.org/abs/2606.09498) lets a model improve its own
  model-specific harness with regression gates.
- [Life-Harness](https://arxiv.org/abs/2605.22166) converts deterministic
  interface failures into reusable interventions and reports transfer to 17
  other models.
- [Adaptive Auto-Harness](https://arxiv.org/abs/2606.01770) handles open-ended
  streams and distribution shifts.
- [TTHE](https://arxiv.org/abs/2607.08124) evolves the harness during evaluation
  from unlabeled test traces.
- [Offline Harness Control](https://arxiv.org/abs/2607.05458) formulates a Harness
  MDP and learns a controller with offline RL.

Open space: SEA now implements anytime-valid harness-edit gates, so proposing a
gate is occupied. The narrower unresolved question is whether false-promotion
control and forward behavior remain valid under endogenous proposals,
policy-induced distribution shift, and repeated adaptive task reuse. TTHE still
leaves prequential scoring unimplemented, but prequential scoring alone is now a
thin contribution.

## 4. Transfer results conflict rather than converge

The literature does not support a universal “harnesses transfer” or “harnesses
are model-specific” claim:

| Evidence | Result | Boundary left unresolved |
|---|---|---|
| Stop Comparing | harness variance dominates on one controlled SWE-bench slice | ratio is explicitly not claimed universal |
| General Agent Evaluation | backbone dominates overall across six benchmarks | architecture effects still reach 12 points |
| AHE | frozen harness transfers across benchmark and model families | step budget and timeout were fitted to one model; portability is confounded with operating-point coupling |
| Life-Harness | Qwen3-4B-derived interventions help 17 other models | limited to deterministic, rule-governed environments |
| Self-Harness | useful edits are model-specific | no cross-model swap matrix of learned edits |
| AdaCoM | transfer is best between similarly capable agents | only context management is studied |
| Harness Updating Is Not Harness Benefit | benefit is non-monotone in capability | does not provide a general transport estimator |

This contradiction is the strongest empirical motivation for a transportability
paper.

## 5. Failure diagnosis is crowded; prospective intervention is not

- AgentRx, StepFinder, FALAT, and HarnessFix diagnose failed trajectories.
- [AgenTracer](https://arxiv.org/abs/2509.03312) uses programmed faults and
  counterfactual replay.
- [Causal Agent Replay](https://arxiv.org/abs/2606.08275) intervenes on a past
  step and re-executes the future, with confidence intervals and Shapley credit.

Open space: prospective randomized interventions. Retrospective replay asks
which past step caused failure. A micro-randomized harness study instead asks
whether an intervention at the current state improves a proximal future outcome,
and how that effect changes with context load, uncertainty, anomaly type, and
remaining budget.

## 6. Context, memory, tools, and cost are active but component-specific

Strong recent work already covers:

- context management and transfer: AdaCoM, CWL, Plans Don't Persist;
- safety loss through compaction: Governance Decay;
- proactive memory intervention and causal memory selection;
- minimal tool filtering and tool-menu benchmarks;
- matched-budget feedback quality through Effective Feedback Compute;
- equal-token comparisons of single-agent and multi-agent systems.

A new paper should use these as intervention primitives, not pitch one more
memory or tool-filtering module as the main novelty.

## 7. Stability testing is proposed but not established

Stop Comparing records a perturbation stress-test protocol in its appendix but
does not run it. The proposed perturbations include context ordering and tool
feedback changes. Tool-robustness and prompt-injection papers study narrower
surfaces, but there is no accepted `Harness-C`-style common-corruption suite that
reports stability envelopes across context, tools, state, scheduling, and
verification.

This is open enough for a benchmark-method paper, but the nearest-prior-work
distance is smaller than for prequential validity or micro-randomized trials.

## 8. What “main-conference contribution” means here

A viable AAAI paper needs more than a new table. It should contain at least
three of the following four elements:

1. a new formal object or estimand;
2. a method or protocol that another group can reuse;
3. controlled multi-model evidence with uncertainty;
4. a substantive finding that changes evaluation or design practice.

AAAI-27 explicitly allows theoretical, methodological, algorithmic, empirical,
integrative, and critical contributions, but evaluates significance, novelty,
soundness, relevance, clarity, and reproducibility. The 7-page main-content
limit rewards one sharp claim rather than a sprawling agent system.
