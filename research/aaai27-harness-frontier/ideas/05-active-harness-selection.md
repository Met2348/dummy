# Idea 5: ActiveHarness

## Sample-Efficient Interaction-Aware Harness Selection

**Verdict:** retire in its current form. HARBOR directly occupies constrained,
cost-aware, multi-fidelity Bayesian optimization over harness configurations.

**One-sentence claim:** because harness components interact and violate
submodularity, greedy addition is unreliable; interaction-aware active design can
identify a near-optimal, cost-aware configuration with far fewer agent rollouts.

## Why this is open

Cross-Component Interference shows that the all-in harness is suboptimal, greedy
selection can fail, and exhaustive evaluation grows as `2^p`. Meta-Harness, AHE,
Self-Harness, and related systems search a much richer code space, but they do not
offer a fixed-confidence best-configuration guarantee under stochastic expensive
rollouts.

The open slot is not “search harnesses with an LLM.” It is “allocate evaluation
budget statistically and know when the selected configuration is trustworthy.”

The citation-chasing audit found the decisive collision: HARBOR formalizes noisy
mixed-variable harness optimization, uses a block-additive sparse surrogate,
cost-aware multi-fidelity acquisition, TuRBO trust regions, and a posterior safety
constraint. It also reports an end-to-end production-harness run. A fixed-confidence
best-arm guarantee alone is unlikely to clear the novelty bar. The plausible pivot
is transfer-aware warm starting across model upgrades, which should be folded into
Harness Transport rather than submitted as this standalone idea.

## Formal object

Represent a harness by a binary or categorical component vector `x`. Each
evaluation returns stochastic quality `Y(x)` and cost `C(x)`. Under rollout budget
`N`, identify a configuration on the quality-cost Pareto frontier and bound simple
regret relative to the best configuration in the declared space.

The method can combine:

- a hierarchical sparse ANOVA surrogate that admits main and selected interaction
  terms;
- uncertainty-aware acquisition over unexplored configurations;
- successive elimination or best-arm identification for close finalists;
- paired tasks and common random seeds where available;
- explicit cost constraints rather than quality alone.

## Contributions needed for a main paper

1. An interaction-aware active-design algorithm for stochastic harness evaluation.
2. A finite-sample selection or simple-regret statement under declared
   assumptions.
3. Offline replay on a full-factorial dataset to measure sample efficiency.
4. Prospective validation on a larger 8-12 component space where exhaustive search
   is impractical.
5. Comparison with random search, greedy addition, standard Bayesian optimization,
   and an LLM-proposer baseline.

## Minimal experiment

Stage A, cheap validation:

- use the full `2^5` CCI response table as an offline oracle;
- repeatedly subsample evaluations and compare probability of selecting the true
  best configuration.

Stage B, prospective:

- 8 binary components;
- 2 models and 2 task types;
- evaluate at most 20-25% of the 256 configurations;
- fully reevaluate the top 5 finalists on fresh tasks.

## Killer figure

Probability of selecting a configuration within 2 points of optimal versus
rollout budget. ActiveHarness reaches 90% selection probability with one quarter
of the evaluations required by random or greedy search, while maintaining an
honest confidence bound.

## Nearest collisions and required distinction

- Cross-Component Interference: establishes the combinatorial problem but does
  not solve sample-efficient selection.
- HARBOR: direct collision on constrained noisy Bayesian harness optimization;
  distinguish only through transfer-aware warm starts, formal selection guarantees,
  or a genuinely different interaction structure backed by prospective evidence.
- Meta-Harness/AHE/Self-Harness: proposal systems without fixed-confidence
  selection guarantees.
- Bayesian-Agent: posterior-guided skill evolution; this idea must focus on
  experimental allocation and certified selection, not skill lifecycle prose.
- classical best-arm and hyperparameter optimization: methodological ancestors;
  the new work must exploit paired tasks, component hierarchy, interaction
  sparsity, and stochastic agent costs.

## 18-day execution path

Days 1-4: obtain a full-factorial replay dataset and implement four baselines.

Days 5-7: prototype the surrogate and verify selection calibration offline.

Days 8-12: prospective 8-component study.

Days 13-15: fresh-task finalist evaluation and regret analysis.

Days 16-18: paper and proof cleanup.

## Kill criterion

Stop if the method does not beat random search clearly on the offline oracle by
day 5, or if a defensible uncertainty statement cannot be written by day 7.

## Main risk

This can become generic Bayesian optimization wearing harness vocabulary. The
paper needs a harness-specific statistical structure and a prospective result;
otherwise it will not justify the domain framing.
