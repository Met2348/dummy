# Idea 3: MRT-Harness

## Micro-Randomized Trials for Sequential Agent-Harness Interventions

**Verdict:** highest surviving methodological novelty; high sprint risk and the
recommended pivot if transport lacks a predictive signal.

**One-sentence claim:** the effect of verification, retry, and context refresh is
state-dependent and time-varying; micro-randomization identifies when each
intervention helps instead of treating a harness component as globally on or off.

## Why this is open

Existing component studies use episode-level configurations. Cross-Component
Interference toggles five components for an entire run; AgentSpec composes modules
at system level. Causal Agent Replay intervenes retrospectively on a failed past
step. Offline Harness Control learns a policy, but policy optimization does not
identify the causal effect of an intervention at a particular decision state.

The missing design is prospective randomization at repeated decision points.
Micro-randomized trials were developed for just-in-time adaptive interventions
and provide causal excursion effects under known randomization probabilities.
The citation-chasing audit adds the stratified MRT design for randomization times
that depend on outcomes of prior treatment; this is especially relevant when an
agent intervention becomes available only after a detected error or state change.

## Formal object

At each available decision point `t`, the harness observes state features `S_t`
and randomizes intervention `A_t` with logged probability `p_t`:

- verify now vs no verification;
- refresh/re-ground context vs continue;
- retry/rollback vs continue from current state.

For horizon `k`, estimate the causal excursion effect on a proximal outcome:

`beta_k(s) = E[Y_(t+k)(A_t=1) - Y_(t+k)(A_t=0) | S_t=s, available]`.

Useful proximal outcomes include anomaly recovery within three steps, transition
to a task-advancing state, tool-error recurrence, constraint retention, and token
cost. Final task success is secondary because it is statistically sparse.

Use weighted and centered least squares with known randomization probabilities;
cluster uncertainty by episode and report sensitivity to carryover windows.

## Contributions needed for a main paper

1. The first micro-randomized protocol for agent-harness interventions.
2. State-conditional causal excursion effects with uncertainty.
3. Evidence that globally helpful components can be harmful in specific states.
4. A policy distilled from identified effects and evaluated on fresh episodes.
5. An open trace format containing availability, randomization probability,
   treatment, proximal outcome, and carryover state.

## Minimal experiment

Start with one intervention, verification, to avoid an underpowered multi-arm
study.

- 2 models;
- 20-30 deterministic multi-step tasks;
- 8-10 repeats or task variants;
- target 1,500-2,000 randomized decision opportunities;
- randomization probability 0.5 when the intervention is available;
- fresh held-out episodes for the derived policy.

After the first signal, add context refresh at high context load as a second
intervention. Do not begin with a three-way factorial at every step.

## Killer figure

A state-effect surface: verification helps strongly after tool errors and at
moderate uncertainty, has near-zero value in clean states, and hurts under very
low remaining budget. A fixed “always verify” policy is dominated by a simple
effect-informed policy at equal cost.

## Nearest collisions and required distinction

- Cross-Component Interference: whole-episode static factorial.
- Causal Agent Replay: retrospective re-execution and blame assignment.
- Offline Harness Control: optimization from logged trajectories without
  prospective randomized identification.
- proactive memory/AdaCoM: learned component policies, not a general causal
  experimental design.

## 18-day execution path

Days 1-3: instrument availability and one binary intervention in the existing
agent loop.

Days 4-5: run a 200-opportunity pilot and verify randomization balance, logging,
and proximal-outcome density.

Days 6-10: full MRT collection and WCLS analysis.

Days 11-13: derive and prospectively test a simple intervention policy.

Days 14-18: robustness, carryover sensitivity, paper, and artifact release.

## Kill criterion

Stop if there are fewer than five usable decision opportunities per episode,
proximal outcomes cannot be graded without an LLM judge, or intervention effects
are swamped by irreproducible environment state.

## Main risk

Sequential interference and carryover can invalidate naive analysis. The first
paper should use mocked or deterministic tools, one primary intervention, and a
predeclared proximal horizon. Scientific restraint is part of the contribution.
