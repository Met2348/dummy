# Idea 1: PREQ-Harness

## Prequential and Anytime-Valid Evaluation of Self-Evolving Agent Harnesses

**Verdict:** no longer the primary bet after the full-corpus audit; retain only as
an empirical audit of endogenous certificate validity or false-promotion behavior.

**One-sentence claim:** self-evolving harnesses are currently rewarded on data
that also influenced their evolution; a prequential protocol with anytime-valid
uncertainty separates apparent adaptation from forward generalization.

## Why this is open

Automated harness evolution is already crowded, but its evaluation is not
settled. TTHE explicitly says its transductive protocol does not establish
forward generalization and leaves prequential scoring to future work. Self-Harness
uses a held-out split as a promotion gate repeatedly; that is a useful regression
test, but after repeated adaptive queries it is no longer an untouched final test
set. AI Agents That Matter warned about benchmark overfitting in agents, while
adaptive data analysis supplies tools for repeated hypothesis selection.

The novelty window narrowed sharply on 2026-07-01: *Self-Evolving Agents with
Anytime-Valid Certificates* (SEA) admits harness edits through confidence-sequence
and e-process gates. SEA also states that its endogenous-loop compositions remain
unproven and reports only single-run expensive evaluations. A surviving paper must
therefore test whether such certificates actually control false promotions under
endogenous proposals and distribution shift; proposing an anytime-valid gate is
no longer sufficient.

The paper is therefore not “we built a better evolver.” It is “we make claims
about evolvers statistically auditable.”

## Formal object

Let task batches arrive as `B1, ..., BT`. Harness `H_t` may use all traces and
proxy signals through batch `B_t`, but it is scored on `B_(t+1)` before seeing or
adapting to that batch.

Define the paired forward gain over a frozen baseline `H0`:

`G_pre = mean_t [ score(H_t, B_(t+1)) - score(H0, B_(t+1)) ]`.

Report three different quantities that current work often mixes:

- **transductive gain:** adapt and score on the same batch;
- **promotion-gate gain:** select with a repeatedly reused validation split;
- **prequential gain:** score on the next unseen batch before adaptation.

For binary task outcomes, maintain an anytime-valid confidence sequence or
e-process for the paired difference. The claim remains valid even if the team
checks results after every round and stops early, provided the next-batch order
and betting rule are fixed before observing its outcomes.

## Contributions needed for a main paper

1. A prequential protocol and terminology for self-evolving harness evaluation.
2. An anytime-valid paired estimator for forward gain and false-promotion risk.
3. An empirical audit of at least two evolution strategies showing when
   transductive and forward gains diverge.
4. A reproducible stream construction with an untouched final audit block.
5. A practical promotion rule that reduces false upgrades without suppressing
   real improvement.

## Minimal experiment

Use two task surfaces with deterministic grading:

- one inexpensive local tool/workspace suite for many evolution rounds;
- one public agent benchmark slice such as Harness-Bench, `tau`-bench, or
  Terminal-Bench for external validity.

Minimum matrix:

- 2 base models;
- 2 evolution methods: TTHE-lite and Self-Harness-style proposal plus gate;
- 3 protocols: transductive, reused gate, prequential;
- 5-8 chronological or randomized blocks;
- 2 complete stream orderings.

Primary outcomes: task success, forward gain, false-promotion rate, cumulative
regret, token cost, and the width of the confidence sequence.

## Killer figure

A three-line learning curve over evolution rounds:

- transductive score rises;
- reused-gate score rises more slowly;
- next-batch prequential score stalls or falls for some settings.

The second panel shows the anytime-valid interval and which promotions would have
been rejected by the proposed rule.

## Nearest collisions and required distinction

- TTHE: evolves at test time but admits no forward-generalization proof.
- SEA: directly implements anytime-valid harness-edit gates; the remaining slot is
  an audit of their endogenous-loop validity and forward behavior, not another gate.
- Self-Harness: uses regression gating but not an untouched sequential test.
- AHE/Meta-Harness/RHO: optimize harnesses; they are baselines, not the same
  contribution.
- AI Agents That Matter: identifies agent benchmark overfitting broadly; this
  paper must add the sequential estimand, estimator, and evolution-specific
  evidence.
- Reusable Holdout: methodological ancestor; the paper must explain why agent
  traces, candidate harness code, and blockwise nonstationarity require a new
  operational protocol.

## 18-day execution path

Days 1-2: reproduce one small evolver and freeze the stream construction.

Days 3-4: run a 3-round pilot. The go/no-go signal is a visible gap between
transductive and next-block scores or at least a measurable false-promotion rate.

Days 5-9: complete the local factorial and implement confidence sequences.

Days 10-12: run one public-benchmark confirmation and sensitivity analyses over
block order and size.

Days 13-18: paper, figures, robustness, and release.

## Kill criterion

Stop or downgrade if all protocols agree within narrow intervals across both
models and task surfaces, or if no evolution method can be reproduced by day 3.

## Main risk

The novelty window is immediate: TTHE was posted on 2026-07-09 and explicitly
names prequential scoring. The defense is to contribute a statistically valid
method and a convincing audit, not merely run its suggested experiment.
