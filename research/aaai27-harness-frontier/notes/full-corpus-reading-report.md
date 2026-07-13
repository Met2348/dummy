# Full-Corpus Reading Report

Cutoff: 2026-07-10 (Asia/Shanghai)

## Executive Outcome

The full-corpus audit changes the sprint decision. PREQ-Harness is no longer the
best standalone bet, and ActiveHarness in its original form should be retired.
The strongest remaining scientific slot is **prospective transport of atomic
harness interventions**, with **micro-randomized harness interventions** as the
higher-novelty, higher-risk pivot.

This is not a guarantee of AAAI acceptance. It is a collision-aware decision made
after downloading, indexing, and evidence-scanning the complete local corpus.

## Corpus Integrity

- Final deduplicated manifest: **1,150 records**.
- Local source documents: **1,149 validated PDFs** plus **1 archived full-text HTML**.
- PDF volume: **3,726,264,148 bytes** (about 3.47 GiB).
- Extracted text: **1,150 files**, **97,535,088 characters**.
- Per-paper evidence cards: **1,150**.
- Search database: **312,860,672-byte SQLite FTS5 index**.
- Close-reading set: **76 core papers + 32 direct competitors = 108 papers**.
- Integrity check: zero invalid PDF headers, missing texts, missing cards, or orphan files.

The machine-readable check is
[`final_corpus_status.json`](../metadata/final_corpus_status.json).

## Withdrawal Handling

Two records are withdrawn and must not be treated as validated current evidence:

1. `2602.17547` KLong was withdrawn because significant data errors affect result
   validity. arXiv removed the PDF; the corpus retains an archived HTML full text
   only for historical collision analysis.
2. `2606.14066` FastContext was withdrawn over product-IP issues. The corpus retains
   the last available pre-withdrawal arXiv version, marked as historical evidence.

See [`withdrawn_records.json`](../metadata/withdrawn_records.json).

## What "Read" Means Here

The work used three evidence layers:

1. **All 1,150 papers:** complete local text was parsed and searched; conventional
   method, experiment, result, limitation, and conclusion sections were mapped;
   idea-specific phrases and traceable evidence sentences were extracted.
2. **108 core/direct papers:** contribution, result, limitation, and collision
   evidence was consolidated into a single close-reading dossier.
3. **Decision-changing papers:** SEA, TTHE, SEAGym, HARBOR, MAS-FIRE, Claw-Eval,
   RepoMirage, Life-Harness, HarnessFix, Bayesian-Agent, and the statistical
   foundations were inspected directly in full text around formulas, protocols,
   experiments, limitations, and future-work claims.

This is a reproducible, AI-assisted full-text review. It is not honest to call the
613 peripheral search hits a sentence-by-sentence human cover-to-cover reading.
They received full-text evidence triage; the 108-paper set received the deeper
claim-level audit. Every item remains locally available for manual escalation.

## Decision-Changing Collisions

### PREQ-Harness

`2607.00871` SEA directly admits self-evolving harness edits through anytime-valid
confidence-sequence and e-process gates. `2607.08124` TTHE explicitly distinguishes
its transductive evaluation from prequential scoring. A new paper cannot lead with
"use anytime-valid gates" or "evaluate prequentially" alone.

The surviving slot is narrower: empirically test false-promotion control and
forward behavior when proposals, data, and policy-induced distributions are
endogenous. SEA itself says its composed endogenous-loop guarantees are conjectural
and its expensive evaluation is single-run. This is useful, but no longer the
safest independent 18-day contribution.

### Harness Transport

SEAGym already shows cross-model sign reversals for evolved whole harnesses and
qualitatively links them to backend-specific failure surfaces. Meta-Harness,
Life-Harness, HarnessFix, HarnessBridge, AgentTether, and Probe-and-Refine add
positive and negative transfer evidence.

The remaining main-paper claim must be prospective and predictive:

- isolate atomic interventions;
- define source and target treatment effects;
- seal target model-task cells before analysis;
- predict effect sign or magnitude from intervention and operating-point features;
- beat constant-effect and nearest-model baselines on unseen cells.

Another transfer heatmap without prediction is not enough.

### MRT-Harness

No corpus paper prospectively randomizes harness interventions at repeated agent
decision points and estimates state-conditional causal excursion effects. Existing
work uses episode-level factorials, retrospective causal replay, or learned control
policies. The methodological slot remains open.

The main risk is execution: availability rules, carryover, proximal-outcome design,
and clustered uncertainty must be correct. The first sprint should randomize only
verification and use deterministic tools.

### Harness-C

MAS-FIRE, Claw-Eval, RepoMirage, self-healing orchestrators, and the broader agent
reliability literature already cover fault injection, perturbations, robustness,
and recovery. A generic multi-fault benchmark is crowded.

The idea survives only if it contributes a cross-layer severity law, stability
envelopes with uncertainty, and a scientific finding such as predictable ranking
reversal or a compact subset that preserves stability order.

### ActiveHarness

HARBOR is a direct collision. It formulates automated harness optimization as
constrained noisy Bayesian optimization over a mixed, cost-heterogeneous flag
space, uses a sparse block-additive surrogate, multi-fidelity cost-aware
acquisition, TuRBO trust regions, and a posterior safety constraint, and reports an
end-to-end production-harness run.

HARBOR is limited to a conditional agent-task-model setting, claims no transfer,
and shows noisy cheap-fidelity behavior. Those gaps point toward transfer-aware
warm starts, but that is better treated as part of Harness Transport than as the
original ActiveHarness paper.

## Updated Sprint Recommendation

**Primary:** Prospective Harness Transport.

The 72-hour go/no-go test should use two source models, two task families, three
atomic interventions, and one sealed target slice. Continue only if intervention
effects vary and a small moderator model predicts the target better than simple
baselines.

**Pivot:** MRT-Harness with one binary verification intervention.

Continue only if a 200-decision-point pilot yields at least five usable decision
points per episode, dense deterministic proximal outcomes, balanced randomization,
and visible state-dependent effect heterogeneity.

**Do not submit:** generic anytime-valid harness gating, a descriptive transfer
matrix, a generic corruption benchmark, or Bayesian harness configuration search.

## Navigation

- [Five detailed research proposals](../proposals/README.md)
- [All 1,150 papers](all-library-index.md)
- [All 1,150 reading cards](../reading-cards/)
- [108-paper core/direct dossier](direct-competitor-dossier.md)
- [Five-idea full-corpus audit](full-corpus-idea-audit.md)
- [Citation-gap audit](citation-gaps.md)
- [Updated idea ranking](../ideas/idea-ranking.md)
- [Local FTS5 database](../metadata/corpus.sqlite)

Search example:

```powershell
python scripts/search_corpus.py prequential --database metadata/corpus.sqlite --limit 20
```
