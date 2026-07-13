# AAAI-27 Harness Frontier Intelligence Pack

This directory is intentionally isolated from the existing `harness-lit/` and
`docs/literature/llm-auto-research/` collections. It records a reproducible
literature search and idea-selection process for model-harness-task research.

## Directory map

- `scripts/`: reproducible metadata search, ranking, and download tools.
- `metadata/`: raw and curated machine-readable records.
- `papers/`: locally cached PDFs for the highest-priority papers.
- `papers-all/`: every downloadable PDF in the final corpus.
- `texts-all/`: extracted full text for every corpus record.
- `reading-cards/`: one traceable evidence card per paper.
- `withdrawn/`: archived records and pre-withdrawal material, kept separate.
- `notes/`: search protocol, frontier map, evidence matrix, and synthesis.
- `ideas/`: five pressure-tested paper ideas and a comparison matrix.
- `proposals/`: five standalone, experiment-ready research proposals.

## Scope

The target is not another general-purpose agent. The scope is the execution
layer around an LLM agent and the scientific questions created by its
interaction with models, tasks, budgets, and trajectories:

1. harness/scaffold/system-level evaluation;
2. model x harness x task interaction and transport;
3. component-level attribution and sequential interventions;
4. robustness, reliability, cost, and ranking stability;
5. harness optimization, adaptation, and benchmark overfitting.

The search cutoff is 2026-07-10 (Asia/Shanghai).

## Current snapshot

- 1,016 unique records in the broad candidate pool;
- 121 records in the narrow statistical supplement;
- 6 citation-chased additions that closed important keyword-search gaps;
- 1,150 deduplicated records: 1,149 validated PDFs and one archived HTML full text;
- 1,150 extracted texts, 1,150 evidence cards, and a 312 MB SQLite FTS5 index;
- 76 core papers plus 32 additional direct competitors in the close-reading dossier;
- 5 re-audited paper ideas with collision, experiment, and kill criteria.

The authoritative integrity record is
[`metadata/final_corpus_status.json`](metadata/final_corpus_status.json). The
full reading outcome and its limits are documented in
[`notes/full-corpus-reading-report.md`](notes/full-corpus-reading-report.md).

Start with [`START_HERE.md`](START_HERE.md).
