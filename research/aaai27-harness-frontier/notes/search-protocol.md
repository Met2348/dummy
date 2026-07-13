# Search Protocol and Audit Trail

## Objective

Map the research frontier around model-harness-task interactions and identify
paper ideas that can clear a main-conference novelty bar. The search was not
designed to collect every paper that uses the word `harness`; it was designed to
find the nearest prior work that could invalidate a proposed contribution.

Cutoff: **2026-07-10, Asia/Shanghai**.

## Search layers

### Layer 1: broad arXiv retrieval

The reproducible query script ran 26 searches over 2023-01-01 through
2026-07-10. Query families covered:

- explicit `agent harness`, `model harness`, and `agent scaffold` language;
- agent/system-level evaluation and benchmark methodology;
- trajectory evaluation, long-horizon reliability, and robustness;
- context, memory, tools, verification, and test-time scaling;
- causal, factorial, interaction, variance, ranking, and overfitting terms.

The first 20 broad queries produced **1,016 unique candidate records**. Six
narrow statistical queries were rerun after rate limiting and produced a
supplemental pool of **121 records**. Raw records, abstracts, matched queries,
and relevance scores are preserved in `metadata/`.

### Layer 2: backward and forward collision search

The highest-relevance papers were inspected at abstract and full-text level.
Their references, limitations, and explicit future-work statements were used to
search for:

- locked-harness and factorial evaluation;
- psychometric model/scaffold decomposition;
- component interactions and Shapley attribution;
- counterfactual failure replay;
- automated and test-time harness evolution;
- cross-model transfer and model-specific adaptation;
- budget-matched evaluation and dynamic harness control.

Only primary technical sources (papers and official conference pages) are used
to support technical claims in the synthesis.

### Layer 3: hand screening

Seventy papers were retained as the core library. Selection favored work that
does at least one of the following:

1. directly studies harness/scaffold effects;
2. establishes a benchmark or evaluation protocol used by the field;
3. occupies a method slot that a new idea might accidentally duplicate;
4. supplies statistical machinery needed for a stronger contribution;
5. provides contradictory evidence that exposes an unresolved boundary.

The core contains **70 metadata records and 70 locally cached PDFs**. Two
superseded but useful PDFs remain as supplemental files. All core PDFs were also
converted to text for local full-text search.

### Layer 4: full-corpus download and evidence scan

The broad, narrow, and selected pools were deduplicated and every record was
resolved to a local full-text representation. All 1,149 available PDFs passed a
`%PDF` magic-header check and text extraction; withdrawn KLong was represented by
an archived full-text HTML document. Every text was section-mapped and converted
into an evidence card. The final corpus contains **1,150 records**.

### Layer 5: citation-gap chasing

All extracted texts were scanned for arXiv identifiers absent from the manifest.
This produced 9,467 raw reference gaps, most of which were general models,
benchmarks, or unrelated methods. Idea-specific citation contexts and core-paper
citations were screened, then six decision-relevant papers were added: HARBOR,
MAS-FIRE, Claw-Eval, safe anytime-valid inference, performative prediction, and
the stratified micro-randomized trial. This layer exposed the HARBOR collision
that keyword search missed.

## Inclusion and exclusion rules

Included:

- controlled empirical work with disclosed model, harness, task, and budget;
- benchmark/evaluation work with reusable tasks, traces, or protocols;
- methods with a clear intervention, estimator, algorithm, or formal object;
- recent preprints when they are the closest collision to the proposed idea;
- older foundations when they supply indispensable methodology.

Excluded from the core:

- domain applications that merely call their workflow a harness;
- architecture papers with no controlled harness comparison;
- surveys that add no useful taxonomy or gap evidence;
- duplicate versions and papers with only incidental keyword matches;
- claims supported only by blogs, product pages, or secondary summaries.

## Reproducibility

Run from the repository root:

```powershell
python research\aaai27-harness-frontier\scripts\search_arxiv.py `
  --output-dir research\aaai27-harness-frontier\metadata `
  --max-results 80

python research\aaai27-harness-frontier\scripts\build_selected_manifest.py `
  --selection research\aaai27-harness-frontier\metadata\selection_ids.tsv `
  --output research\aaai27-harness-frontier\metadata\selected_papers.json

python research\aaai27-harness-frontier\scripts\download_selected.py `
  --manifest research\aaai27-harness-frontier\metadata\selected_papers.json `
  --papers-dir research\aaai27-harness-frontier\papers

python research\aaai27-harness-frontier\scripts\build_corpus_index.py `
  --manifest research\aaai27-harness-frontier\metadata\all_candidates.jsonl `
  --texts-dir research\aaai27-harness-frontier\texts-all `
  --output research\aaai27-harness-frontier\metadata\corpus.sqlite

python research\aaai27-harness-frontier\scripts\verify_corpus.py `
  --manifest research\aaai27-harness-frontier\metadata\all_candidates.jsonl `
  --papers-dir research\aaai27-harness-frontier\papers-all `
  --texts-dir research\aaai27-harness-frontier\texts-all `
  --cards-dir research\aaai27-harness-frontier\reading-cards `
  --withdrawn research\aaai27-harness-frontier\metadata\withdrawn_records.json `
  --output research\aaai27-harness-frontier\metadata\final_corpus_status.json
```

## Important caveat

Most direct harness papers appeared in 2026 and many are preprints. “High
impact” therefore means centrality to the active research problem, strength of
the experimental collision, venue status where available, and likelihood of
shaping subsequent work. It does not mean mature citation count.
