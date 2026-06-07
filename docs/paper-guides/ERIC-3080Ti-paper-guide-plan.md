# ERIC-3080Ti Paper Guide Plan

> Branch: `ERIC-3080Ti/paper-guides`
>
> Date: 2026-06-07
>
> Goal: add a reproducible paper-reading layer to every learning topic.

## Scope

This repo has 46 learning topics. The paper-guide layer uses a staged but uniform structure:

```text
learning/<topic>/paper/
  README.md
  01_<paper_slug>.pdf
  guide_01_<paper_slug>.md
  guide_01_<paper_slug>.pdf
```

The first pass gives every topic one anchor paper or authoritative primary document. The anchor is the paper a beginner should read first before branching into the rest of the topic. Existing `papers/` folders are left untouched because some of them already contain older notes or PDFs with different conventions.

## Quality Bar

Each guide must answer:

1. What problem existed before this paper?
2. Why did this paper become important?
3. What are the core concepts a beginner must understand first?
4. What design choices did the authors make, and why are they plausible?
5. What math or objective function should be read slowly?
6. What experimental evidence supports the claim?
7. How does this connect to the local lecture/source code?
8. What should the learner be able to explain without notes?

The guides are written in Chinese, beginner-friendly, and intentionally avoid copying long passages from the original papers.

## 2026-06-07 Quality Correction

The first generated version was too short: it behaved like a reading route card instead of a paper-contained guide. The corrected standard is:

- A guide should reconstruct the paper's own story, not merely recommend how to read it.
- Reading the guide should be close to a serious first pass through the paper: background, contribution, method, math, experiment evidence, ablations, limitations, and local code mapping.
- A generated template is only a scaffold. It is useful for coverage, but it must not be treated as a finished deep guide unless the paper-specific content has been manually expanded and checked against the source PDF.

Guide quality tiers:

| Tier | Meaning | Current marker |
|---|---|---|
| `manual-deep-guide` | Paper-specific deep guide; acceptable as a serious first-read substitute, with original paper still kept for figures/tables/details. | `<!-- manual-deep-guide -->` |
| `generated-deep-v2` | Expanded scaffold with paper-specific manifest fields; useful as a reading starter but still needs per-paper manual enrichment. | no marker |
| `route-card` | Too brief; deprecated. | should not be generated anymore |

The first `manual-deep-guide` exemplar is:

- `learning/transformer-deep/paper/guide_01_attention_is_all_you_need.md`

Future manual enrichment should use that file's density as the baseline.

## Source Policy

Preferred sources:

- arXiv PDF or abstract pages.
- Conference official PDFs such as NeurIPS, USENIX, MLSys, ACL Anthology, OpenReview.
- Official vendor or project documents only for topics where the canonical source is a system manual rather than an academic paper, for example CUDA.

If a topic's best entry point is not a paper, the file is still stored under `paper/` but marked as an authoritative primary document.

## Execution

Run from repository root:

```powershell
.\.venv\Scripts\python.exe scripts\paper_guide_pipeline.py
```

Useful options:

```powershell
.\.venv\Scripts\python.exe scripts\paper_guide_pipeline.py --only transformer-deep
.\.venv\Scripts\python.exe scripts\paper_guide_pipeline.py --no-download
.\.venv\Scripts\python.exe scripts\paper_guide_pipeline.py --no-pdf
```

The script reads `docs/paper-guides/paper_manifest.json`, writes per-topic folders, downloads source PDFs when possible, converts guides with `pandoc`, and writes `docs/paper-guides/ERIC-3080Ti-paper-guide-report.md`.
