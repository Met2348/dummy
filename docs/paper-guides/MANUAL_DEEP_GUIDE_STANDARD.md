# Manual Deep Guide Standard

> This is the persistent writing memory for the `paper/guide_*.md` upgrade pass.
>
> Target branch: `ERIC-3080Ti/paper-guides`
>
> Exemplar: `learning/transformer-deep/paper/guide_01_attention_is_all_you_need.md`

## Goal

Each `manual-deep-guide` should be close to a serious first read of the paper. It is not a replacement for the source PDF's exact figures and tables, but it should let a beginner understand the paper's story, mechanism, math, experiments, code connection, and modern relevance before opening the original.

## Required Marker

Every finished manual guide must contain this marker near the title:

```html
<!-- manual-deep-guide -->
```

Generated scaffolds without this marker are not considered finished.

## Required Sections

Use paper-specific headings when useful, but every guide must cover these functions:

1. **Paper Positioning**
   - Why this paper/document mattered at publication time.
   - What practical bottleneck or research disagreement existed before it.
   - What changed after the paper became influential.

2. **Historical Context and Motivation**
   - Reconstruct the field's state before the paper.
   - Explain why the old solution was unsatisfactory.
   - Explain the authors' design motivation in plain language.

3. **Paper Structure Map**
   - Tell the reader how the original paper is organized.
   - Name the sections, figures, tables, or experiments worth reading slowly.

4. **Core Concepts**
   - Define every concept the paper depends on.
   - Explain concept relationships, not just vocabulary.
   - Include common beginner misconceptions.

5. **Method / System / Objective Deep Dive**
   - Explain the actual method, not only the headline idea.
   - If it is a model paper, include architecture flow.
   - If it is a training paper, include objective/loss, data, and optimizer/training loop.
   - If it is a systems paper, include workload, scheduler, memory/communication path, and metrics.
   - If it is an evaluation paper, include task construction, scoring, aggregation, and bias/validity issues.

6. **Diagrams**
   - Include PDF-friendly ASCII diagrams in fenced code blocks.
   - Prefer one method overview diagram and one detailed data/tensor/resource flow diagram.
   - Mermaid is optional in Markdown, but ASCII must be present because pandoc PDF should remain readable.

7. **Tensor / Data / Resource Shapes**
   - For ML model papers, include tensor shapes and where parameters live.
   - For RL/preference papers, include sample structure, logprob/reward/advantage tensors, and loss reduction.
   - For serving/infra papers, include request flow, memory budget, latency/throughput decomposition, and communication volume.
   - For eval/safety papers, include dataset item shape, label schema, judge input/output, and aggregation.

8. **Math and Theory**
   - Rewrite the key formula in guide notation.
   - Explain what each variable means.
   - Explain why the formula is designed that way.
   - State assumptions and where they can fail.

9. **Code Examples**
   - Include at least one minimal Python or pseudocode snippet.
   - Snippets should be runnable or directly translatable to this repo's `src/` files when possible.
   - The code should teach the paper's mechanism, not just import a library.

10. **Experiments and Evidence Chain**
    - Explain what the main result table proves.
    - Explain the strongest ablation.
    - Explain what the paper does not prove.
    - Mention key metrics, datasets, and baselines where available.

11. **Modern Meaning**
    - Explain why the paper still matters now.
    - Connect it to later work and current LLM practice.
    - Say whether it is a foundation, a stepping stone, a warning, or a superseded method.

12. **Local Repo Connection**
    - Link to lectures and source files.
    - Include one concrete 30-60 minute local experiment.
    - Include expected observations.

13. **Closed-book Mastery Check**
    - Include questions that force explanation, derivation, diagram drawing, or code mapping.
    - Avoid trivia-only questions.

## Length Target

- Normal anchor paper: 180-320 lines of Markdown.
- Survey or technical report: 240-420 lines.
- Small paper with simple mechanism: may be shorter, but must still include diagrams, code, theory, and evidence.

## Writing Rules

- Write in Chinese for the learner.
- Use English technical terms when they are standard in code or papers.
- Do not paste long passages from the source paper.
- Paraphrase figures/tables and cite them by number or role.
- Prefer concrete mechanisms over slogans.
- Treat "读完 guide 就大概读过论文" as the bar, while keeping the original PDF for exact wording, figures, and tables.

## PDF Rules

Each guide must be rendered to `guide_*.pdf` with `pandoc` + `xelatex`.

Visual acceptance:

- Chinese text is readable.
- Code blocks do not overflow badly.
- Tables fit or are converted to lists.
- No obvious blank or broken pages.
- Source PDF and guide PDF both live in the same `paper/` folder.

## Upgrade Workflow

For each remaining scaffold:

1. Extract source PDF text with `pdftotext` when possible.
2. Read abstract/introduction/method/main experiments/limitations from the extracted text.
3. Rewrite the guide by hand or with paper-specific notes.
4. Add `<!-- manual-deep-guide -->`.
5. Render the PDF.
6. Run a lightweight check:

```powershell
Select-String -Path learning\<topic>\paper\guide_*.md -Pattern "manual-deep-guide"
pdftoppm -png -f 1 -l 1 learning\<topic>\paper\guide_*.pdf tmp\pdfs\<topic>_guide
```

7. Update `docs/paper-guides/ERIC-3080Ti-paper-guide-report.md`.
