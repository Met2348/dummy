# Keynote-First Learning Index

Collected on 2026-06-11 for the "learn from the best people first" method.

The core idea is simple: keynote and invited speakers are a high-precision
filter. A normal top-conference paper can be noisy, but people repeatedly
invited by ACL, EMNLP, CoLM, ICLR, ICML, and NeurIPS usually have unusually good
problem taste, evidence discipline, and narrative control. Use this folder to
build your taste before you drown in random accepted papers.

## What Is Here

- `sources.csv`: source manifest with venue, year, URL, priority, and local path.
- `download_log.json`: audit trail for local snapshots.
- `source-pages/main/`: main-conference keynote and invited-talk pages.
- `source-pages/workshops/`: official workshop master lists.
- `source-pages/workshops/priority-pages/`: selected NLP/LLM workshop pages.
- `source-pages/people/`: researcher pages that should become paper-reading entry points.
- `fallback/`: public GitHub/community indexes for cases where official pages do
  not expose directly downloadable slides, PDFs, or reading lists.

One external page was not archived because Windows Defender flagged the HTML
snapshot. The URL is still retained in `sources.csv` and the skip note is in
`source-pages/workshops/priority-pages/colm_2025_visions_lm.skipped.txt`.

## Main Conference Sources

| Venue | Year | High-signal people / themes | Local snapshot | Source |
|---|---:|---|---|---|
| ACL | 2025 | Luke Zettlemoyer, Verena Rieser; pretraining, alignment, dialogue/NLG | `source-pages/main/acl_2025_keynotes.html` | https://2025.aclweb.org/program/keynotes/ |
| ACL | 2024 | Sunita Sarawagi; databases, ML, applied NLP | `source-pages/main/acl_2024_keynotes.html` | https://2024.aclweb.org/program/keynotes/ |
| EMNLP | 2025 | Heng Ji, Jana Diesner, Hannaneh Hajishirzi; information extraction, social systems, open LM research | `source-pages/main/emnlp_2025_keynotes.html` | https://2025.emnlp.org/program/keynotes/ |
| EMNLP | 2024 | Percy Liang, Anca Dragan; open foundation models, AI safety/alignment | `source-pages/main/emnlp_2024_keynotes.html` | https://2024.emnlp.org/program/keynotes/ |
| CoLM | 2025 | Luke Zettlemoyer, Nicholas Carlini; language model training and security | `source-pages/main/colm_2025_keynotes_panel.html` | https://colmweb.org/2025/plenary.html |
| CoLM | 2024 | Christopher Manning, Raquel Fernandez, Evelina Fedorenko, Hannaneh Hajishirzi | `source-pages/main/colm_2024_keynote_speakers.html` | https://colmweb.org/2024/Keynotes.html |
| ICLR | 2025 | Zico Kolter, Song-Chun Zhu, Yi Ma, Dawn Song, Danqi Chen; robust AI, AGI framing, academic LM training | `source-pages/main/iclr_2025_invited_talks.html` | https://iclr.cc/virtual/2025/events/invited%20talk |
| ICLR | 2024 | Kyunghyun Cho, Priya Donti, Kate Downing, Raia Hadsell, Devi Parikh, Jie Tang | `source-pages/main/iclr_2024_invited_talks.html` | https://iclr.cc/virtual/2024/events/invited%20talk |
| ICML | 2025 | Jon Kleinberg, Pamela Samuelson, Frauke Kreuter, Anca Dragan, Andreas Krause | `source-pages/main/icml_2025_invited_talks.html` | https://icml.cc/virtual/2025/events/invited%20talk |
| ICML | 2024 | Soumith Chintala, Vukosi Marivate, Chelsea Finn, Javier Duarte, Lucilla Sioli | `source-pages/main/icml_2024_invited_talks.html` | https://icml.cc/virtual/2024/events/invited%20talk |
| NeurIPS | 2025 | Richard Sutton, Zeynep Tufekci, Yejin Choi, Melanie Mitchell, Kyunghyun Cho, Andrew Saxe | `source-pages/main/neurips_2025_invited_talks.html` | https://neurips.cc/virtual/2025/eventlistwithbios/invited%20talk |
| NeurIPS | 2024 | Alison Gopnik, Sepp Hochreiter, Fei-Fei Li, Lidong Zhou, Arnaud Doucet, Julie Shah, Been Kim | `source-pages/main/neurips_2024_invited_talks.html` | https://neurips.cc/virtual/2024/eventlistwithbios/invited%20talk |

Supporting pages:

- ICLR 2025 blog: `source-pages/main/iclr_2025_keynote_blog.html`
- ICLR 2024 blog: `source-pages/main/iclr_2024_invited_speakers_blog.html`
- ICML 2025 downloads: `source-pages/main/icml_2025_downloads.html`
- ICML 2024 downloads: `source-pages/main/icml_2024_downloads.html`
- NeurIPS 2025 speaker lineup/topic posts:
  `source-pages/main/neurips_2025_speaker_lineup.html`,
  `source-pages/main/neurips_2025_invited_speaker_topics.html`
- NeurIPS 2024 newsletters:
  `source-pages/main/neurips_2024_august_newsletter.html`,
  `source-pages/main/neurips_2024_october_newsletter.html`

## Workshop Source Pages

Official workshop master lists:

| Venue | Year | Local snapshot | Source |
|---|---:|---|---|
| ACL | 2025 | `source-pages/workshops/acl_2025_workshops.html` | https://2025.aclweb.org/program/workshops/ |
| ACL | 2024 | `source-pages/workshops/acl_2024_workshops.html` | https://2024.aclweb.org/program/workshops/ |
| EMNLP | 2025 | `source-pages/workshops/emnlp_2025_workshops.html` | https://2025.emnlp.org/program/workshops/ |
| EMNLP | 2024 | `source-pages/workshops/emnlp_2024_workshops.html` | https://2024.emnlp.org/program/workshops/ |
| CoLM | 2025 | `source-pages/workshops/colm_2025_workshops.html` | https://colmweb.org/2025/workshops.html |
| ICLR | 2025 | `source-pages/workshops/iclr_2025_workshops.html` | https://iclr.cc/virtual/2025/events/workshop |
| ICLR | 2024 | `source-pages/workshops/iclr_2024_workshops.html` | https://iclr.cc/virtual/2024/events/workshop |
| ICML | 2025 | `source-pages/workshops/icml_2025_workshops.html` | https://icml.cc/virtual/2025/events/workshop |
| ICML | 2024 | `source-pages/workshops/icml_2024_workshops.html` | https://icml.cc/virtual/2024/events/workshop |
| NeurIPS | 2025 | `source-pages/workshops/neurips_2025_workshops.html` | https://neurips.cc/virtual/2025/events/workshop |
| NeurIPS | 2024 | `source-pages/workshops/neurips_2024_workshops.html` | https://neurips.cc/virtual/2024/events/workshop |

Priority NLP/LLM workshop pages already archived:

| Theme | Venue/year | Notable people / signals found | Local snapshot |
|---|---|---|---|
| Agent language models | ACL 2025 REALM | Chris Manning, Diyi Yang, Siva Reddy, Yu Su, Daniel Fried, Tao Yu | `source-pages/workshops/priority-pages/acl_2025_realm.html` |
| LLM security | ACL 2025 LLMSEC | Erick Galinkin, Niloofar Mireshghallah; agentic security/privacy | `source-pages/workshops/priority-pages/acl_2025_llmsec.html` |
| Argument mining | ACL 2025 | Andreas Vlachos; fact-checking as conversation | `source-pages/workshops/priority-pages/acl_2025_argmining.html` |
| Multilingual data quality | CoLM 2025 WMDQS | Julia Kreutzer, David Ifeoluwa Adelani, Sebastian Nagel | `source-pages/workshops/priority-pages/colm_2025_wmdqs.html` |
| Multilingual/equitable LM | CoLM 2025 MELT | Monojit Choudhury, Pedro Ortiz Suarez, Imanol Schlag, Shane Gu, Julia Kreutzer | `source-pages/workshops/priority-pages/colm_2025_melt_keynotes.html` |
| Test-time scaling/reasoning | CoLM 2025 ScalR | Aviral Kumar, Xuezhi Wang, Nathan Lambert, Lewis Tunstall, Azalia Mirhoseini, Eric Wallace | `source-pages/workshops/priority-pages/colm_2025_scalr.html` |
| Explainable reasoning/planning | CoLM 2025 XLLM | Greg Durrett, Yonatan Belinkov, Ana Marasovic, Huan Sun, Mark Riedl | `source-pages/workshops/priority-pages/colm_2025_xllm_reasoning_planning.html` |
| Pragmatic reasoning | CoLM 2025 PragLM | Vera Demberg, Michael Franke, Daniel Fried, Jennifer Hu | `source-pages/workshops/priority-pages/colm_2025_praglm.html` |
| Optimal reliance/accountability | CoLM 2025 ORIGen | Malihe Alikhani, Andreas Vlachos, Bertram Malle | `source-pages/workshops/priority-pages/colm_2025_origen_speakers.html` |

## People Entry

Teacher-mentioned entry:

| Person | Why track | Local snapshots |
|---|---|---|
| Xin (Eric) Wang | NLP + CV + multimodal/embodied agents; useful bridge if your future work moves toward multimodal agents | `source-pages/people/eric_xin_wang_homepage.html`, `source-pages/people/eric_xin_wang_ucsb_profile.html`, `source-pages/people/eric_xin_wang_openreview.html` |

## Suggested First Reading Order

Do not read randomly. Start with people whose work can teach research taste:

1. Luke Zettlemoyer: pretraining, data, architecture, self-supervision. Good for understanding how to ask "what should the model learn before the task?"
2. Danqi Chen: academic language-model training and retrieval/LM infrastructure. Good for learning how to do strong LM research under compute limits.
3. Percy Liang: open foundation models, benchmarks, reproducibility, and public research infrastructure.
4. Yejin Choi and Kyunghyun Cho: reasoning, commonsense, problem finding, and the habit of asking whether benchmarks reflect real intelligence.
5. Christopher Manning and Hannaneh Hajishirzi: NLP foundations plus modern LM systems; useful for connecting classical NLP, data, and current LLMs.
6. Nicholas Carlini, Dawn Song, Niloofar Mireshghallah: security, privacy, adversarial thinking; use these to prevent "capability-only" tunnel vision.
7. Xin (Eric) Wang: multimodal and embodied agents; use him as a bridge from NLP to CV/agents.

## How To Use AI Agents Correctly

The agent should not replace your memory. The agent should manufacture friction
at the right places so the knowledge enters your head.

For each keynote speaker, ask the agent to create a `speaker_card` with:

- Research identity: what question does this person repeatedly return to?
- Historical context: what was the field confused about before this line of work?
- Signature papers: 3 recent papers plus 1 older/root paper.
- Taste diagnosis: why this direction is better than a random accepted paper.
- Evidence chain: what datasets, baselines, ablations, and negative controls make the work convincing?
- Reproduction target: one tiny implementation or experiment you can finish in 1-2 days.

For each paper, force the agent into this structure:

- "Explain the story before the method": old paradigm, bottleneck, and why the question mattered then.
- "Explain the method at tensor level": shapes, objectives, losses, sampling/training loops.
- "Trace every claim to evidence": which table/figure/ablation proves which claim?
- "Give me a failure-mode critique": when would this method not work?
- "Quiz me": 12 closed-book questions, then grade my answer harshly.
- "Make me implement": a minimal notebook that reproduces the central mechanism, not the full SOTA result.

Bad agent usage:

- Asking only "summarize this paper".
- Accepting uncited paper lists.
- Reading an AI guide without opening the original abstract, figures, and experiment tables.
- Letting the agent produce polished prose before you can explain the idea verbally.
- Skipping the reproduction step because the paper is "too big".

Good agent usage:

- Ask for one-sentence thesis, then one-paragraph story, then full technical derivation.
- Ask for an "evidence map" from each paper claim to its experiment.
- Ask the agent to create flashcards and quiz you the next day.
- Ask for a deliberately simple PyTorch/Jupyter reproduction.
- Ask the agent to compare a keynote speaker's paper with a mediocre accepted paper and explain the difference in problem taste.

## Weekly Rhythm

Use a 5-day cycle:

1. Monday: choose one speaker from the main table; read the keynote abstract, bio, and 2-3 recent paper abstracts.
2. Tuesday: read one signature paper deeply; build the story, method, evidence map, and open questions.
3. Wednesday: implement a toy version or reproduce one table/figure at small scale.
4. Thursday: write a one-page "taste memo": why this was keynote-level work, what problem-selection move was special, and what you would imitate.
5. Friday: closed-book recall with an agent; if you cannot explain it without notes, reread the original figure/table instead of asking for another summary.

## What To Look For In A "Beautiful" Paper

- The problem is framed so clearly that the method feels inevitable.
- The contribution changes how people name or measure a problem, not just one benchmark score.
- The method has a small number of moving parts but explains many observations.
- The experiments are not just leaderboard wins; they isolate why the method works.
- The ablations are honest enough to expose the real causal mechanism.
- The writing teaches you the field's map, not only the authors' result.

This is the reason to start from keynote people. You are not only collecting
papers; you are training your taste function.

## Fallback Search Rule

When an official keynote/workshop page does not provide a direct PDF, slide
deck, video, or reading list, use this order:

1. Search the speaker's homepage and lab page.
2. Search the conference/workshop official page and virtual-event page.
3. Search GitHub/community lists, but mark them as non-official.
4. Search OpenReview, ACL Anthology, Semantic Scholar, and Google Scholar for the speaker's recent representative papers.
5. Only keep materials that are clearly public; do not archive private, paywalled, or login-gated content.

The current public fallback set is in `fallback/README.md`. It includes GitHub
indexes such as `qingsongedu/awesome-AI-tutorials-surveys`,
`qingsongedu/Talks-Keynotes-slides`, `soulbliss/NLP-conference-compendium`,
`AmberLJC/LLMSys-PaperList`, and related topic lists for causal NLP,
social-good NLP, knowledge graphs, best papers, and LLM optimization.
